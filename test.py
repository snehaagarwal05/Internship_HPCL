from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Any

import mysql.connector
from mysql.connector import Error as MySQLError

from google import genai
from google.genai import types
from google.genai.errors import APIError

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (single source of truth — edit only this section)
# ─────────────────────────────────────────────────────────────────────────────

MODEL                   = "gemini-3.1-flash-lite"
MAX_RETRIES             = 3          # JSON-validity retries per chunk
API_MAX_RETRIES         = 3          # Transient API-error retries
API_RETRY_DELAY_SECONDS = 5          # Base backoff (exponential: delay * attempt)
SLEEP_BETWEEN_CALLS     = 2          # Pause between retry attempts on same chunk
CHUNK_SIZE              = 5         # Rows per fetch/AI call cycle

# How long (minutes) a row may stay in PROCESSING before being reset to PENDING.
# Catches instances that crashed mid-batch without cleaning up.
STALE_PROCESSING_MINUTES = 20

# Database credentials — prefer environment variables over hard-coded values.
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_PASSWORD_HERE",     
    "database": "hpcl",
}

TABLE_NAME = "SNEHA"

# Column name constants
COL_MATERIAL_CODE       = "material_code"
COL_SHORT_DESC          = "SHORT DESC"
COL_LONG_DESC           = "LONG DESC"
COL_CRITICAL_GAPS       = "Critical Gaps"
COL_MISSING_BIDDING     = "Missing Information for Bidding"
COL_MISSING_EXECUTION   = "Missing information for Execution"
COL_AMBIGUITIES         = "Ambiguities"
COL_OVERALL_ASSESSMENT  = "Overall Assessment"
COL_RECOMMENDED_IMPROVE = "Recommended Improvement"
COL_STATUS              = "processing_status"   # NEW concurrency-safety column

# Allowed values for processing_status
STATUS_PENDING    = "PENDING"
STATUS_PROCESSING = "PROCESSING"
STATUS_DONE       = "DONE"
STATUS_FAILED     = "FAILED"

# Gemini client
client = genai.Client(
    api_key="YOUR_API_KEY_HERE",  # Replace with your actual API key
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("procurement")
    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))

    # File handler — DEBUG and above
    fh = logging.FileHandler("processing.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

_shutdown_requested = False

def _handle_sigint(signum, frame):
    global _shutdown_requested
    log.warning("Shutdown signal received — will stop after the current batch.")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _handle_sigint)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an industrial procurement specification auditor.

Your input is one or more procurement specifications in a single message,
each introduced by its own "Specification N" header. Treat every
specification COMPLETELY INDEPENDENTLY — never let details, context, or
terminology from one specification leak into another.

Your only output is a single valid JSON array. The array must contain
EXACTLY one object per specification given, IN THE SAME ORDER they were
given (the object for "Specification 1" comes first, and so on). Each
object must match the schema below exactly. Do not output anything outside
the JSON array — no prose, no markdown fences, no wrapping object, and no
"batch_index" or similar field inside the objects (ordering alone identifies
which specification each object belongs to).

─────────────────────────────────────────
RULES (non-negotiable)
─────────────────────────────────────────
1. Base every field value strictly on what the specification EXPLICITLY states.
2. If something is not stated, write "Not specified." — never invent a value.
3. Never invent standard numbers (ASTM, ISO, API, IEC, BIS, EN, DIN, etc.).
   If a test method or standard is not cited in the specification, write "Not specified."
4. Never state that something is outdated, obsolete, non-compliant, or incorrect
   unless the specification itself makes that contradiction explicit.
5. Never add procurement or legal assumptions (e.g. "this conflicts with regulations").
   Observations must be factual and evidence-based.
6. Only flag ambiguities that are genuinely present in the text
   (missing units, undefined abbreviations, contradictory values).
   Do not fabricate ambiguities.
7. Only flag missing items that are genuinely relevant to the identified product type.
   Do not apply mechanical-equipment checks to chemicals and vice versa.
8. All arrays use the same key names defined in the schema. No extra keys. No renamed keys.

─────────────────────────────────────────
TASK SEQUENCE — apply Steps 1–8 separately to EACH specification
─────────────────────────────────────────
Step 1 — Identify the item.
  Determine item_category, product_name, and likely_application
  (application only if clearly inferable; otherwise "Not specified.").
  Also write short_description: ONE plain-language sentence that summarises
  what this item is and what it's for, in everyday words, so someone scanning
  many items at once can understand it without reading the raw spec text.
  Base it only on what is stated — do not invent details.

Step 2 — Extract covered information.
  List every technical and commercial parameter explicitly stated.
  Do not infer. Do not paraphrase beyond normalisation.

Step 3 — Identify missing information for bidding.
  List only parameters a bidder genuinely needs that are absent.
  Omit anything not applicable to this product type.

Step 4 — Identify missing information for execution.
  List only documents or operational requirements applicable to this product type
  that are absent (e.g. CoA, MTC, FAT, calibration certificate, SDS where relevant).

Step 5 — Identify ambiguities.
  Quote the exact ambiguous text. Explain why it is ambiguous.
  Return [] if none exist.

Step 6 — Identify top gaps.
  Select the 3–7 most procurement-critical gaps and rank them 1 (highest) downward.

Step 7 — Assign risk_level and tender_readiness.
  risk_level: "Low" | "Medium" | "High"
  tender_readiness: "Ready" | "Conditional" | "Not Ready"

Step 8 — Recommended additions.
  List the additions most likely to resolve the top gaps, ranked by priority.

─────────────────────────────────────────
OUTPUT SCHEMA — a JSON ARRAY of objects, one per specification (exact — no extra keys per object)
─────────────────────────────────────────
[
  {
    "item_category": "",
    "product_name": "",
    "likely_application": "",
    "short_description": "",
    "covered_information": [{"label": "", "value": ""}],
    "top_gaps": [{"priority": 1, "gap": "", "reason": ""}],
    "missing_for_bidding": [{"label": "", "reason": ""}],
    "missing_for_execution": [{"item": "", "reason": ""}],
    "key_ambiguities": [{"quote": "", "reason": ""}],
    "risk_level": "",
    "tender_readiness": "",
    "recommended_additions": [{"priority": 1, "addition": ""}]
  }
]
"""

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DEFINITION  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA: dict[str, tuple[type, Any]] = {
    "item_category":         (str,  ""),
    "product_name":          (str,  ""),
    "likely_application":    (str,  ""),
    "short_description":     (str,  ""),
    "covered_information":   (list, []),
    "top_gaps":              (list, []),
    "missing_for_bidding":   (list, []),
    "missing_for_execution": (list, []),
    "key_ambiguities":       (list, []),
    "risk_level":            (str,  ""),
    "tender_readiness":      (str,  ""),
    "recommended_additions": (list, []),
}

ARRAY_SCHEMAS: dict[str, dict[str, Any]] = {
    "covered_information":   {"label": "", "value": ""},
    "top_gaps":              {"priority": 0, "gap": "", "reason": ""},
    "missing_for_bidding":   {"label": "", "reason": ""},
    "missing_for_execution": {"item": "", "reason": ""},
    "key_ambiguities":       {"quote": "", "reason": ""},
    "recommended_additions": {"priority": 0, "addition": ""},
}

VALID_RISK_LEVELS = {"Low", "Medium", "High"}
VALID_READINESS   = {"Ready", "Conditional", "Not Ready"}

REQUIRED_KEYS = set(SCHEMA.keys())

# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_array(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in model output.")
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    raise ValueError("Incomplete JSON array in model output.")

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA VALIDATION AND AUTO-REPAIR  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_array_item(item: Any, required_keys: dict[str, Any]) -> dict:
    if not isinstance(item, dict):
        first_key = next(iter(required_keys))
        item = {first_key: str(item)}
    ALIASES = {
        "explanation": "reason", "description": "reason",
        "details": "reason", "note": "reason", "gap_description": "gap",
    }
    for old, new in ALIASES.items():
        if old in item and new not in item:
            item[new] = item.pop(old)
    for key, default in required_keys.items():
        if key not in item:
            item[key] = default
    return {k: v for k, v in item.items() if k in required_keys}


def validate_and_repair(data: dict) -> dict:
    repaired: dict[str, Any] = {}
    for key, (expected_type, default) in SCHEMA.items():
        if key not in data:
            log.debug("[repair] Missing key '%s' — using default.", key)
            repaired[key] = default
        elif not isinstance(data[key], expected_type):
            log.debug("[repair] Key '%s' wrong type (%s vs %s) — using default.",
                      key, type(data[key]).__name__, expected_type.__name__)
            repaired[key] = default
        else:
            repaired[key] = data[key]
    unexpected = set(data.keys()) - set(SCHEMA.keys())
    if unexpected:
        log.debug("[repair] Removing unexpected top-level keys: %s", unexpected)
    for field, required_keys in ARRAY_SCHEMAS.items():
        raw_list = repaired.get(field, [])
        if not isinstance(raw_list, list):
            log.debug("[repair] Field '%s' is not a list — resetting to [].", field)
            repaired[field] = []
            continue
        repaired[field] = [_normalise_array_item(item, required_keys) for item in raw_list]
    if repaired["risk_level"] not in VALID_RISK_LEVELS:
        raw = repaired["risk_level"]
        match = next((v for v in VALID_RISK_LEVELS if v.lower() == raw.lower()), None)
        repaired["risk_level"] = match or "Medium"
        if not match:
            log.debug("[repair] Invalid risk_level '%s' — defaulting to 'Medium'.", raw)
    if repaired["tender_readiness"] not in VALID_READINESS:
        raw = repaired["tender_readiness"]
        match = next((v for v in VALID_READINESS if v.lower() == raw.lower()), None)
        repaired["tender_readiness"] = match or "Conditional"
        if not match:
            log.debug("[repair] Invalid tender_readiness '%s' — defaulting to 'Conditional'.", raw)
    return repaired

# ─────────────────────────────────────────────────────────────────────────────
# PRE-WRITE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_result_for_db(result: dict) -> tuple[bool, str]:
    """Return (True, '') if result is safe to write; (False, reason) otherwise."""
    missing = REQUIRED_KEYS - set(result.keys())
    if missing:
        return False, f"Missing required keys: {missing}"
    if result.get("risk_level") not in VALID_RISK_LEVELS:
        return False, f"Invalid risk_level: {result.get('risk_level')!r}"
    if result.get("tender_readiness") not in VALID_READINESS:
        return False, f"Invalid tender_readiness: {result.get('tender_readiness')!r}"
    return True, ""

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CALLS  (UNCHANGED logic, logging added)
# ─────────────────────────────────────────────────────────────────────────────

def build_batch_prompt(specifications: list[str]) -> str:
    blocks = [
        f"Specification {i}\n{'-' * 16}\n{spec.strip()}"
        for i, spec in enumerate(specifications, start=1)
    ]
    return (
        "Analyze the following procurement specifications independently.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nReturn ONLY a JSON array with exactly {len(specifications)} "
        + "objects, one per specification above, in the same order."
    )


def call_model(specifications: list[str], think: bool, stats: dict) -> str:
    config_kwargs: dict[str, Any] = {
        "temperature": 0,
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
    }
    if think:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="high")
    config = types.GenerateContentConfig(**config_kwargs)
    prompt = build_batch_prompt(specifications)

    for api_attempt in range(1, API_MAX_RETRIES + 1):
        try:
            stats["api_calls"] += 1
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config,
            )
            return response.text or ""
        except APIError as exc:
            delay = API_RETRY_DELAY_SECONDS * api_attempt  # exponential backoff
            log.warning("API error (%s) — attempt %d/%d, retrying in %ds.",
                        exc, api_attempt, API_MAX_RETRIES, delay)
            if api_attempt == API_MAX_RETRIES:
                raise
            time.sleep(delay)
    return ""


def run_with_retries(specifications: list[str], stats: dict) -> list[dict]:
    expected_len = len(specifications)
    for attempt in range(1, MAX_RETRIES + 1):
        use_think = attempt > 1
        log.debug("[attempt %d/%d] think=%s (chunk of %d)",
                  attempt, MAX_RETRIES, use_think, expected_len)

        raw = call_model(specifications, think=use_think, stats=stats)

        if not raw.strip():
            log.warning("Empty response on attempt %d — retrying.", attempt)
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        try:
            array_str = extract_json_array(raw)
            parsed = json.loads(array_str)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("JSON parse error on attempt %d: %s", attempt, exc)
            if attempt == MAX_RETRIES:
                log.error("All attempts failed. Raw output:\n%s", raw)
                raise SystemExit(1)
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        if not isinstance(parsed, list):
            log.warning("Expected array, got %s on attempt %d.", type(parsed).__name__, attempt)
            if attempt == MAX_RETRIES:
                raise SystemExit(1)
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        if len(parsed) != expected_len:
            log.warning("Array length mismatch: got %d, expected %d on attempt %d.",
                        len(parsed), expected_len, attempt)
            if attempt == MAX_RETRIES:
                raise SystemExit(1)
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue

        repaired = [
            validate_and_repair(item if isinstance(item, dict) else {})
            for item in parsed
        ]
        log.debug("Chunk parsed and repaired successfully.")
        return repaired

    raise SystemExit(1)


def run_batch(specifications: list[str], start_index: int, stats: dict) -> list[dict]:
    total = len(specifications)
    results: list[dict] = [None] * total  # type: ignore[list-item]

    to_send: list[tuple[int, str]] = []
    for pos, spec in enumerate(specifications):
        idx = start_index + pos
        if not spec.strip():
            log.info("Spec %d is empty, skipping.", idx)
            results[pos] = {"batch_index": idx, "error": "Empty specification text."}
        else:
            to_send.append((pos, spec))

    if not to_send:
        return results

    log.info("Sending %d spec(s) to model (1 API call).", len(to_send))
    try:
        specs_text = [spec for _, spec in to_send]
        chunk_results = run_with_retries(specs_text, stats)
        for (pos, _), item in zip(to_send, chunk_results):
            results[pos] = {"batch_index": start_index + pos, **item}
    except SystemExit:
        log.error("Chunk failed all attempts — recording error for %d spec(s).", len(to_send))
        for pos, _ in to_send:
            results[pos] = {
                "batch_index": start_index + pos,
                "error": "Failed to get valid JSON after all retries.",
            }
    except APIError as exc:
        log.error("Unrecoverable API error — recording error for %d spec(s): %s", len(to_send), exc)
        for pos, _ in to_send:
            results[pos] = {"batch_index": start_index + pos, "error": f"API error: {exc}"}

    return results

# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def format_top_gaps(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        label = str(item.get("label", "")).strip()
        reason = str(item.get("reason", "")).strip()

        lines.append(f"{i}. {label}\n{reason}")
    return "\n\n".join(lines)

def format_missing_for_bidding(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        label = str(item.get("label", "")).strip()
        reason = str(item.get("reason", "")).strip()

        lines.append(f"{i}. {label}\n{reason}")
    return "\n\n".join(lines)

def format_missing_for_execution(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        label = str(item.get("label", "")).strip()
        reason = str(item.get("reason", "")).strip()

        lines.append(f"{i}. {label}\n{reason}")
    return "\n\n".join(lines)

def format_key_ambiguities(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        lines.append(f'{i}. "{(item.get("quote") or "").strip()}"\n{(item.get("reason") or "").strip()}')
    return "\n\n".join(lines)

def format_overall_assessment(result: dict) -> str:
    return (
        f"Risk Level: {result.get('risk_level', '')}\n"
        f"Tender Readiness: {result.get('tender_readiness', '')}\n"
        f"Item Category: {result.get('item_category', '')}\n"
        f"Product Name: {result.get('product_name', '')}\n"
        f"Likely Application: {result.get('likely_application', '')}"
    )

def format_recommended_additions(items: list[dict]) -> str:
    return "\n".join(
        f"{i}. {(item.get('addition') or '').strip()}"
        for i, item in enumerate(items, start=1)
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


# ── Startup checks ────────────────────────────────────────────────────────────

def startup_checks() -> None:
    """Verify DB connection, table existence, and required columns. Exit on failure."""
    log.info("Running startup checks…")
    try:
        conn = get_connection()
    except MySQLError as exc:
        log.critical("Cannot connect to database: %s", exc)
        sys.exit(1)

    required_columns = {
        COL_MATERIAL_CODE, COL_SHORT_DESC, COL_LONG_DESC,
        COL_CRITICAL_GAPS, COL_MISSING_BIDDING, COL_MISSING_EXECUTION,
        COL_AMBIGUITIES, COL_OVERALL_ASSESSMENT, COL_RECOMMENDED_IMPROVE,
        COL_STATUS,
    }

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (DB_CONFIG["database"], TABLE_NAME),
        )
        (count,) = cursor.fetchone()
        if count == 0:
            log.critical("Table '%s' does not exist in database '%s'.",
                         TABLE_NAME, DB_CONFIG["database"])
            sys.exit(1)

        cursor.execute(f"SHOW COLUMNS FROM `{TABLE_NAME}`")
        existing = {row[0] for row in cursor.fetchall()}
        missing = required_columns - existing
        if missing:
            log.critical(
                "Required column(s) missing from table '%s': %s\n"
                "Create the missing column(s) before running, e.g.:\n"
                "  ALTER TABLE `%s` ADD COLUMN `processing_status` "
                "ENUM('PENDING','PROCESSING','DONE','FAILED') NOT NULL DEFAULT 'PENDING';",
                TABLE_NAME, missing, TABLE_NAME,
            )
            sys.exit(1)

        cursor.close()
    finally:
        conn.close()

    log.info("Startup checks passed.")


# ── Stale-row recovery ────────────────────────────────────────────────────────

def reset_stale_rows(conn) -> int:
    """
    Reset rows stuck in PROCESSING for longer than STALE_PROCESSING_MINUTES
    back to PENDING. Returns the number of rows reset.
    This handles crashed instances that never cleaned up their claimed rows.
    """
    # We use a dedicated updated_at column if available; otherwise we rely on
    # the caller having set the status recently enough. Here we assume the
    # table has a `processing_claimed_at` DATETIME column updated at claim time.
    # If you don't have that column, replace the WHERE clause with a broader
    # "all PROCESSING rows" reset and accept that a briefly-paused live instance
    # could lose its claim. A dedicated timestamp column is the safest approach.
    cutoff = datetime.now() - timedelta(minutes=STALE_PROCESSING_MINUTES)
    query = (
        f"UPDATE `{TABLE_NAME}` "
        f"SET `{COL_STATUS}` = %s "
        f"WHERE `{COL_STATUS}` = %s "
        f"  AND `processing_claimed_at` <= %s"
    )
    cursor = conn.cursor()
    try:
        cursor.execute(query, (STATUS_PENDING, STATUS_PROCESSING, cutoff))
        count = cursor.rowcount
        conn.commit()
        if count:
            log.info("Reset %d stale PROCESSING row(s) back to PENDING.", count)
        return count
    except MySQLError as exc:
        # If the column doesn't exist, fall back to resetting ALL PROCESSING rows.
        # This is less precise but safe for single-instance or dev environments.
        log.warning("Could not reset by timestamp (%s) — resetting ALL PROCESSING rows.", exc)
        conn.rollback()
        cursor2 = conn.cursor()
        cursor2.execute(
            f"UPDATE `{TABLE_NAME}` SET `{COL_STATUS}` = %s WHERE `{COL_STATUS}` = %s",
            (STATUS_PENDING, STATUS_PROCESSING),
        )
        count = cursor2.rowcount
        conn.commit()
        cursor2.close()
        if count:
            log.info("Reset %d PROCESSING row(s) back to PENDING (fallback).", count)
        return count
    finally:
        cursor.close()


# ── Row claiming ──────────────────────────────────────────────────────────────

def claim_batch(conn, limit: int = CHUNK_SIZE) -> list[dict]:
    """
    Atomically select PENDING rows and mark them PROCESSING in a single
    transaction. Returns the claimed rows, or [] if none are available.

    Ordering by material_code ensures multiple instances work the same queue
    from the front, reducing the chance of two instances selecting the same
    row in a race before the UPDATE fires.
    """
    cursor = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()

        # SELECT … FOR UPDATE locks the rows until we commit, preventing
        # another instance from claiming the same rows simultaneously.
        cursor.execute(
            f"SELECT `{COL_MATERIAL_CODE}`, `{COL_SHORT_DESC}`, `{COL_LONG_DESC}` "
            f"FROM `{TABLE_NAME}` "
            f"WHERE `{COL_STATUS}` = %s "
            f"ORDER BY `{COL_MATERIAL_CODE}` "
            f"LIMIT %s "
            f"FOR UPDATE",
            (STATUS_PENDING, limit),
        )
        rows = cursor.fetchall()

        if not rows:
            conn.rollback()
            return []

        codes = [row[COL_MATERIAL_CODE] for row in rows]
        placeholders = ", ".join(["%s"] * len(codes))
        now = datetime.now()

        # Try to update claimed_at if the column exists; ignore if it doesn't.
        try:
            cursor.execute(
                f"UPDATE `{TABLE_NAME}` "
                f"SET `{COL_STATUS}` = %s, `processing_claimed_at` = %s "
                f"WHERE `{COL_MATERIAL_CODE}` IN ({placeholders})",
                [STATUS_PROCESSING, now] + codes,
            )
        except MySQLError:
            cursor.execute(
                f"UPDATE `{TABLE_NAME}` "
                f"SET `{COL_STATUS}` = %s "
                f"WHERE `{COL_MATERIAL_CODE}` IN ({placeholders})",
                [STATUS_PROCESSING] + codes,
            )

        claimed = cursor.rowcount
        if claimed != len(codes):
            log.error("Claim mismatch: tried to claim %d rows, updated %d — rolling back.",
                      len(codes), claimed)
            conn.rollback()
            return []

        conn.commit()
        log.info("Claimed %d row(s) for processing.", claimed)
        return rows

    except MySQLError as exc:
        conn.rollback()
        log.error("Failed to claim batch: %s", exc)
        return []
    finally:
        cursor.close()


def build_spec_text(row: dict) -> str:
    long_desc = (row.get(COL_LONG_DESC) or "").strip()
    if long_desc:
        return long_desc
    return (row.get(COL_SHORT_DESC) or "").strip()


# ── Result writing ────────────────────────────────────────────────────────────

def mark_rows_failed(conn, codes: list[Any]) -> None:
    if not codes:
        return
    placeholders = ", ".join(["%s"] * len(codes))
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"UPDATE `{TABLE_NAME}` SET `{COL_STATUS}` = %s "
            f"WHERE `{COL_MATERIAL_CODE}` IN ({placeholders})",
            [STATUS_FAILED] + codes,
        )
        conn.commit()
        log.info("Marked %d row(s) as FAILED.", cursor.rowcount)
    except MySQLError as exc:
        conn.rollback()
        log.error("Could not mark rows as FAILED: %s", exc)
    finally:
        cursor.close()


def update_row(conn, material_code: Any, result: dict, current_short_desc: str | None) -> bool:
    """
    Write one AI result back to its row keyed by material_code.
    Returns True on success, False on failure.
    Verifies rowcount == 1 after the UPDATE.
    """
    set_clauses = [
        f"`{COL_CRITICAL_GAPS}` = %s",
        f"`{COL_MISSING_BIDDING}` = %s",
        f"`{COL_MISSING_EXECUTION}` = %s",
        f"`{COL_AMBIGUITIES}` = %s",
        f"`{COL_OVERALL_ASSESSMENT}` = %s",
        f"`{COL_RECOMMENDED_IMPROVE}` = %s",
        f"`{COL_STATUS}` = %s",
    ]
    params: list[Any] = [
        format_top_gaps(result.get("top_gaps", [])),
        format_missing_for_bidding(result.get("missing_for_bidding", [])),
        format_missing_for_execution(result.get("missing_for_execution", [])),
        format_key_ambiguities(result.get("key_ambiguities", [])),
        format_overall_assessment(result),
        format_recommended_additions(result.get("recommended_additions", [])),
        STATUS_DONE,
    ]

    if not (current_short_desc or "").strip():
        set_clauses.append(f"`{COL_SHORT_DESC}` = %s")
        params.append(result.get("short_description", ""))

    params.append(material_code)
    query = (
        f"UPDATE `{TABLE_NAME}` SET {', '.join(set_clauses)} "
        f"WHERE `{COL_MATERIAL_CODE}` = %s"
    )
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if cursor.rowcount != 1:
            log.error("UPDATE for material %s affected %d rows (expected 1).",
                      material_code, cursor.rowcount)
            return False
        return True
    except MySQLError as exc:
        log.error("DB error updating material %s: %s", material_code, exc)
        return False
    finally:
        cursor.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DRIVER LOOP
# ─────────────────────────────────────────────────────────────────────────────

def process_all_records() -> None:
    startup_checks()

    stats = {
        "api_calls":       0,
        "total_processed": 0,
        "total_skipped":   0,
        "total_failed":    0,
        "batch_times":     [],
    }

    run_start = time.monotonic()
    conn = get_connection()
    batch_num = 0

    try:
        reset_stale_rows(conn)

        while not _shutdown_requested:
            batch_start = time.monotonic()
            batch_num += 1

            rows = claim_batch(conn, limit=CHUNK_SIZE)
            if not rows:
                log.info("No more PENDING rows — processing complete.")
                break

            log.info("Batch %d: claimed %d row(s).", batch_num, len(rows))

            material_codes: list[tuple[Any, str | None]] = []
            specs: list[str] = []
            skip_codes: list[Any] = []

            for row in rows:
                code = row[COL_MATERIAL_CODE]
                spec_text = build_spec_text(row)
                if not spec_text:
                    skip_codes.append(code)
                else:
                    material_codes.append((code, row.get(COL_SHORT_DESC)))
                    specs.append(spec_text)

            # Mark no-description rows as FAILED (not DONE — they need attention).
            if skip_codes:
                log.info("Skipping %d row(s) with no description text: %s",
                         len(skip_codes), skip_codes)
                mark_rows_failed(conn, skip_codes)
                stats["total_skipped"] += len(skip_codes)

            if not specs:
                log.info("Batch %d: no specs to send — moving on.", batch_num)
                continue

            # ── AI processing ──────────────────────────────────────────────
            try:
                batch_results = run_batch(specs, start_index=1, stats=stats)
            except Exception as exc:  # noqa: BLE001
                log.error("Batch %d AI call failed entirely: %s", batch_num, exc)
                failed_codes = [c for c, _ in material_codes]
                mark_rows_failed(conn, failed_codes)
                stats["total_failed"] += len(material_codes)
                continue

            # ── Write results ──────────────────────────────────────────────
            success_codes: list[Any] = []
            failure_codes: list[Any] = []

            for (code, current_short_desc), result in zip(material_codes, batch_results):
                if "error" in result:
                    log.warning("Skipping update for material %s: %s", code, result["error"])
                    failure_codes.append(code)
                    stats["total_failed"] += 1
                    continue

                ok, reason = validate_result_for_db(result)
                if not ok:
                    log.error("DB validation failed for material %s: %s — skipping write.", code, reason)
                    failure_codes.append(code)
                    stats["total_failed"] += 1
                    continue

                if update_row(conn, code, result, current_short_desc):
                    success_codes.append(code)
                    stats["total_processed"] += 1
                    log.info("Updated material: %s", code)
                else:
                    failure_codes.append(code)
                    stats["total_failed"] += 1

            # Commit successful updates; mark failures explicitly.
            if success_codes:
                try:
                    conn.commit()
                    log.info("Committed %d row(s) in batch %d.", len(success_codes), batch_num)
                except MySQLError as exc:
                    log.error("Commit failed for batch %d — rolling back: %s", batch_num, exc)
                    conn.rollback()
                    # On rollback the rows are still PROCESSING; reset them.
                    mark_rows_failed(conn, success_codes)
                    stats["total_failed"] += len(success_codes)
                    stats["total_processed"] -= len(success_codes)

            if failure_codes:
                mark_rows_failed(conn, failure_codes)

            elapsed = time.monotonic() - batch_start
            stats["batch_times"].append(elapsed)
            log.info("Batch %d finished in %.1fs.", batch_num, elapsed)

        if _shutdown_requested:
            log.warning("Shutdown requested — exiting after batch %d.", batch_num)

    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error in main loop: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    total_elapsed = time.monotonic() - run_start
    avg_batch = (
        sum(stats["batch_times"]) / len(stats["batch_times"])
        if stats["batch_times"] else 0.0
    )

    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info("  Total processed : %d", stats["total_processed"])
    log.info("  Total skipped   : %d", stats["total_skipped"])
    log.info("  Total failed    : %d", stats["total_failed"])
    log.info("  Total API calls : %d", stats["api_calls"])
    log.info("  Avg batch time  : %.1fs", avg_batch)
    log.info("  Total run time  : %.1fs", total_elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    process_all_records()