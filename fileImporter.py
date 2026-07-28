import pandas as pd
import mysql.connector

# ---------------- Database ----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_PASSWORD_HERE",  # Replace with your actual password
    "database": "hpcl"
}

TABLE_NAME = "SNEHA"

# ---------------- Read Excel ----------------
file_path = r"C:\Users\sneha\OneDrive\Desktop\HPCL\project 3\SNEHA.xlsx"

df = pd.read_excel(
    file_path,
    dtype=str
)

# Replace NaN with None
df = df.where(pd.notnull(df), None)

# ---------------- Connect ----------------
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# ---------------- Create Table ----------------
# ---------------- Create Table ----------------
cursor.execute(f"""
CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (

`material_code` VARCHAR(100),
`SHORT DESC` TEXT,
`LONG DESC` LONGTEXT,
`matl_group` TEXT,
`sbu_owner` TEXT,
`po_count` TEXT,
`activity_status` TEXT,
`zmatcode` TEXT,
`zmatcode_status` TEXT,
`cluster_id` TEXT,
`cluster_keep_status` TEXT,
`stage1_flag` TEXT,
`is_canonical` TEXT,
`remediation_decision` TEXT,
`Material type` TEXT,

`Critical Gaps` LONGTEXT NULL,
`Missing Information for Bidding` LONGTEXT NULL,
`Missing information for Execution` LONGTEXT NULL,
`Ambiguities` LONGTEXT NULL,
`Overall Assessment` LONGTEXT NULL,
`Recommended Improvement` LONGTEXT NULL,

`processing_status` ENUM(
    'PENDING',
    'PROCESSING',
    'DONE',
    'FAILED'
) NOT NULL DEFAULT 'PENDING',

INDEX idx_status_material (`processing_status`, `material_code`)

)
CHARACTER SET utf8mb4
COLLATE utf8mb4_0900_ai_ci;
""")

# ---------------- Insert ----------------
insert_sql = f"""
INSERT INTO `{TABLE_NAME}` (

`material_code`,
`SHORT DESC`,
`LONG DESC`,
`matl_group`,
`sbu_owner`,
`po_count`,
`activity_status`,
`zmatcode`,
`zmatcode_status`,
`cluster_id`,
`cluster_keep_status`,
`stage1_flag`,
`is_canonical`,
`remediation_decision`,
`Material type`

)

VALUES (
%s,%s,%s,%s,%s,
%s,%s,%s,%s,%s,
%s,%s,%s,%s,%s
)
"""

rows = list(df.itertuples(index=False, name=None))

cursor.executemany(insert_sql, rows)

conn.commit()

print(f"Imported {cursor.rowcount} rows.")

cursor.close()
conn.close()

print("Done.")