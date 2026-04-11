import sqlite3
import pandas as pd
import re

class SQLExecutor:
    def __init__(self, db_path):
        self.db_path = db_path

    def execute(self, sql_query):
        # 1. Remove comment lines (lines that start with --)
        lines = sql_query.strip().split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('--'):
                clean_lines.append(line)
        sql_clean = ' '.join(clean_lines).strip()

        # 2. Only take the first statement (before the semicolon).
        if ';' in sql_clean:
            sql_clean = sql_clean.split(';')[0].strip()

        sql_upper = sql_clean.upper()
        if not sql_upper.startswith("SELECT"):
            return False, "僅允許 SELECT 查詢", None

        dangerous = ["DROP", "INSERT", "UPDATE", "DELETE", "ALTER", "CREATE"]
        for kw in dangerous:
            if re.search(rf"\b{kw}\b", sql_upper):
                return False, f"查詢包含禁止關鍵字：{kw}", None

        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_clean, conn)
            conn.close()
            return True, f"成功取得 {len(df)} 筆資料", df
        except Exception as e:
            return False, f"SQL 執行錯誤：{e}", None