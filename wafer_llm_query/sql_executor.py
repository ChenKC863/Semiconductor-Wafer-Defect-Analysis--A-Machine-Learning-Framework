import sqlite3
import pandas as pd
import re

class SQLExecutor:
    def __init__(self, db_path):
        self.db_path = db_path

    def execute(self, sql_query):
        sql_upper = sql_query.strip().upper()
        if not sql_upper.startswith("SELECT"):
            return False, "僅允許 SELECT 查詢", None

        dangerous = ["DROP", "INSERT", "UPDATE", "DELETE", "ALTER", "CREATE"]
        for kw in dangerous:
            if re.search(rf"\b{kw}\b", sql_upper):
                return False, f"查詢包含禁止關鍵字：{kw}", None

        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return True, f"成功取得 {len(df)} 筆資料", df
        except Exception as e:
            return False, f"SQL 執行錯誤：{e}", None