import sqlite3

conn = sqlite3.connect("wafer_features_classwise_S.db") #S and M two databases, you can switch to the one of them if needed.
cursor = conn.cursor()

# Please check the value of true_label / pred_label
cursor.execute("SELECT true_label, COUNT(*) FROM wafers GROUP BY true_label ORDER BY COUNT(*) DESC")
print("=== true_label 統計 ===")
for label, cnt in cursor.fetchall():
    print(label, cnt)

cursor.execute("SELECT pred_label, COUNT(*) FROM wafers GROUP BY pred_label ORDER BY COUNT(*) DESC")
print("\n=== pred_label 統計 ===")
for label, cnt in cursor.fetchall():
    print(label, cnt)

print("\n=== Donut 類別的前 5 筆資料 ===")
cursor.execute("""
    SELECT image_path, anomaly_score, is_anomaly
    FROM wafers
    WHERE lower(true_label) = 'donut'
    ORDER BY anomaly_score DESC
    LIMIT 5
""")
for row in cursor.fetchall():
    print(row)

print("\n=== 測試集中異常圖片數量 ===")
cursor.execute("SELECT COUNT(*) FROM wafers WHERE split = 'test' AND is_anomaly = 1")
print(cursor.fetchone()[0])

conn.close()