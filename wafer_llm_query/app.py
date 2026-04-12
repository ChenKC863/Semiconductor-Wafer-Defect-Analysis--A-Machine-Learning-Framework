import streamlit as st
from llm_client import LLMClient
from sql_executor import SQLExecutor
import re

st.set_page_config(page_title="晶圓缺陷自然語言查詢", layout="wide")
st.title("🔍 晶圓缺陷資料庫 - 自然語言查詢")
st.markdown("輸入自然語言問題，系統將轉為 SQL 並回傳結果")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("設定")
    db_path = st.selectbox(
        "選擇資料庫",
        ["wafer_features_classwise_S.db", "wafer_features_classwise_M.db"]
    )
    # Keep only Ollama; model selection uses a drop-down menu.
    model = st.selectbox(
        "Ollama 模型名稱",
        ["llama3.2:3b", "qwen2.5-coder:7b"]
    )
    language = st.selectbox("Language / 語言", ["中文", "English"])

    if st.button("清除對話歷史"):
        st.session_state.messages = []
        st.rerun()

SYSTEM_PROMPT = {
    "中文": """
你是一個 SQL 專家。資料庫 Schema 如下：
表格: wafers
欄位:
- image_path (TEXT): 影像路徑
- split (TEXT): 資料集分割 (train/valid/test)
- true_label (TEXT): 真實缺陷類別 (首字母大寫，例如 'Donut', 'Center', 'Edge-Loc')
- pred_label (TEXT): 模型預測類別
- pred_prob (REAL): 預測機率 (0~1)
- anomaly_score (REAL): 異常分數，數值越高代表越不異常，數值越低代表越異常
- is_anomaly (INTEGER): 是否為異常 (1=異常, 0=正常)

**嚴格遵守以下規則：**
1. 文字比較必須不區分大小寫。使用 `LOWER(true_label) = LOWER('donut')`。
2. 「異常分數最高」→ `ORDER BY anomaly_score DESC`；「最異常」→ `ORDER BY anomaly_score ASC`。
3. 只回傳 SQL 語句，不要有任何其他文字、解釋、標記或引號。
4. **當使用者要求列出具體圖片並涉及異常分數排序時，SELECT 必須同時包含 `image_path` 和 `anomaly_score`。**
5. **絕對不要使用 `title_label` 這個欄位，正確欄位名是 `true_label`。**

正確範例：
使用者問：「Donut 類別中，異常分數最高的前 5 筆資料是哪幾張圖片？」
你必須輸出：
SELECT image_path, anomaly_score FROM wafers WHERE LOWER(true_label) = LOWER('Donut') ORDER BY anomaly_score DESC LIMIT 5;
""",
    "English": """
You are a SQL expert. Schema: table wafers with columns: image_path, split, true_label, pred_label, pred_prob, anomaly_score, is_anomaly.

Strict rules:
1. Case‑insensitive: use `LOWER(true_label) = LOWER(value)`.
2. "highest anomaly score" → `ORDER BY anomaly_score DESC`; "most anomalous" → `ORDER BY anomaly_score ASC`.
3. Return ONLY the SQL statement, no extra text, no quotes, no markdown.
4. When user asks for images with anomaly score ordering, SELECT must include both `image_path` and `anomaly_score`.
5. The correct column name is `true_label`, never `title_label`.

Example – user: "Donut class, top 5 images with highest anomaly score"
You must output:
SELECT image_path, anomaly_score FROM wafers WHERE LOWER(true_label) = LOWER('Donut') ORDER BY anomaly_score DESC LIMIT 5;
"""
}

SUMMARY_TEMPLATE = {
    "中文": """
使用者問題：{question}
資料庫查詢 SQL：{sql}
查詢結果：
{data}
請用繁體中文總結這些資料，回答使用者的問題。如果結果很多，可以說明總筆數並舉例。
""",
    "English": """
User question: {question}
SQL query: {sql}
Query result:
{data}
Please summarize the data in English to answer the user's question. If there are many results, mention the total count and give examples.
"""
}

try:
    llm = LLMClient(provider="ollama", model=model, api_key=None)
except Exception as e:
    st.sidebar.error(f"LLM 初始化失敗：{e}")
    st.stop()

executor = SQLExecutor(db_path)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("請輸入您的問題 / Enter your question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            system_msg = SYSTEM_PROMPT[language]
            messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
            sql_raw = llm.chat(messages, temperature=0.0).strip()
            
            # Clean up SQL: Remove markdown code blocks, extra quotes, and leading and trailing whitespace.
            sql_clean = re.sub(r'```sql\s*|```\s*', '', sql_raw, flags=re.IGNORECASE)
            sql_clean = re.sub(r'^[\'"]|[\'"]$', '', sql_clean.strip())
            if sql_clean.startswith("'") and sql_clean.endswith("'"):
                sql_clean = sql_clean[1:-1]
            if sql_clean.startswith('"') and sql_clean.endswith('"'):
                sql_clean = sql_clean[1:-1]
            
            sql = sql_clean.strip()
            
            st.code(sql, language="sql")

            success, msg, df = executor.execute(sql)

            if not success:
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": f"查詢失敗：{msg}"})
            else:
                if df.empty:
                    summary = "查詢結果為空，沒有找到符合條件的資料。" if language == "中文" else "The query returned no results."
                else:
                    df_preview = df.head(20)
                    data_text = df_preview.to_string(index=False)
                    summary_prompt = SUMMARY_TEMPLATE[language].format(question=prompt, sql=sql, data=data_text)
                    summary = llm.chat([{"role": "user", "content": summary_prompt}])

                st.success(f"✅ {msg}")
                st.write("### 自然語言回答 / Natural Language Answer")
                # 修改：使用 st.text 避免路徑被渲染為連結（導致顏色變化）
                st.text(summary)
                st.write("### 查詢結果表格 / Query Result")
                st.dataframe(df)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"查詢到 {len(df)} 筆資料。\n\n{summary}" if language == "中文" else f"Found {len(df)} rows.\n\n{summary}"
                })