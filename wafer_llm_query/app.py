import streamlit as st
from llm_client import LLMClient
from sql_executor import SQLExecutor

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
    provider = st.selectbox("LLM 後端", ["ollama", "openai"])
    if provider == "ollama":
        model = st.text_input("Ollama 模型名稱", value="llama3.2:3b")
        api_key = None
    else:
        model = st.text_input("OpenAI 模型", value="gpt-4o-mini")
        api_key = st.text_input("OpenAI API Key", type="password")
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
- true_label (TEXT): 真實缺陷類別 (注意：儲存的值為首字母大寫，例如 'Donut', 'Center', 'Edge-Loc' 等)
- pred_label (TEXT): 模型預測類別
- pred_prob (REAL): 預測機率 (0~1)
- anomaly_score (REAL): 異常分數，數值越高代表越不異常，數值越低代表越異常
- is_anomaly (INTEGER): 是否為異常 (1=異常, 0=正常)

請根據使用者的自然語言問題，生成標準 SQLite 語法的 SELECT 查詢。
**重要規則：**
1. 文字比較必須不區分大小寫。你**必須**使用 `LOWER(true_label) = LOWER('donut')` 而不是 `true_label = 'donut'`。
2. 當使用者說「異常分數最高」或「分數最高」時，表示 anomaly_score 數值最大，請使用 `ORDER BY anomaly_score DESC`。
3. 當使用者說「最異常」或「異常分數最低」時，請使用 `ORDER BY anomaly_score ASC`。
4. 只回傳 SQL 語句，不要包含任何其他文字、解釋或標記。
""",
    "English": """
You are a SQL expert. The database schema is as follows:
Table: wafers
Columns:
- image_path (TEXT): image file path
- split (TEXT): dataset split (train/valid/test)
- true_label (TEXT): true defect class
- pred_label (TEXT): predicted defect class
- pred_prob (REAL): prediction probability (0~1)
- anomaly_score (REAL): anomaly score (lower means more anomalous)
- is_anomaly (INTEGER): whether it is anomalous (1=yes, 0=no)

Based on the user's natural language question, generate a standard SQLite SELECT query.
Text comparisons should be case-insensitive. For example, `true_label='donut'` should use `LOWER(true_label) = LOWER('donut')` or `true_label COLLATE NOCASE = 'donut'`.
Return only the SQL statement, no additional text or explanation.
"""
}

SUMMARY_TEMPLATE = {
    "中文": """
使用者問題：{question}
資料庫查詢 SQL：{sql}
查詢結果（前 20 筆）：
{data}
請用繁體中文總結這些資料，回答使用者的問題。如果結果很多，可以說明總筆數並舉例。
""",
    "English": """
User question: {question}
SQL query: {sql}
Query result (first 20 rows):
{data}
Please summarize the data in English to answer the user's question. If there are many results, mention the total count and give examples.
"""
}

try:
    llm = LLMClient(provider=provider, model=model, api_key=api_key)
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
            sql = llm.chat(messages, temperature=0.0).strip()
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
                st.markdown(summary)
                st.write("### 查詢結果表格 / Query Result")
                st.dataframe(df)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"查詢到 {len(df)} 筆資料。\n\n{summary}" if language == "中文" else f"Found {len(df)} rows.\n\n{summary}"
                })