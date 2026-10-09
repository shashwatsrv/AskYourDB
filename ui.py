import streamlit as st
import uuid
import os
import pandas as pd
from app.core import run_query, explain_sql, load_file_to_sqlite

st.set_page_config(page_title="AskYourDB", layout="wide")
st.title("AskYourDB")
st.caption("Connect your Postgres database or upload a CSV/XLSX and query it in plain English")


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then fall back to environment variables."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


# --- Session state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.header("Data source")
    source = st.radio("Connect via", ["Postgres", "CSV / XLSX"], index=0)

    db_url = None
    if source == "Postgres":
        db_url = st.text_input(
            "Connection string",
            value=get_secret("DEMO_DB_URL"),
            placeholder="postgresql://user:password@host:5432/dbname",
            type="password"
        )
    else:
        uploaded = st.file_uploader(
            "Upload file",
            type=["csv", "xlsx"],
            help="Max 10MB"
        )
        if uploaded:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("File exceeds 10MB limit.")
                st.stop()
            db_url, table_name = load_file_to_sqlite(uploaded)
            st.success(f"Loaded table: `{table_name}`")

    st.divider()
    st.header("LLM")
    mode = st.radio("Inference mode", ["Cloud (Groq)", "Local (Ollama)"], index=0)

    if mode == "Cloud (Groq)":
        user_api_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="Leave blank to use the demo key"
        ).strip()
        api_key = user_api_key or get_secret("LLM_API_KEY_GROQ")
        if not user_api_key and api_key:
            st.caption("Using the demo key.")
        base_url = "https://api.groq.com/openai/v1"
        model = get_secret("LLM_MODEL_GROQ", "openai/gpt-oss-120b")
    else:
        api_key = "ollama"
        base_url = "http://localhost:11434/v1"
        model = "qwen2.5-coder:7b"
        st.caption("⚠️ Requires Ollama running locally on port 11434.")

    st.divider()
    st.header("Session")
    st.caption(f"ID: {st.session_state.session_id[:8]}...")

    if st.button("New conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# --- Validate inputs ---
if not db_url:
    if source == "Postgres":
        user_db_url = st.text_input(
            "Connection string",
            placeholder="Leave blank to use the demo database",
            type="password"
        ).strip()
        db_url = user_db_url or get_secret("DEMO_DB_URL")
        if not user_db_url and db_url:
            st.caption("Using the demo database (Northwind, read-only).")

if not api_key:
    st.warning("Enter your API key in the sidebar.")
    st.stop()


# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg.get("sql"):
            with st.expander("Generated SQL"):
                st.code(msg["sql"], language="sql")

        if msg.get("rows"):
            with st.expander("Results"):
                st.dataframe(pd.DataFrame(msg["rows"], columns=msg.get("columns")))

        if msg.get("confidence") is not None:
            st.caption(f"Confidence: {msg['confidence']} · Intent: {msg.get('intent')}")

        if msg.get("warning"):
            st.warning(msg["warning"])


# --- Input ---
if prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                output = run_query(
                    user_input=prompt,
                    db_url=db_url,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    history=st.session_state.history
                )

                if output.get("error"):
                    st.error(f"Error: {output['error']}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {output['error']}"
                    })

                else:
                    st.write("Query executed successfully.")

                    with st.expander("Generated SQL", expanded=True):
                        st.code(output["sql"], language="sql")

                    if output["rows"]:
                        with st.expander("Results", expanded=True):
                            st.dataframe(
                                pd.DataFrame(output["rows"], columns=output.get("columns"))
                            )
                    else:
                        st.info("Query returned no results.")

                    st.caption(f"Confidence: {output['confidence']} · Intent: {output['intent']}")

                    if output.get("warning"):
                        st.warning(output["warning"])

                    # explain
                    explanation = explain_sql(output["sql"], api_key, model, base_url)
                    if explanation:
                        st.info(f"💡 {explanation}")

                    # update session history
                    st.session_state.history.append({"role": "user", "content": prompt})
                    st.session_state.history.append({"role": "assistant", "content": output["sql"]})

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Query executed successfully.",
                        "sql": output["sql"],
                        "rows": output["rows"],
                        "columns": output.get("columns"),
                        "confidence": output["confidence"],
                        "intent": output["intent"],
                        "warning": output.get("warning")
                    })

            except Exception as e:
                st.error(f"Something went wrong: {e}")