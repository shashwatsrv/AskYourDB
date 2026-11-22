import streamlit as st
import uuid
import os
import pandas as pd
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from app.core import run_query, explain_sql, load_files_to_sqlite, execute_write, get_audit_log

MAX_FILES = 5
MAX_MB = 10
st.set_page_config(page_title="AskYourDB", layout="wide")
st.title("AskYourDB")
st.caption("Connect your Postgres database or upload a CSV/XLSX and query it in plain English")


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then fall back to environment variables."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

def describe_target(url: str) -> str:
    """Human-readable database name for a connection URL, never including credentials."""
    try:
        u = make_url(url)
    except Exception:
        return "the connected database"
    if u.get_backend_name() == "sqlite":
        return "your uploaded files (temporary copy)"
    return f"{u.host or 'localhost'} / {u.database}"

def friendly_error(err) -> str:
    """Turn raw database / guard errors into short plain-language messages."""
    msg = str(err or "")
    msg = re.sub(r"\[SQL:.*", "", msg, flags=re.S)
    msg = re.sub(r"\(Background on this error at:[^)]*\)", "", msg).strip()
    low = msg.lower()
    m = re.search(r"DETAIL:\s*(.*)", msg, flags=re.S)
    detail = m.group(1).strip() if m else ""
    if "notnullviolation" in low or "not null constraint" in low:
        head = "A required column was left empty (it cannot be NULL). Include a value for every required column, such as the ID."
    elif "uniqueviolation" in low or "unique constraint" in low:
        head = "That would duplicate a value that must be unique, such as an ID that already exists."
    elif "foreignkeyviolation" in low or "foreign key constraint" in low:
        head = "Blocked by a link between tables: other rows still reference these rows, or the referenced row does not exist."
    elif "only a single sql statement" in low:
        head = "The model produced more than one statement. Ask about one table at a time, or name the table."
    else:
        return msg or "Unknown error."
    return f"{head} {detail}".strip()

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
    user_db_url = ""
    if source == "Postgres":
        # Defaults are never placed in the widget, so the eye icon only
        # reveals what the visitor typed themselves.
        user_db_url = st.text_input(
            "Connection string",
            placeholder="Leave blank to use the demo database",
            type="password"
        ).strip()
        db_url = user_db_url or get_secret("DEMO_DB_URL")
        if not user_db_url and db_url:
            st.caption("Using the demo database (Northwind, read-only).")
    else:

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            help=f"Up to {MAX_FILES} files, max {MAX_MB}MB each"
        )
        if uploaded_files:
            if len(uploaded_files) > MAX_FILES:
                st.error(f"Please upload at most {MAX_FILES} files.")
                st.stop()
            for f in uploaded_files:
                if f.size > MAX_MB * 1024 * 1024:
                    st.error(f"{f.name} exceeds the {MAX_MB}MB limit.")
                    st.stop()

            # Load once per distinct set of files, not on every rerun
            sig = tuple((f.name, f.size) for f in uploaded_files)
            if st.session_state.get("upload_sig") != sig:
                with st.spinner("Loading files..."):
                    url, tables = load_files_to_sqlite(uploaded_files)
                st.session_state.upload_sig = sig
                st.session_state.upload_db_url = url
                st.session_state.upload_tables = tables
                # data changed, so old conversation no longer applies
                st.session_state.messages = []
                st.session_state.history = []

            db_url = st.session_state.upload_db_url
            for t in st.session_state.upload_tables:
                st.success(f"`{t['table']}` ({t['rows']} rows) from {t['source']}")
    # Edits are only offered on the visitor's own data: uploaded files,
    # or a connection string they typed. Never the default demo database.
    can_edit = bool(db_url) and (source == "CSV / XLSX" or bool(user_db_url))
    allow_writes = False
    if can_edit:
        allow_writes = st.toggle(
            "Enable edits (INSERT / UPDATE / DELETE)",
            value=False,
            help="Every change is previewed, needs your confirmation, runs in a transaction and is logged."
        )
    elif source == "Postgres":
        st.caption("Edits are off for the shared demo database.")
    if allow_writes:
        st.warning(f"Edits will modify: {describe_target(db_url)}")
    else:
        st.session_state.pop("pending", None)

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

    with st.expander("Change log (this session)"):
        entries = get_audit_log(session_id=st.session_state.session_id, limit=20)
        if not entries:
            st.caption("No changes yet.")
        for e in entries:
            st.markdown(
                f"**#{e['id']}** {e['kind'].upper()} on `{e['table_name']}`: "
                f"**{e['status']}**, {e['rows_affected']} row(s)"
            )
            st.caption(e["ts"])
            st.code(e["sql"], language="sql")
            if e.get("error"):
                st.caption(f"Error: {e['error']}")

    if st.button("New conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# --- Validate inputs ---
if not db_url:
    if source == "Postgres":
        st.info("👈 Enter your Postgres connection string in the sidebar to get started.")
    else:
        st.info("👈 Upload one or more CSV/XLSX files in the sidebar to get started.")
    st.stop()

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

# --- Pending write: preview + confirm ---
def refresh_upload_counts(url):
    """After a committed change on uploaded files, update the sidebar row counts."""
    try:
        eng = create_engine(url)
        with eng.connect() as conn:
            for t in st.session_state.get("upload_tables", []):
                t["rows"] = conn.execute(text(f'SELECT COUNT(*) FROM "{t["table"]}"')).scalar()
        eng.dispose()
    except Exception:
        pass


pending = st.session_state.get("pending")
if pending:
    pv = pending["preview"]
    kind = pending["kind"]
    needs_all_rows = kind in ("update", "delete") and not pv.get("has_where")
    needs_typed = kind == "delete" or needs_all_rows

    with st.container(border=True):
        st.subheader(f"Review proposed {kind.upper()}")
        st.code(pending["sql"], language="sql")
        label = "Rows to be added" if kind == "insert" else "Rows currently matching (before the change)"
        st.caption(f"Database: {describe_target(pending['db_url'])}")
        st.caption(f"Table: `{pv.get('table')}` | {label}: {pv['total']}")
        if kind == "update":
            st.caption("Columns starting with new_ show the value each row will have after the change.")
        for w in pv.get("warnings", []):
            st.warning(w)
        if pv.get("rows"):
            st.dataframe(pd.DataFrame(pv["rows"], columns=pv.get("columns")))

        typed = ""
        if needs_typed:
            typed = st.text_input("Type CONFIRM to enable the button", key=f"confirm_{pending['id']}")
        c1, c2 = st.columns(2)
        confirm = c1.button(
            "Confirm and run", type="primary",
            disabled=needs_typed and typed.strip() != "CONFIRM",
        )
        cancel = c2.button("Cancel")

    if cancel:
        st.session_state.pending = None
        st.session_state.messages.append({"role": "assistant", "content": "Change cancelled. Nothing was modified."})
        st.rerun()

    if confirm:
        with st.spinner("Running in a transaction..."):
            res = execute_write(
                db_url=pending["db_url"],
                sql=pending["sql"],
                expected_total=pv["total"],
                session_id=st.session_state.session_id,
                allow_all_rows=needs_all_rows,
            )
        st.session_state.pending = None
        if res["ok"]:
            st.session_state.history.append({"role": "user", "content": pending["prompt"]})
            st.session_state.history.append({"role": "assistant", "content": pending["sql"]})
            msg = f"Committed: {res['rows_affected']} row(s) affected (audit #{res['audit_id']})."
            if source == "CSV / XLSX":
                refresh_upload_counts(pending["db_url"])
        else:
            msg = f"Not applied, no data was changed ({res['status']}). {friendly_error(res.get('error'))}"
        st.session_state.messages.append({"role": "assistant", "content": msg, "sql": pending["sql"]})
        st.rerun()

# --- Input ---
if prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.pending = None  # a new question discards any unconfirmed change
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
                    history=st.session_state.history,
                    allow_writes=allow_writes
                )

                if output.get("error"):
                    st.error(f"Error: {friendly_error(output['error'])}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {friendly_error(output['error'])}"
                    })

                elif output.get("notice"):
                    st.info(output["notice"])
                    st.session_state.messages.append({"role": "assistant", "content": output["notice"]})

                elif output.get("pending_write"):
                    st.session_state.pending = {
                        "id": str(uuid.uuid4()),
                        "prompt": prompt,
                        "sql": output["sql"],
                        "kind": output["kind"],
                        "preview": output["preview"],
                        "db_url": db_url,
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Proposed change ready. Review the preview below, then confirm or cancel.",
                        "sql": output["sql"],
                    })
                    st.rerun()

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