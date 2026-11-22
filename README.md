# AskYourDB

Ask your database questions in plain English. AskYourDB turns a question into SQL, validates it, runs it, shows the results, and explains the query back to you.

**Live demo:** https://askurdb.streamlit.app/ (read-only Northwind database, or bring your own Postgres / CSV / XLSX; edits are available on your own data only)

---

## Features

- **Natural language to SQL** for Postgres databases, or any CSV / XLSX file (loaded into a temporary SQLite database)
- **Schema-aware generation**: only the relevant tables are sent to the LLM (semantic schema retrieval)
- **Safety first**: read-only by default, validated on the AST before anything touches the database. Optional edits on your own data are previewed, confirmed, run in a transaction and logged (see MakeYourDB below)
- **Plain-English explanation** of every generated query
- **Confidence score** and intent label shown with each answer
- **Follow-up questions**: the last 10 messages are kept as context
- **Cloud or local inference**: Groq (cloud) or Ollama (local)

---

## How it works

```
                        "top 5 products by revenue"
                                    |
                                    v
                    +-------------------------------+
                    |        Intent classifier      |   facebook/bart-large-mnli (zero-shot)
                    |  aggregation | filter | join  |   -> used as a prompt hint, not a gate
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |      Schema retrieval (RAG)   |   all-MiniLM-L6-v2, cosine similarity
                    |  top 6 tables (all if vague)  |
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |         Prompt builder        |   schema + intent hint + last 10 messages
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |              LLM              |   Groq (cloud) or Ollama (local)
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |      SQL extraction + guard   |   must contain a SELECT / WITH query
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |    AST validation (sqlglot)   |   syntax, blocked operations,
                    |                               |   unknown-table check
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    |    Execute  +  Confidence     |   coverage 0.5 / complexity 0.3 / validity 0.2
                    +---------------+---------------+
                                    |
                                    v
                       Results  +  SQL  +  Explanation
```

---

## MakeYourDB (v2): controlled edits

Edits are an opt-in mode (the **Enable edits** switch). Supported statements: `INSERT`, `UPDATE`, `DELETE`. Schema changes (`ALTER`, `DROP`, `CREATE`, `TRUNCATE`, ...) are never allowed.

**Where edits are allowed:** uploaded CSV / XLSX files (each visitor edits their own temporary copy) and a Postgres connection string you paste yourself (your database, your permissions). The shared demo database is never writable: the app does not offer the switch for it, and its role is `SELECT`-only.

**Safety layers**

1. **Allowlist guard.** Exactly one statement; only `SELECT` (and `INSERT` / `UPDATE` / `DELETE` when edits are on). Multi-statement input, data-modifying CTEs and `SELECT ... INTO` are rejected, and every table must exist.
2. **Preview, nothing changes yet.** `UPDATE` / `DELETE` show the rows that currently match (the "before" image), and `UPDATE` adds `new_<column>` values; `INSERT` shows the rows to be added. Warnings for no `WHERE`, no matching rows, and more than 100 rows. Forms that cannot be previewed faithfully (`UPDATE ... FROM`, `DELETE ... USING`, `ON CONFLICT`) are refused.
3. **Explicit confirmation.** Confirm / Cancel, and a typed `CONFIRM` for every `DELETE` and for any `UPDATE` / `DELETE` without a `WHERE`.
4. **Transaction with checks.** The statement is re-validated at execution. The number of matching rows must still equal the previewed count, and the rows affected must equal it afterwards; any mismatch or database error rolls everything back.
5. **Audit log, written before the change.** If the log is unavailable, the write does not run. Entries hold time, session, database (credentials are never stored), SQL, row counts, the before-image (up to 1000 rows) and the outcome. The log is a separate SQLite file, so nothing is added to your database, and it is visible per session in the sidebar.

**Helping the model write valid changes.** In edit mode the schema sent to the model flags columns as required, auto-filled or generated. For a single-row `INSERT` with no ID given, the model may fill a required integer ID with the next value (`MAX(id) + 1`); for any other missing required value it asks you.

**Testing.** The guard, preview and execution layer were covered by 76 sandbox test cases during development (not yet a committed test suite). Live checks were run on SQLite (uploaded files) and on Postgres 16 (local Docker and a Neon branch): committed `INSERT` / `UPDATE` / `DELETE`, plus rollbacks on foreign-key and duplicate-key violations, verified directly in the database.

---

## Two code paths

The repository contains two versions of the pipeline.

| | **Deployed app** | **v1 backend (FastAPI)** |
|---|---|---|
| Entry point | `ui.py` (Streamlit) | `app/main.py` (FastAPI) |
| Pipeline | `app/core.py` | `app/pipeline.py`, `rag.py`, `intent.py`, `validator.py` |
| Schema retrieval | In-memory cosine similarity | pgvector |
| Caching | None | Redis exact cache + pgvector semantic cache (0.92 cosine) |
| Validation | sqlglot AST checks | sqlglot AST checks + `EXPLAIN` plan check |
| Multi-turn | `st.session_state` | Redis sessions |
| Extras | CSV / XLSX upload, SQL explanation in the UI | `/query`, `/history`, `/explain` endpoints, API-key auth, query history table |
| Runs on | Streamlit Community Cloud | Docker (Postgres + pgvector, Redis) + uvicorn |

The deployed app is deliberately lightweight so it runs on a free tier. The FastAPI backend is the fuller service architecture and runs locally.

---

## Design decisions

- **The intent classifier is a hint, not a gate.** Zero-shot confidence is low and noisy (typically 0.3 to 0.5), so a wrong label must never block a query. It only nudges the prompt.
- **Retrieve tables, don't dump the schema.** Sending only relevant tables keeps prompts small and reduces hallucinated columns. For vague questions the full schema is sent instead.
- **Validate on the AST, with an allowlist.** sqlglot parses the query, and only `SELECT` (plus `INSERT`, `UPDATE`, `DELETE` when edits are on) is accepted. Everything else, including multi-statement input, is rejected, and unknown tables are caught before execution.
- **Defence in depth.** The public demo connects with a Postgres role that only has `SELECT`. Even if a bad query got through validation, the database would reject it.
- **Secrets stay on the server.** Default credentials are never rendered in the UI. A visitor can paste their own connection string or API key, and those stay viewable to them for debugging.
- **Confidence is grounded, not opaque.** It combines schema coverage (0.5), query complexity (0.3) and validity (0.2).

---

## Known limitations

- **Intent classification is weak.** It is used only as a hint for exactly this reason.
- **Retrieval can miss join tables.** For example, "sales by region" may not surface `order_details`. Top-k of 6 and full-schema retrieval for vague questions mitigate this. At 100+ tables this needs table-level embeddings with two-pass retrieval or foreign-key-aware expansion.
- **Validation is structural, not semantic.** sqlglot checks syntax, blocked operations and table existence, not logical correctness or column names. The `EXPLAIN` check in the FastAPI backend catches column errors and runaway plans, but it is not in the deployed app.
- **Ambiguous wording gets an interpretation.** "Most expensive product" can mean highest unit price or highest total spend. The model picks one. Be specific for exact results.
- **Confidence is a heuristic**, not a calibrated probability. Validity is always 1.0 at scoring time because invalid queries return earlier.
- **Ollama mode works only when run locally.**
- **Edits have no real user authentication.** The audit log identifies the Streamlit session, not a login.
- **The audit log is ephemeral on Streamlit Community Cloud** (it lives in the OS temp directory; set `ASKYOURDB_AUDIT_DB` to keep it elsewhere).
- **No undo yet.** Before-images are stored, but there is no restore button.
- **The model can misread a request.** The preview and confirmation are the control. The app does not check column names or `NOT NULL` itself; the database rejects bad writes and the transaction rolls back.
- **Not tested:** several users editing the same data at once, and edits through the deployed Streamlit Cloud app.

---

## Run locally

### Streamlit app

```powershell
git clone https://github.com/shashwatsrv/AskYourDB.git
cd AskYourDB
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (already git-ignored) so the app has defaults:

```toml
DEMO_DB_URL = "postgresql://user:password@host:5432/dbname"
LLM_API_KEY_GROQ = "your-groq-key"
LLM_MODEL_GROQ = "openai/gpt-oss-120b"
```

```powershell
streamlit run ui.py
```

Or leave the secrets out and paste your connection string and API key into the sidebar. For local models, install [Ollama](https://ollama.com), run `ollama pull qwen2.5-coder:7b`, and choose **Local (Ollama)** in the sidebar.

### FastAPI backend (v1)

Needs Docker for Postgres with pgvector and Redis.

```powershell
docker compose up -d
pip install -r requirements-prod.txt
```

Create a `.env` file:

```
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
APP_API_KEY=choose-a-key
LLM_API_KEY_GROQ=your-groq-key
LLM_BASE_URL_GROQ=https://api.groq.com/openai/v1
LLM_MODEL_GROQ=openai/gpt-oss-120b
```

```powershell
python -m app.rag
uvicorn app.main:app --reload
```

`python -m app.rag` embeds your schema into pgvector. Send requests with the `x-api-key` header.

---

## Tech stack

| Area | Tools |
|---|---|
| UI | Streamlit |
| Backend (v1) | FastAPI, uvicorn |
| LLM | Groq (`openai/gpt-oss-120b`), Ollama (`qwen2.5-coder:7b`), OpenAI-compatible client |
| ML | Hugging Face Transformers (`bart-large-mnli`), Sentence-Transformers (`all-MiniLM-L6-v2`), PyTorch (CPU) |
| SQL | sqlglot, SQLAlchemy, psycopg2 |
| Data and infra | Postgres, pgvector, Redis, SQLite, pandas, openpyxl, Docker |
| Demo hosting | Streamlit Community Cloud, Neon (Postgres) |

---

## Roadmap

**Done in v2:** `INSERT` / `UPDATE` / `DELETE` with preview, confirmation, transaction rollback and an audit log.

**Next:** undo from the stored before-images, a persistent audit log with real authentication, OCR ingestion (images and scans into a queryable database), foreign-key-aware schema retrieval, and a column-existence check.
---

## Author

Shashwat Saurav ([@shashwatsrv](https://github.com/shashwatsrv))
