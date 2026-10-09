# AskYourDB

Ask your database questions in plain English. AskYourDB turns a question into SQL, validates it, runs it, shows the results, and explains the query back to you.

**Live demo:** https://askurdb.streamlit.app/ (read-only Northwind database, or bring your own Postgres / CSV / XLSX)

---

## Features

- **Natural language to SQL** for Postgres databases, or any CSV / XLSX file (loaded into a temporary SQLite database)
- **Schema-aware generation**: only the relevant tables are sent to the LLM (semantic schema retrieval)
- **Safety first**: read-only queries only, validated on the AST before anything touches the database
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
- **Validate on the AST, not with regex.** sqlglot parses the query so blocked operations (`DROP`, `DELETE`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`, `CREATE`) and unknown tables are caught reliably, before execution.
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
- **Read-only.** Write operations are not supported yet (see roadmap).

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

**MakeYourDB (v2)**: controlled write operations, built only after v1 is solid.

- `INSERT` first, then `UPDATE`, then `DELETE`. `ALTER` is never allowed.
- Row preview before any change, explicit user confirmation, and a transaction wrapper with automatic rollback.
- Audit log of every write.

---

## Author

Shashwat Saurav ([@shashwatsrv](https://github.com/shashwatsrv))
