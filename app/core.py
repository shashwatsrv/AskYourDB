from openai import OpenAI
from sqlalchemy import create_engine, text, inspect
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
import sqlglot
import os
import re
import numpy as np
import sqlite3
import pandas as pd
import tempfile
import json
from datetime import datetime, timezone
from sqlalchemy.engine import make_url

# --- Models (loaded once) ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
intent_classifier = hf_pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

INTENT_LABELS = ["aggregation", "filter", "join", "ambiguous"]
INTENT_HINTS = {
    "aggregation": "Use GROUP BY, COUNT, SUM, AVG, or other aggregate functions.",
    "filter": "Use WHERE clause to filter rows based on conditions.",
    "join": "Use JOIN to combine data from multiple tables.",
    "ambiguous": "The question may be vague. Make the most reasonable assumption and still return a SQL query. Never ask for clarification.",
}

# Statement kinds the guard understands. Anything else (DROP, ALTER, CREATE,
# TRUNCATE, GRANT, COPY, EXPLAIN, ...) is rejected: the guard is an allowlist.
WRITE_KINDS = {"insert", "update", "delete"}

# --- Schema ---
def get_schema(engine) -> list[dict]:
    inspector = inspect(engine)
    chunks = []
    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        col_defs = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        chunks.append({
            "table": table,
            "text": f"Table {table} has columns: {col_defs}"
        })
    return chunks

# --- RAG ---
def retrieve_schema(query: str, chunks: list[dict], top_k: int = 6) -> str:
    query_emb = embedding_model.encode(query)
    chunk_embs = embedding_model.encode([c["text"] for c in chunks])
    scores = np.dot(chunk_embs, query_emb) / (
        np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-9
    )
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = [chunks[i]["text"] for i in top_indices]
    return "Relevant tables:\n" + "\n".join(f"- {t}" for t in retrieved)

# --- Intent ---
def classify_intent(query: str) -> dict:
    result = intent_classifier(query, candidate_labels=INTENT_LABELS)
    intent = result["labels"][0]
    return {
        "intent": intent,
        "score": round(result["scores"][0], 3),
        "hint": INTENT_HINTS[intent]
    }

# --- Validation ---
def classify_statement(parsed) -> str:
    """Return 'read', 'insert', 'update', 'delete', or 'other'.
    'other' means any statement type that is not explicitly allowed."""
    exp = sqlglot.exp
    if isinstance(parsed, exp.Insert):
        return "insert"
    if isinstance(parsed, exp.Update):
        return "update"
    if isinstance(parsed, exp.Delete):
        return "delete"
    if isinstance(parsed, (exp.Select, exp.SetOperation)):
        return "read"
    return "other"


def validate_sql(sql: str, known_tables: set, dialect: str = "postgres",
                 allow_writes: bool = False) -> dict:
    exp = sqlglot.exp

    # 1. parse; exactly one statement (blocks "SELECT 1; DROP TABLE x")
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=dialect) if s]
    except sqlglot.errors.ParseError as e:
        import re
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e))
        return {"valid": False, "error": f"Syntax error: {msg}", "kind": None}
    if len(statements) != 1:
        return {"valid": False, "kind": None,
                "error": "Only a single SQL statement is allowed."}
    parsed = statements[0]

    # 2. statement type must be on the allowlist
    kind = classify_statement(parsed)
    if kind == "other":
        return {"valid": False, "kind": kind,
                "error": f"Operation not allowed: {parsed.key.upper()}"}
    if kind in WRITE_KINDS and not allow_writes:
        return {"valid": False, "kind": kind,
                "error": f"Write operations are disabled: {kind.upper()}"}

    # 3. no hidden writes: data-modifying CTEs, SELECT ... INTO (creates a table)
    nested = [n for n in parsed.find_all(exp.Insert, exp.Update, exp.Delete)
              if n is not parsed]
    if nested or parsed.find(exp.Into):
        return {"valid": False, "kind": kind,
                "error": "Nested writes and SELECT INTO are not allowed."}

    # 4. every referenced table must exist (CTE names are not real tables)
    cte_names = {c.alias.lower() for c in parsed.find_all(exp.CTE)}
    referenced = {t.name.lower() for t in parsed.find_all(exp.Table)} - cte_names
    unknown = referenced - known_tables
    if unknown:
        return {"valid": False, "kind": kind, "error": f"Unknown tables: {unknown}"}

    return {"valid": True, "error": None, "kind": kind}

def _as_text(sql: str):
    """Wrap generated SQL for SQLAlchemy. Colons are escaped so a literal like
    'see :ref' is never mistaken for a bind parameter."""
    return text(sql.replace(":", r"\:"))


# --- Write preview ---
WRITE_ROW_WARN = 100   # warn when a write touches more rows than this
PREVIEW_SAMPLE = 50    # max rows shown in a preview


class PreviewError(Exception):
    """Raised when a write cannot be previewed safely. The write is then refused."""


def build_preview(sql: str, dialect: str = "postgres") -> dict:
    """Turn a validated INSERT/UPDATE/DELETE into a description of what it would touch.
    UPDATE/DELETE -> a SELECT of the rows that match the WHERE clause (the 'before' image).
    INSERT ... VALUES -> the literal rows to be added.
    INSERT ... SELECT -> that SELECT.
    Forms we cannot preview faithfully are refused (fail closed)."""
    exp = sqlglot.exp
    parsed = sqlglot.parse_one(sql, dialect=dialect)
    kind = classify_statement(parsed)

    if kind in ("update", "delete"):
        if parsed.args.get("from_") or parsed.args.get("from") or parsed.args.get("using"):
            raise PreviewError(f"{kind.upper()} with FROM/USING is not supported yet.")
        table = parsed.this
        select = exp.select("*").from_(table.copy())
        where = parsed.args.get("where")
        if where is not None:
            select = select.where(where.this.copy())
        info = {"kind": kind, "table": table.name, "has_where": where is not None,
                "preview_sql": select.sql(dialect=dialect)}
        if kind == "update":
            # same rows plus the value each SET would write (display only)
            sets = []
            for eq in parsed.args.get("expressions") or []:
                if isinstance(eq, exp.EQ) and isinstance(eq.this, exp.Column):
                    sets.append(exp.alias_(eq.expression.copy(), f"new_{eq.this.name}"))
            if sets:
                after = exp.select("*", *sets).from_(table.copy())
                if where is not None:
                    after = after.where(where.this.copy())
                info["after_sql"] = after.sql(dialect=dialect)
        return info
    if kind == "insert":
        if parsed.args.get("conflict"):
            raise PreviewError("INSERT ... ON CONFLICT is not supported yet.")
        target = parsed.this
        is_schema = isinstance(target, exp.Schema)
        table = (target.this if is_schema else target).name
        columns = [c.name for c in target.expressions] if is_schema else []
        source = parsed.args.get("expression")
        if isinstance(source, exp.Values):
            rows = [[e.sql(dialect=dialect) for e in tup.expressions]
                    for tup in source.expressions]
            return {"kind": kind, "table": table, "has_where": True,
                    "columns": columns, "literal_rows": rows}
        if isinstance(source, (exp.Select, exp.SetOperation)):
            return {"kind": kind, "table": table, "has_where": True,
                    "preview_sql": source.sql(dialect=dialect)}
        raise PreviewError("This INSERT form is not supported yet.")

    raise PreviewError(f"Cannot preview a {kind} statement.")


def run_preview(engine, info: dict, dialect: str = "postgres") -> dict:
    """Run the (read-only) preview and collect warnings. Never modifies data."""
    kind, table = info["kind"], info["table"]
    columns, rows, total = [], [], 0

    if info.get("preview_sql"):
        psql = info["preview_sql"]
        # defence in depth: the preview itself must be one plain read
        checked = sqlglot.parse(psql, dialect=dialect)
        if len(checked) != 1 or classify_statement(checked[0]) != "read":
            raise PreviewError("Could not build a safe preview.")
        with engine.connect() as conn:
            total = conn.execute(_as_text(f"SELECT COUNT(*) FROM ({psql}) AS _preview")).scalar()
            res = conn.execute(_as_text(f"SELECT * FROM ({psql}) AS _preview LIMIT {PREVIEW_SAMPLE}"))
            columns = list(res.keys())
            rows = [list(r) for r in res.fetchall()]
            after_sql = info.get("after_sql")
            if after_sql:
                try:
                    chk = sqlglot.parse(after_sql, dialect=dialect)
                    if len(chk) == 1 and classify_statement(chk[0]) == "read":
                        res2 = conn.execute(_as_text(
                            f"SELECT * FROM ({after_sql}) AS _preview LIMIT {PREVIEW_SAMPLE}"))
                        columns = list(res2.keys())
                        rows = [list(r) for r in res2.fetchall()]
                except Exception:
                    pass  # fall back to the plain before-image
    else:
        rows = info["literal_rows"]
        total = len(rows)
        columns = info.get("columns") or []
        if not columns:
            try:
                real = [c["name"] for c in inspect(engine).get_columns(table)]
            except Exception:
                real = []
            width = len(rows[0]) if rows else 0
            columns = real if len(real) == width else [f"col{i + 1}" for i in range(width)]

    warnings = []
    if kind in ("update", "delete"):
        if not info["has_where"]:
            warnings.append(f"No WHERE clause: this will change ALL rows in {table}.")
        if total == 0:
            warnings.append("No rows match, so nothing would change.")
    elif total == 0:
        warnings.append("No rows would be inserted.")
    if total > WRITE_ROW_WARN:
        warnings.append(f"This affects {total} rows (more than {WRITE_ROW_WARN}).")

    return {"table": table, "preview_sql": info.get("preview_sql"), "columns": columns,
            "rows": rows, "total": total, "has_where": info["has_where"],
            "warnings": warnings}

# --- Confidence ---
def score_confidence(sql: str, schema: str, valid: bool) -> float:
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
        referenced = {t.name.lower() for t in parsed.find_all(sqlglot.exp.Table)}
    except:
        referenced = set()

    retrieved = set()
    for line in schema.split("\n"):
        if line.startswith("- Table"):
            retrieved.add(line.split(" ")[2].lower())

    coverage = len(referenced & retrieved) / max(len(referenced), 1)
    joins = sql.upper().count("JOIN")
    subs = sql.upper().count("SELECT") - 1
    complexity = 1.0 if joins == 0 and subs == 0 else 0.6 if joins <= 2 else 0.3
    valid_bonus = 1.0 if valid else 0.0
    return round(coverage * 0.5 + complexity * 0.3 + valid_bonus * 0.2, 2)

# --- Clean SQL ---
def clean_sql(raw: str) -> str:
    raw = re.sub(r"```sql|```", "", raw)
    raw = raw.replace("`", '"')
    return raw.strip()

# --- Extract SQL ---
# Finds where the SQL starts. Write statements are matched only in their full
# opening form (INSERT INTO / UPDATE x SET / DELETE FROM) so ordinary words in
# a chatty reply ("to update the price...") are not mistaken for SQL. Whether a
# write is *allowed* is decided later, in validate_sql, not here.
_SQL_START = re.compile(
    r"\b(SELECT\b"
    r"|WITH\s+(?:RECURSIVE\s+)?[\w.\"]+\s*(?:\([^)]*\)\s*)?AS\s*(?:NOT\s+MATERIALIZED\s+|MATERIALIZED\s+)?\("
    r"|INSERT\s+INTO\b"
    r"|UPDATE\s+[\w.\"]+(?:\s+(?:AS\s+)?\w+)?\s+SET\b|DELETE\s+FROM\b)",
    re.IGNORECASE,
)

def extract_sql(raw: str) -> str:
    raw = clean_sql(raw or "")
    m = _SQL_START.search(raw)
    return raw[m.start():].strip() if m else ""


# --- Main query function ---
def run_query(
    user_input: str,
    db_url: str,
    api_key: str,
    model: str = os.getenv("LLM_MODEL_GROQ", "openai/gpt-oss-120b"),
    base_url: str = "https://api.groq.com/openai/v1",
    history: list = [],
    allow_writes: bool = False
) -> dict:
    engine = create_engine(db_url)
    dialect = "sqlite" if "sqlite" in db_url else "postgres"
    client = OpenAI(api_key=api_key, base_url=base_url)

    # intent
    intent = classify_intent(user_input)

    # schema + RAG
    chunks = get_schema(engine)
    known_tables = {c["table"].lower() for c in chunks}
    known_tables.update({"columns", "tables", "information_schema.columns"})
    top_k = len(chunks) if intent["intent"] == "ambiguous" else 6
    schema = retrieve_schema(user_input, chunks, top_k=top_k)

    # extra instructions only when the visitor has enabled edits
    write_note = ""
    if allow_writes:
        write_note = """

Data changes are enabled. If the user asks to add, change or remove data, return exactly ONE INSERT, UPDATE or DELETE statement (the query hint above does not apply to it). Rules: use only tables and columns from the schema; list column names explicitly in INSERT and put all new rows in one multi-row VALUES list; give every UPDATE and DELETE a WHERE clause unless the user clearly asks for all rows; never use DDL (CREATE, ALTER, DROP, TRUNCATE); never invent values the user did not give. If the user only asks to see data, return a SELECT as usual."""

    # build messages
    messages = [
        {"role": "system", "content": f"""You are a {dialect.upper()} expert. Given this schema:
{schema}

Query hint: {intent['hint']}

Use {dialect} syntax only. Return only the SQL query, nothing else.{write_note}"""}
    ]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_input})

    # LLM call
    response = client.chat.completions.create(model=model, messages=messages)
    raw = response.choices[0].message.content or ""
    sql = extract_sql(raw)

    # guard
    if not sql and re.match(r"\s*(DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b", raw, re.IGNORECASE):
        return {"sql": "", "rows": [], "intent": intent["intent"], "confidence": 0.0,
                "error": None,
                "notice": "Blocked: schema changes (DROP, ALTER, CREATE, TRUNCATE) are never allowed. "
                          "Only INSERT, UPDATE and DELETE are supported."}
    if not sql:
        return {"sql": "", "rows": [], "intent": intent["intent"],
                "confidence": 0.0,
                "error": None, "notice": raw.strip()[:300] or "Could not generate SQL — try rephrasing."}
    # validate
    validation = validate_sql(sql, known_tables, dialect=dialect, allow_writes=allow_writes)
    if (not validation["valid"] and validation["kind"] is None
            and validation["error"].startswith("Syntax error")):
        # the model most likely answered in prose: show its text, not a parser error
        return {"sql": "", "rows": [], "intent": intent["intent"], "confidence": 0.0,
                "error": None, "notice": raw.strip()[:300]}
    if not validation["valid"]:
        return {"sql": sql, "rows": [], "intent": intent["intent"],
                "confidence": 0.0, "error": validation["error"]}

    confidence = score_confidence(sql, schema, True)

    # writes are never executed here: they are previewed (read-only) and handed
    # back for the confirm -> transaction flow (next step of v2)
    if validation["kind"] != "read":
        try:
            preview = run_preview(engine, build_preview(sql, dialect), dialect)
        except PreviewError as e:
            return {"sql": sql, "rows": [], "intent": intent["intent"],
                    "confidence": confidence, "error": f"Cannot preview this write: {e}"}
        except Exception as e:
            return {"sql": sql, "rows": [], "intent": intent["intent"],
                    "confidence": confidence, "error": str(e)}
        return {"sql": sql, "rows": [], "kind": validation["kind"],
                "pending_write": True, "preview": preview,
                "intent": intent["intent"], "confidence": confidence, "error": None}

    # execute
    try:
        with engine.connect() as conn:
            result = conn.execute(_as_text(sql))
            rows = [list(r) for r in result.fetchall()]
            columns = list(result.keys())
    except Exception as e:
        return {"sql": sql, "rows": [], "intent": intent["intent"],
                "confidence": confidence, "error": str(e)}

    return {
        "sql": sql,
        "rows": rows,
        "columns": columns,
        "kind": "read",
        "intent": intent["intent"],
        "confidence": confidence,
        "warning": "Low confidence — verify results" if confidence < 0.5 else None,
        "error": None
    }

# --- Write execution (v2) ---
AUDIT_DB_PATH = os.getenv("ASKYOURDB_AUDIT_DB",
                          os.path.join(tempfile.gettempdir(), "askyourdb_audit.db"))
AUDIT_OLD_ROWS_MAX = 1000   # max rows of old values stored per audit entry


class WriteAborted(Exception):
    """Raised inside the transaction to abort a write that no longer matches the preview."""


def _audit_conn():
    conn = sqlite3.connect(AUDIT_DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS write_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, session_id TEXT, database TEXT, kind TEXT, table_name TEXT,
        sql TEXT, expected_rows INTEGER, rows_affected INTEGER,
        old_values TEXT, old_values_truncated INTEGER,
        status TEXT, error TEXT)""")
    return conn


def _db_label(db_url: str) -> str:
    """Host/database only. Never user names or passwords."""
    u = make_url(db_url)
    return f"{u.get_backend_name()}://{u.host or ''}/{u.database or ''}"


def _audit_start(session_id, db_url, kind, table, sql, expected) -> int:
    with _audit_conn() as conn:   # the with-block commits
        cur = conn.execute(
            "INSERT INTO write_audit (ts, session_id, database, kind, table_name, sql,"
            " expected_rows, status) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), session_id,
             _db_label(db_url), kind, table, sql, expected, "started"))
        return cur.lastrowid


def _audit_finish(audit_id, status, rows_affected=None, old_rows=None,
                  truncated=False, error=None):
    old = json.dumps(old_rows, default=str) if old_rows is not None else None
    with _audit_conn() as conn:
        conn.execute(
            "UPDATE write_audit SET status=?, rows_affected=?, old_values=?,"
            " old_values_truncated=?, error=? WHERE id=?",
            (status, rows_affected, old, int(truncated), error, audit_id))


def get_audit_log(session_id: str | None = None, limit: int = 20) -> list[dict]:
    """Most recent audit entries, newest first (optionally for one session)."""
    with _audit_conn() as conn:
        conn.row_factory = sqlite3.Row
        if session_id:
            rows = conn.execute("SELECT * FROM write_audit WHERE session_id=?"
                                " ORDER BY id DESC LIMIT ?", (session_id, limit))
        else:
            rows = conn.execute("SELECT * FROM write_audit ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows.fetchall()]


def execute_write(db_url: str, sql: str, expected_total: int,
                  session_id: str = "", allow_all_rows: bool = False) -> dict:
    """Run ONE confirmed write inside a transaction.

    Safety rules, all enforced here (not only in the UI):
      1. the SQL is re-validated from scratch (allowlist, one statement, real tables)
      2. UPDATE/DELETE without WHERE needs allow_all_rows=True
      3. the intent is written to the audit log BEFORE anything changes; no audit, no write
      4. inside the transaction the matching row count must equal what the user
         confirmed (expected_total), otherwise nothing is written
      5. after the write, rows affected must equal that count, otherwise it is rolled back
      6. any error rolls the whole transaction back
    Returns {"ok", "status", "rows_affected", "audit_id", "error"}.
    status is one of: committed, aborted, rolled_back, refused."""
    def result(ok, status, affected=None, audit_id=None, error=None):
        return {"ok": ok, "status": status, "rows_affected": affected,
                "audit_id": audit_id, "error": error}

    engine = create_engine(db_url)
    dialect = "sqlite" if "sqlite" in db_url else "postgres"

    # 1. validate again from scratch
    chunks = get_schema(engine)
    known_tables = {c["table"].lower() for c in chunks}
    validation = validate_sql(sql, known_tables, dialect=dialect, allow_writes=True)
    if not validation["valid"]:
        return result(False, "refused", error=validation["error"])
    if validation["kind"] == "read":
        return result(False, "refused", error="Not a write statement.")
    try:
        info = build_preview(sql, dialect)
    except PreviewError as e:
        return result(False, "refused", error=f"Cannot preview this write: {e}")

    # 2. all-rows guard
    if info["kind"] in ("update", "delete") and not info["has_where"] and not allow_all_rows:
        return result(False, "refused",
                      error="No WHERE clause: this needs the explicit all-rows confirmation.")

    # 3. write-ahead audit entry
    try:
        audit_id = _audit_start(session_id, db_url, info["kind"], info["table"], sql, expected_total)
    except Exception as e:
        return result(False, "refused", error=f"Audit log unavailable, write not executed: {e}")

    old_rows, truncated = None, False
    try:
        with engine.begin() as conn:   # commits on success, rolls back on any exception
            # 4. count inside the transaction and compare with what was confirmed
            if info.get("preview_sql"):
                psql = info["preview_sql"]
                total = conn.execute(_as_text(f"SELECT COUNT(*) FROM ({psql}) AS _p")).scalar()
                if info["kind"] in ("update", "delete"):
                    res = conn.execute(_as_text(
                        f"SELECT * FROM ({psql}) AS _p LIMIT {AUDIT_OLD_ROWS_MAX}"))
                    cols = list(res.keys())
                    old_rows = [dict(zip(cols, r)) for r in res.fetchall()]
                    truncated = total > AUDIT_OLD_ROWS_MAX
            else:
                total = len(info["literal_rows"])
            if total != expected_total:
                raise WriteAborted(f"The data changed since the preview ({total} rows now "
                                   f"match, {expected_total} were confirmed). Run it again.")

            # the write itself
            affected = conn.execute(_as_text(sql)).rowcount

            # 5. what happened must equal what was confirmed
            if affected != total:
                raise WriteAborted(f"The write affected {affected} rows but {total} were "
                                   f"previewed. Rolled back.")
    except WriteAborted as e:
        _audit_finish(audit_id, "aborted", error=str(e))
        return result(False, "aborted", audit_id=audit_id, error=str(e))
    except Exception as e:   # 6. anything else: the transaction has been rolled back
        _audit_finish(audit_id, "rolled_back", error=str(e))
        return result(False, "rolled_back", audit_id=audit_id, error=str(e))

    _audit_finish(audit_id, "committed", rows_affected=affected,
                  old_rows=old_rows, truncated=truncated)
    return result(True, "committed", affected, audit_id)


# --- Explain ---
def explain_sql(sql: str, api_key: str, model: str, base_url: str) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Explain this SQL query in plain English. 2-3 sentences max."},
            {"role": "user", "content": sql}
        ]
    )
    return response.choices[0].message.content



# --- CSV file ---
def load_file_to_sqlite(file) -> tuple[str, str]:
    """Load CSV/XLSX into a temp SQLite DB, return (engine_url, table_name)"""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # sanitise table name from filename
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', file.name.rsplit('.', 1)[0].lower())

    # write to temp SQLite file
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(tmp.name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    return f"sqlite:///{tmp.name}", table_name


# --- Multi-file loader (v2) ---
def _clean_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", str(raw).strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or name[0].isdigit():
        name = "t_" + name
    return name


def _unique(name: str, used: set) -> str:
    base, i = name, 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def load_files_to_sqlite(files) -> tuple[str, list[dict]]:
    """Load several CSV/XLSX files (every sheet) into ONE temp SQLite DB.
    Returns (engine_url, tables), where tables is a list of
    {"table": ..., "source": ..., "rows": ..., "columns": [...]}."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    used_tables: set = set()
    tables: list[dict] = []

    for f in files:
        stem = f.name.rsplit(".", 1)[0]
        if f.name.lower().endswith(".csv"):
            sheets = {None: pd.read_csv(f)}
        else:
            sheets = pd.read_excel(f, sheet_name=None)  # all sheets

        for sheet, df in sheets.items():
            if df.shape[1] == 0:
                continue  # skip empty sheets
            label = stem if (sheet is None or len(sheets) == 1) else f"{stem}_{sheet}"
            table = _unique(_clean_name(label), used_tables)

            cols_used: set = set()
            df.columns = [_unique(_clean_name(c), cols_used) for c in df.columns]

            df.to_sql(table, conn, if_exists="replace", index=False)
            source = f.name if sheet is None else f"{f.name} [{sheet}]"
            tables.append({"table": table, "source": source,
                           "rows": len(df), "columns": list(df.columns)})

    conn.close()
    return f"sqlite:///{tmp.name}", tables