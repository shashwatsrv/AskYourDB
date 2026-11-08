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
        return {"valid": False, "error": f"Syntax error: {e}", "kind": None}
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
    r"\b(SELECT\b|WITH\b|INSERT\s+INTO\b|UPDATE\s+\S+\s+SET\b|DELETE\s+FROM\b)",
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

    # build messages
    messages = [
        {"role": "system", "content": f"""You are a {dialect.upper()} expert. Given this schema:
{schema}

Query hint: {intent['hint']}

Use {dialect} syntax only. Return only the SQL query, nothing else."""}
    ]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_input})

    # LLM call
    response = client.chat.completions.create(model=model, messages=messages)
    raw = response.choices[0].message.content or ""
    sql = extract_sql(raw)

    # guard
    if not sql:
        return {"sql": "", "rows": [], "intent": intent["intent"],
                "confidence": 0.0,
                "error": raw.strip()[:300] or "Could not generate SQL — try rephrasing."}
    # validate
    validation = validate_sql(sql, known_tables, dialect=dialect, allow_writes=allow_writes)
    if not validation["valid"]:
        return {"sql": sql, "rows": [], "intent": intent["intent"],
                "confidence": 0.0, "error": validation["error"]}

    confidence = score_confidence(sql, schema, True)

    # writes are never executed here: they are handed back for the
    # preview -> confirm -> transaction flow (next step of v2)
    if validation["kind"] != "read":
        return {"sql": sql, "rows": [], "kind": validation["kind"],
                "pending_write": True, "intent": intent["intent"],
                "confidence": confidence, "error": None}

    # execute
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
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