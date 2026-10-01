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
    "ambiguous": "Ask the user to clarify their question."
}

BLOCKED_OPS = {"DROP", "DELETE", "ALTER", "TRUNCATE", "INSERT", "UPDATE", "CREATE"}

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
def validate_sql(sql: str, known_tables: set, dialect: str = "postgres") -> dict:
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError as e:
        return {"valid": False, "error": f"Syntax error: {e}"}
    if parsed.key.upper() in BLOCKED_OPS:
        return {"valid": False, "error": f"Operation not allowed: {parsed.key.upper()}"}

    referenced = {t.name.lower() for t in parsed.find_all(sqlglot.exp.Table)}
    unknown = referenced - known_tables
    if unknown:
        return {"valid": False, "error": f"Unknown tables: {unknown}"}

    return {"valid": True, "error": None}

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

# --- Main query function ---
def run_query(
    user_input: str,
    db_url: str,
    api_key: str,
    model: str = os.getenv("LLM_MODEL_GROQ", "openai/gpt-oss-120b"),
    base_url: str = "https://api.groq.com/openai/v1",
    history: list = []
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
    sql = clean_sql(response.choices[0].message.content)

    # guard
    if not sql.strip().upper().startswith(("SELECT", "WITH", "EXPLAIN")):
        return {"sql": "", "rows": [], "intent": intent["intent"],
                "confidence": 0.0, "error": "Could not generate SQL — try rephrasing."}

    # validate
    validation = validate_sql(sql, known_tables, dialect=dialect)
    if not validation["valid"]:
        return {"sql": sql, "rows": [], "intent": intent["intent"],
                "confidence": 0.0, "error": validation["error"]}

    # execute
    confidence = score_confidence(sql, schema, True)
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


