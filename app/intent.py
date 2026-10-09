from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

INTENT_LABELS = ["aggregation", "filter", "join", "ambiguous"]

INTENT_EXAMPLES = {
    "aggregation": "Use GROUP BY, COUNT, SUM, AVG, or other aggregate functions.",
    "filter": "Use WHERE clause to filter rows based on conditions.",
    "join": "Use JOIN to combine data from multiple tables.",
    "ambiguous": "The question may be vague. Make the most reasonable assumption and still return a SQL query. Never ask for clarification.",
}

def classify_intent(user_query: str) -> dict:
    result = classifier(user_query, candidate_labels=INTENT_LABELS)
    intent = result["labels"][0]
    score = result["scores"][0]
    return {
        "intent": intent,
        "score": round(score, 3),
        "hint": INTENT_EXAMPLES[intent]
    }

if __name__ == "__main__":
    test_queries = [
        "show me total sales by region",
        "find all customers from Germany",
        "show me orders with their customer names",
        "what"
    ]
    for q in test_queries:
        result = classify_intent(q)
        print(f"Query: {q}")
        print(f"Intent: {result['intent']} (confidence: {result['score']})")
        print(f"Hint: {result['hint']}\n")