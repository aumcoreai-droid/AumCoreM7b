
from .vector_store import vector_store

def retrieve_context(payload, classification, k=5):
    query_text = f"{classification.get('error_type', '')} {payload.get('stack_trace', '')}"
    results = vector_store.query(query_text, k=k)
    return {
        "error_type": classification.get("error_type"),
        "similar_fixes": results,
        "count": len(results)
    }
