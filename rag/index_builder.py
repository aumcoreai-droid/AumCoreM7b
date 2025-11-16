
from .vector_store import vector_store

def build_index(kb_paths):
    sample_fixes = [
        {
            "id": "fix_001",
            "text": "ZeroDivisionError: division by zero - Add zero check before division",
            "metadata": {"error_type": "ZeroDivision", "fix_type": "validation"}
        },
        {
            "id": "fix_002", 
            "text": "TypeError: unsupported operand type - Add type conversion",
            "metadata": {"error_type": "TypeError", "fix_type": "conversion"}
        }
    ]

    for fix in sample_fixes:
        vector_store.add(fix["id"], fix["text"], fix["metadata"])

    return len(sample_fixes)
