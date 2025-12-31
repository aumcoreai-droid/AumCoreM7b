# long_term_memory.py - Persistent Code Memory
import json
import hashlib
from datetime import datetime

class LongTermMemory:
    def __init__(self, file_path="code_memory.json"):
        self.file_path = file_path
        self.memory = self.load_memory()
    
    def load_memory(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"code_snippets": [], "patterns": []}
    
    def save_memory(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def add_code_snippet(self, code, description, language="python"):
        snippet_id = hashlib.md5(code.encode()).hexdigest()[:10]
        snippet = {
            "id": snippet_id,
            "code": code,
            "description": description,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
        self.memory["code_snippets"].append(snippet)
        self.save_memory()
        return snippet_id
    
    def find_similar_code(self, query):
        # Simple keyword matching (can be enhanced with embeddings)
        results = []
        for snippet in self.memory["code_snippets"]:
            if query.lower() in snippet["description"].lower():
                results.append(snippet)
        return results[:5]
