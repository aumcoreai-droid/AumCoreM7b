# Mini Knowledge Synthesizer – Phase-1 Rule-based
from typing import Dict

class MiniKnowledgeSynthesizer:
    def __init__(self):
        self.knowledge: Dict[str, str] = {}

    def add_fact(self, key: str, value: str):
        self.knowledge[key.lower()] = value

    def query(self, key: str) -> str:
        return self.knowledge.get(key.lower(), "Fact not found.")

# Test
if __name__ == "__main__":
    mks = MiniKnowledgeSynthesizer()
    mks.add_fact("Python", "A programming language.")
    mks.add_fact("AI", "Artificial Intelligence field.")
    print("Query Python:", mks.query("Python"))
    print("Query AI:", mks.query("AI"))
    print("Query ML:", mks.query("ML"))
