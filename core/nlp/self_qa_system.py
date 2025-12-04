# Self QA System – Phase-1 Rule-based
from typing import Dict

class SelfQASystem:
    def __init__(self):
        self.qa_pairs: Dict[str, str] = {}

    def add_qa(self, question: str, answer: str):
        self.qa_pairs[question.lower()] = answer

    def answer(self, question: str) -> str:
        return self.qa_pairs.get(question.lower(), "I don't know.")

# Test
if __name__ == "__main__":
    qa = SelfQASystem()
    qa.add_qa("What is AI?", "Artificial Intelligence.")
    qa.add_qa("What is Python?", "A programming language.")
    print("Answer:", qa.answer("What is AI?"))
    print("Answer:", qa.answer("What is Python?"))
    print("Answer:", qa.answer("What is ML?"))
