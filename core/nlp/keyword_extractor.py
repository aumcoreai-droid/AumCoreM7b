# Keyword Extractor – Phase-1 Rule-based
from typing import List, Dict

def extract_keywords(text: str, keyword_list: List[str]) -> List[str]:
    detected = [kw for kw in keyword_list if kw.lower() in text.lower()]
    return detected

# Test
if __name__ == "__main__":
    keywords = ["ai", "chatbot", "nlp", "intent", "dialogue"]
    sample_text = "This AI chatbot handles dialogue and detects intent using NLP."
    result = extract_keywords(sample_text, keywords)
    print("Detected keywords:", result)
