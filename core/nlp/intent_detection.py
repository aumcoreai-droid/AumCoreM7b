# Intent Detection – Phase-1 Rule-based
from typing import List, Dict

def detect_intent(text: str, keywords: Dict[str, str]) -> Dict[str, List[str]]:
    # Simple keyword-based intent detection
    detected_intent = "unknown"
    entities = []
    for k, v in keywords.items():
        if k in text.lower():
            detected_intent = v
            entities.append(k)
    return {"intent": detected_intent, "entities": entities}

# Test
if __name__ == "__main__":
    sample_keywords = {"hi": "greeting", "hello": "greeting", "bye": "farewell", "thanks": "thanks"}
    samples = ["hi there!", "thanks for your help", "goodbye!"]
    for s in samples:
        result = detect_intent(s, sample_keywords)
        print(f"Text: {s} | Intent: {result['intent']} | Entities: {result['entities']}")
