# NLU Engine – Phase-1 Rule-based
def detect_intent(text: str) -> dict:
    # Simple rule-based intent detection
    rules = {
        "hi": "greeting",
        "hello": "greeting",
        "bye": "farewell",
        "thanks": "thanks",
    }
    for k, v in rules.items():
        if k in text.lower():
            return {"intent": v, "entities": []}
    return {"intent": "unknown", "entities": []}

# Test
if __name__ == "__main__":
    samples = ["hi there!", "thanks for your help", "goodbye!"]
    for s in samples:
        print(f"Text: {s} | Detected: {detect_intent(s)}")
