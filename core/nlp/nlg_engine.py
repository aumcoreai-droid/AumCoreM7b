# NLG Engine – Phase-1 Rule-based
def generate_response(intent: str) -> str:
    # Simple rule-based response generator
    responses = {
        "greeting": "Hello! How can I assist you today?",
        "farewell": "Goodbye! Have a great day!",
        "thanks": "You're welcome!",
        "unknown": "I'm not sure how to respond to that.",
    }
    return responses.get(intent, responses["unknown"])

# Test
if __name__ == "__main__":
    test_intents = ["greeting", "thanks", "farewell", "unknown"]
    for intent in test_intents:
        print(f"Intent: {intent} | Response: {generate_response(intent)}")
