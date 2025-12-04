# Dual Language Reply – Phase-1 Rule-based
def generate_dual_reply(text: str, language: str = "en") -> str:
    # Simple rule-based dual language reply
    replies_en = {
        "greeting": "Hello! How can I help you?",
        "farewell": "Goodbye! Take care!",
        "thanks": "You're welcome!",
        "unknown": "I didn't understand that.",
    }
    replies_hi = {
        "greeting": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
        "farewell": "अलविदा! ध्यान रखना!",
        "thanks": "आपका स्वागत है!",
        "unknown": "मुझे समझ नहीं आया।",
    }
    # Determine intent simplistically
    intent = "unknown"
    keywords = {"hi": "greeting", "hello": "greeting", "bye": "farewell", "thanks": "thanks"}
    for k, v in keywords.items():
        if k in text.lower():
            intent = v
            break
    return replies_hi[intent] if language.lower() == "hi" else replies_en[intent]

# Test
if __name__ == "__main__":
    samples = [("hi there", "en"), ("thanks a lot", "en"), ("bye!", "hi"), ("unknown text", "hi")]
    for txt, lang in samples:
        print(f"Text: {txt} | Language: {lang} | Reply: {generate_dual_reply(txt, lang)}")
