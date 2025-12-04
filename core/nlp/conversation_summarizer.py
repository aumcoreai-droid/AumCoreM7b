# Conversation Summarizer – Phase-1 Rule-based
from typing import List

def summarize_conversation(messages: List[str]) -> str:
    if not messages:
        return ""
    summary = " ".join(messages[-3:])  # Simple last 3 messages summary
    return summary

# Test
if __name__ == "__main__":
    chat = ["Hi!", "Hello, how are you?", "I'm fine, thanks!", "Can you help me with a task?"]
    print("Conversation summary:", summarize_conversation(chat))
