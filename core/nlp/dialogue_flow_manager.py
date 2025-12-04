# Dialogue Flow Manager – Phase-1 Rule-based
from typing import List, Dict

class DialogueFlowManager:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_turn(self, user_text: str, bot_text: str):
        self.history.append({"user": user_text, "bot": bot_text})

    def get_last_bot_reply(self) -> str:
        return self.history[-1]["bot"] if self.history else ""

# Test
if __name__ == "__main__":
    dfm = DialogueFlowManager()
    dfm.add_turn("Hi!", "Hello! How can I help?")
    dfm.add_turn("Thanks!", "You're welcome!")
    print("Last bot reply:", dfm.get_last_bot_reply())
    print("Dialogue history:", dfm.history)
