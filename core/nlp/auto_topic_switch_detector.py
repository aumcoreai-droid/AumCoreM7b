# Auto Topic Switch Detector – Phase-1 Rule-based
from typing import List

class AutoTopicSwitchDetector:
    def __init__(self):
        self.topic_history: List[str] = []

    def detect_switch(self, new_topic: str) -> bool:
        new_topic = new_topic.strip().lower()
        if not new_topic:
            return False
        if self.topic_history and new_topic != self.topic_history[-1]:
            self.topic_history.append(new_topic)
            if len(self.topic_history) > 5:
                self.topic_history.pop(0)
            return True
        self.topic_history.append(new_topic)
        return False

    def get_history(self) -> List[str]:
        return self.topic_history

# Test
if __name__ == "__main__":
    detector = AutoTopicSwitchDetector()
    print("Switch detected?", detector.detect_switch("AI"))
    print("Switch detected?", detector.detect_switch("Python"))
    print("History:", detector.get_history())
