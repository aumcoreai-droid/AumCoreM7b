# Topic Tracker – Phase-1 Rule-based
from typing import List

class TopicTracker:
    def __init__(self):
        self.recent_topics: List[str] = []

    def add_topic(self, topic: str):
        topic = topic.strip().lower()
        if topic and topic not in self.recent_topics:
            self.recent_topics.append(topic)
            if len(self.recent_topics) > 5:
                self.recent_topics.pop(0)  # Keep only last 5 topics

    def get_recent_topics(self) -> List[str]:
        return self.recent_topics

# Test
if __name__ == "__main__":
    tt = TopicTracker()
    tt.add_topic("AI")
    tt.add_topic("Python")
    tt.add_topic("Machine Learning")
    print("Recent topics:", tt.get_recent_topics())
