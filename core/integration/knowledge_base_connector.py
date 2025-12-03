# Phase-1 Rule-based Knowledge Base Connector Skeleton
class KnowledgeBaseConnector:
    def __init__(self):
        # Simple in-memory knowledge base
        self.kb = {
            "hours": "We are open from 9 AM to 6 PM.",
            "location": "Our office is located in Nagpur, Maharashtra.",
            "contact": "You can reach us at aumcoreai@gmail.com."
        }

    def query(self, key):
        """Return answer from KB if available."""
        if key in self.kb:
            return self.kb[key]
        return "Sorry, I don't have information about that."
