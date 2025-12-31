# short_term_memory.py - Recent Chat Buffer
from collections import deque

class ShortTermMemory:
    def __init__(self, max_size=10):
        self.memory = deque(maxlen=max_size)
    
    def add(self, user_input, ai_response):
        self.memory.append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context(self):
        return list(self.memory)
    
    def clear(self):
        self.memory.clear()
