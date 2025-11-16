
import uuid
import time
from datetime import datetime

_sessions = {}

class DebugSession:
    def __init__(self, session_id=None):
        self.id = session_id or str(uuid.uuid4())
        self.history = []
        self.created_at = time.time()
        self.fixes_applied = 0

    def push(self, entry):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "data": entry
        })
        if "patch_result" in entry and entry["patch_result"].get("success"):
            self.fixes_applied += 1

    @classmethod
    def get(cls, session_id=None):
        if not session_id:
            new_session = DebugSession()
            _sessions[new_session.id] = new_session
            return new_session
        return _sessions.get(session_id, DebugSession(session_id))
