# chain_of_thought_manager.py - Step-by-Step Code Generation
class ChainOfThought:
    def __init__(self):
        self.steps = []
    
    def generate_thoughts(self, user_input):
        self.steps = [
            f"Step 1: Analyze user request: '{user_input}'",
            "Step 2: Determine required modules and dependencies",
            "Step 3: Design architecture and file structure",
            "Step 4: Write core functionality with error handling",
            "Step 5: Add documentation and comments",
            "Step 6: Test integration points",
            "Step 7: Optimize for performance",
            "Step 8: Prepare for deployment"
        ]
        return "\n".join(self.steps)
    
    def get_current_step(self):
        return self.steps[-1] if self.steps else "No active steps"
    
    def reset(self):
        self.steps = []
