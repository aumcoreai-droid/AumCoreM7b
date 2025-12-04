# Task Understanding – Phase-1 Rule-based
from typing import List

def understand_task(text: str, task_keywords: List[str]) -> List[str]:
    matched_tasks = [task for task in task_keywords if task.lower() in text.lower()]
    return matched_tasks

# Test
if __name__ == "__main__":
    tasks = ["email", "report", "meeting", "schedule"]
    sample_text = "Please prepare the report and schedule the meeting."
    print("Detected tasks:", understand_task(sample_text, tasks))
