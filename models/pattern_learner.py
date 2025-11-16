
def learn_patterns(feedback_records):
    patterns = []
    for record in feedback_records:
        if record.get("approved", False):
            patterns.append({
                "error_type": record.get("error_type"),
                "fix_pattern": record.get("fix_applied"),
                "confidence": 0.8
            })
    return patterns
