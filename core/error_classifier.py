
import json

def classify_error(stack_trace, surrounding_code=""):
    if "TypeError" in stack_trace:
        return {"error_type":"TypeError","severity":"medium","probable_cause":"type mismatch","short_summary":"Type mismatch detected"}
    if "ZeroDivisionError" in stack_trace:
        return {"error_type":"ZeroDivision","severity":"high","probable_cause":"division by zero","short_summary":"Division by zero error"}
    if "Traceback" in stack_trace or "Exception" in stack_trace:
        return {"error_type":"Exception","severity":"high","probable_cause":"runtime exception","short_summary":"Exception in runtime"}
    return {"error_type":"Unknown","severity":"low","probable_cause":"unknown","short_summary":"No clear trace"}
