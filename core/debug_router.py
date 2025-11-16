
from core.error_classifier import classify_error
from rag.retrieval_engine import retrieve_context
from core.fix_generator import generate_fix
from sandbox.diff_tester import apply_patch_and_test
from core.context_manager import DebugSession

def handle_debug_request(payload):
    session = DebugSession.get(payload.get("session_id"))
    classification = classify_error(payload.get("stack_trace",""), payload.get("surrounding_code",""))
    retrieved = retrieve_context(payload, classification)
    diff, explanation, tests = generate_fix(payload, classification, retrieved)
    result = apply_patch_and_test(payload.get("repo_path","/tmp/project"), diff, test_command="pytest -q")
    session.push({"payload": payload, "classification": classification, "patch_result": result})
    return {
        "classification": classification,
        "patch_result": result,
        "patch": diff,
        "explanation": explanation
    }
