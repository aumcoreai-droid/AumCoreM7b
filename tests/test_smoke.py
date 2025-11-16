
def test_debug_router():
    from core.debug_router import handle_debug_request
    result = handle_debug_request({"test": True})
    assert "classification" in result

def test_error_classifier():
    from core.error_classifier import classify_error
    result = classify_error("TypeError: test")
    assert result["error_type"] == "TypeError"
