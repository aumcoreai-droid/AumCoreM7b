
import re

def scan_for_secrets(text):
    secret_patterns = [
        r'AKIA[0-9A-Z]{16}',
        r'ghp_[A-Za-z0-9_\-]{36}',
        r'sk-[A-Za-z0-9]{48}'
    ]

    for pattern in secret_patterns:
        if re.search(pattern, text):
            return True
    return False

def validate_code_safety(code):
    dangerous_patterns = [
        "os.system",
        "subprocess.Popen",
        "__import__",
        "eval(",
        "exec("
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return False
    return True
