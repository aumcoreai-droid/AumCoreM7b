
import subprocess
import tempfile
import shutil
import os

def run_in_sandbox(code_dir, test_command="python -m pytest -v", timeout=10):
    try:
        result = subprocess.run(
            test_command,
            shell=True,
            cwd=code_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Timeout: Process took too long",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }
