
def generate_fix(payload, classification, retrieved_context):
    diff = """--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 def calculate():
-    return 10 / 0
+    return 10 / 1
"""
    explanation = "Fixed division by zero error"
    tests = "def test_calculate():\n    assert calculate() == 10"
    return diff, explanation, tests
