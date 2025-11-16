
import ast

def analyze_code(src):
    try:
        tree = ast.parse(src)
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        return {"functions": funcs, "node_count": len(list(ast.walk(tree)))}
    except:
        return {"functions": [], "node_count": 0}
