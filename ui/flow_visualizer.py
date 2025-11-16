
def render_flow(nodes, edges):
    mermaid = "graph TD\n"
    for node in nodes:
        mermaid += f'    {node["id"]}["{node["label"]}"]\n'
    for edge in edges:
        mermaid += f'    {edge["from"]} --> {edge["to"]}\n'
    return mermaid

def generate_debug_flow(debug_data):
    nodes = [
        {"id": "start", "label": "Debug Request"},
        {"id": "classify", "label": "Error Classification"},
        {"id": "retrieve", "label": "Context Retrieval"},
        {"id": "generate", "label": "Fix Generation"},
        {"id": "test", "label": "Sandbox Test"},
        {"id": "end", "label": "Result"}
    ]

    edges = [
        {"from": "start", "to": "classify"},
        {"from": "classify", "to": "retrieve"},
        {"from": "retrieve", "to": "generate"},
        {"from": "generate", "to": "test"},
        {"from": "test", "to": "end"}
    ]

    return render_flow(nodes, edges)
