"""
AumCore_AI Memory Subsystem - Knowledge Graph Module
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/knowledge_graph.py

Description:
    यह module Phase‑1 में एक simple, in-memory, rule-based
    Knowledge Graph देता है।

    Core Idea (Phase‑1):
        - Nodes (entities, concepts) और edges (relations) को
          deterministic तरीके से manage करना।
        - कोई graph database, model-based reasoning, या advanced traversal नहीं।
        - Clean, testable, explainable behavior।

    Future (Phase‑3+):
        - Semantic relation detection via models
        - Graph-based reasoning
        - Cross‑module integration (notes, user profile, etc.)
"""

# ============================================================
# ✅ Chunk 1: Imports (PEP8)
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================
# ✅ Chunk 1: Local Logger Setup
# ============================================================

def get_kg_logger(name: str = "AumCoreAI.Memory.KnowledgeGraph") -> logging.Logger:
    """
    Knowledge graph module के लिए simple logger।
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [KNOWLEDGE_GRAPH] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_kg_logger()


# ============================================================
# ✅ Chunk 2: Basic Data Structures (Rule-Based)
# ============================================================

@dataclass
class KGNode:
    """
    Knowledge graph node (Phase‑1 simple version)।

    Fields:
        id: unique node identifier (string)
        label: human-readable label
        type: node type/category (e.g., "concept", "person", "topic")
        properties: अतिरिक्त key-value metadata
    """

    id: str
    label: str
    type: str = "concept"
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("KGNode.id non-empty string होना चाहिए।")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("KGNode.label non-empty string होना चाहिए।")
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("KGNode.type non-empty string होना चाहिए।")


@dataclass
class KGEdge:
    """
    Knowledge graph edge (Phase‑1 simple version)।

    Fields:
        id: edge identifier
        source: source node id
        target: target node id
        relation: relation type (e.g., "likes", "uses", "depends_on")
        properties: अतिरिक्त metadata
    """

    id: str
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("KGEdge.id non-empty string होना चाहिए।")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("KGEdge.source non-empty string होना चाहिए।")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("KGEdge.target non-empty string होना चाहिए।")
        if not isinstance(self.relation, str) or not self.relation.strip():
            raise ValueError("KGEdge.relation non-empty string होना चाहिए।")


@dataclass
class KGConfig:
    """
    Knowledge graph के लिए Phase‑1 configuration।
    """

    max_nodes: int = 10_000
    max_edges: int = 50_000
    allow_self_loops: bool = False
    allow_parallel_edges: bool = True

    def __post_init__(self) -> None:
        if self.max_nodes <= 0:
            raise ValueError("max_nodes positive होना चाहिए।")
        if self.max_edges <= 0:
            raise ValueError("max_edges positive होना चाहिए।")


# ============================================================
# ✅ Chunk 2: Sanitization Helpers
# ============================================================

def _sanitize_id(value: str) -> str:
    """
    IDs के लिए simple sanitization।
    """
    if value is None:
        raise ValueError("ID None नहीं हो सकता।")
    s = str(value).strip()
    if not s:
        raise ValueError("ID empty नहीं होना चाहिए।")
    return s


def _sanitize_label(value: str) -> str:
    """
    Labels के लिए simple sanitization।
    """
    if value is None:
        raise ValueError("Label None नहीं हो सकता।")
    s = str(value).strip()
    if not s:
        raise ValueError("Label empty नहीं होना चाहिए।")
    return s


def _sanitize_type(value: str, default: str = "concept") -> str:
    """
    Node/edge type sanitization।
    """
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _sanitize_relation(value: str) -> str:
    """
    Relation नाम sanitization।
    """
    if value is None:
        raise ValueError("Relation None नहीं हो सकता।")
    s = str(value).strip()
    if not s:
        raise ValueError("Relation empty नहीं होना चाहिए।")
    return s


# ============================================================
# ✅ Core Class: KnowledgeGraph (Phase‑1 Implementation)
# ============================================================

class KnowledgeGraph:
    """
    Simple in-memory knowledge graph (Phase‑1)。

    Features:
        - Nodes add/remove/update
        - Edges add/remove/update
        - Basic neighbor queries
        - Simple path exploration (1–2 hops only if needed)
        - No complex algorithms, no external DB
    """

    def __init__(self, config: Optional[KGConfig] = None) -> None:
        self.config: KGConfig = config or KGConfig()

        # Node and edge storage
        self._nodes: Dict[str, KGNode] = {}
        self._edges: Dict[str, KGEdge] = {}

        # Adjacency: node_id -> set(edge_id)
        self._outgoing_edges: Dict[str, Set[str]] = {}
        self._incoming_edges: Dict[str, Set[str]] = {}

        logger.debug("KnowledgeGraph initialized with config: %r", self.config)

    # --------------------------------------------------------
    # Public API: Node Management
    # --------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        label: str,
        type: str = "concept",
        properties: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> KGNode:
        """
        नया node add करता है (या overwrite अगर allowed हो)।
        """
        node_id = _sanitize_id(node_id)
        label = _sanitize_label(label)
        type = _sanitize_type(type)

        if not overwrite and node_id in self._nodes:
            raise ValueError(f"Node {node_id!r} already exists (overwrite=False).")

        if node_id not in self._nodes and len(self._nodes) >= self.config.max_nodes:
            raise ValueError("Max nodes limit reached, नया node add नहीं कर सकते।")

        node = KGNode(
            id=node_id,
            label=label,
            type=type,
            properties=dict(properties or {}),
        )
        self._nodes[node_id] = node

        # ensure adjacency structures
        self._outgoing_edges.setdefault(node_id, set())
        self._incoming_edges.setdefault(node_id, set())

        logger.debug("Node added/updated: %s (%s)", node_id, type)
        return node

    def get_node(self, node_id: str) -> Optional[KGNode]:
        """
        Node ID से node return करता है (या None अगर ना मिले)।
        """
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """
        Node और उससे जुड़े सभी edges remove करता है।
        """
        if node_id not in self._nodes:
            return False

        # Collect all related edges
        out_edges = self._outgoing_edges.get(node_id, set()).copy()
        in_edges = self._incoming_edges.get(node_id, set()).copy()

        for edge_id in out_edges.union(in_edges):
            self.remove_edge(edge_id)

        # अब node हटाओ
        del self._nodes[node_id]
        self._outgoing_edges.pop(node_id, None)
        self._incoming_edges.pop(node_id, None)

        logger.debug("Node removed: %s (edges removed: %d)", node_id, len(out_edges) + len(in_edges))
        return True

    def list_nodes(self) -> List[KGNode]:
        """
        सारे nodes की list (id के हिसाब से sorted).
        """
        nodes = list(self._nodes.values())
        nodes.sort(key=lambda n: n.id)
        return nodes

    # --------------------------------------------------------
    # Public API: Edge Management
    # --------------------------------------------------------

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> KGEdge:
        """
        Edge add करता है (या overwrite अनुमति हो तो update करता है)।
        """
        edge_id = _sanitize_id(edge_id)
        source = _sanitize_id(source)
        target = _sanitize_id(target)
        relation = _sanitize_relation(relation)

        if source == target and not self.config.allow_self_loops:
            raise ValueError("Self-loop edges allow_self_loops=False होने पर allowed नहीं।")

        if source not in self._nodes:
            raise ValueError(f"Source node {source!r} मौजूद नहीं है।")
        if target not in self._nodes:
            raise ValueError(f"Target node {target!r} मौजूद नहीं है।")

        if not overwrite and edge_id in self._edges:
            raise ValueError(f"Edge {edge_id!r} already exists (overwrite=False).")

        if edge_id not in self._edges and len(self._edges) >= self.config.max_edges:
            raise ValueError("Max edges limit reached, नया edge add नहीं कर सकते।")

        # अगर parallel edges allow नहीं हैं तो check करें
        if not self.config.allow_parallel_edges:
            for eid in self._outgoing_edges.get(source, set()):
                existing = self._edges[eid]
                if existing.target == target and existing.relation == relation:
                    raise ValueError(
                        f"Parallel edge मौजूद है source={source!r}, "
                        f"target={target!r}, relation={relation!r}."
                    )

        edge = KGEdge(
            id=edge_id,
            source=source,
            target=target,
            relation=relation,
            properties=dict(properties or {}),
        )
        self._edges[edge_id] = edge

        # adjacency update
        self._outgoing_edges.setdefault(source, set()).add(edge_id)
        self._incoming_edges.setdefault(target, set()).add(edge_id)

        logger.debug(
            "Edge added/updated: %s (%s -> %s, relation=%s)",
            edge_id,
            source,
            target,
            relation,
        )
        return edge

    def get_edge(self, edge_id: str) -> Optional[KGEdge]:
        """
        Edge ID से edge return करता है।
        """
        return self._edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> bool:
        """
        Edge remove करता है।
        """
        edge = self._edges.get(edge_id)
        if edge is None:
            return False

        # adjacency से हटाओ
        if edge.source in self._outgoing_edges:
            self._outgoing_edges[edge.source].discard(edge_id)
        if edge.target in self._incoming_edges:
            self._incoming_edges[edge.target].discard(edge_id)

        del self._edges[edge_id]
        logger.debug("Edge removed: %s", edge_id)
        return True

    def list_edges(self) -> List[KGEdge]:
        """
        सारे edges की list (id के हिसाब से sorted)।
        """
        edges = list(self._edges.values())
        edges.sort(key=lambda e: e.id)
        return edges

    # --------------------------------------------------------
    # Public API: Basic Queries
    # --------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relation: Optional[str] = None,
    ) -> List[KGNode]:
        """
        दिए गए node के neighbors return करता है।

        Args:
            node_id: reference node
            direction:
                - "out": source=node_id वाले edges
                - "in": target=node_id वाले edges
                - "both": दोनों तरह
            relation: अगर दिया हो तो उसी relation वाले neighbors
        """
        node_id = _sanitize_id(node_id)
        if node_id not in self._nodes:
            return []

        neighbors_ids: Set[str] = set()

        if direction in ("out", "both"):
            for eid in self._outgoing_edges.get(node_id, set()):
                edge = self._edges[eid]
                if relation and edge.relation != relation:
                    continue
                neighbors_ids.add(edge.target)

        if direction in ("in", "both"):
            for eid in self._incoming_edges.get(node_id, set()):
                edge = self._edges[eid]
                if relation and edge.relation != relation:
                    continue
                neighbors_ids.add(edge.source)

        results: List[KGNode] = []
        for nid in neighbors_ids:
            node = self._nodes.get(nid)
            if node:
                results.append(node)

        results.sort(key=lambda n: n.id)
        return results

    def get_relations(
        self,
        node_a: str,
        node_b: str,
    ) -> List[KGEdge]:
        """
        Node A से node B के बीच direct relations (edges) return करता है।
        """
        node_a = _sanitize_id(node_a)
        node_b = _sanitize_id(node_b)

        if node_a not in self._nodes or node_b not in self._nodes:
            return []

        edges: List[KGEdge] = []
        for eid in self._outgoing_edges.get(node_a, set()):
            edge = self._edges[eid]
            if edge.target == node_b:
                edges.append(edge)

        edges.sort(key=lambda e: e.id)
        return edges

    def node_exists(self, node_id: str) -> bool:
        """
        Node exist करता है या नहीं।
        """
        return node_id in self._nodes

    def edge_exists(self, edge_id: str) -> bool:
        """
        Edge exist करता है या नहीं।
        """
        return edge_id in self._edges

    # --------------------------------------------------------
    # Public API: Simple Path Exploration (1-2 hops)
    # --------------------------------------------------------

    def one_hop_paths(
        self,
        source: str,
        relation: Optional[str] = None,
    ) -> List[Tuple[KGNode, KGEdge]]:
        """
        Source से एक hop दूर neighbors + edges return करता है।
        """
        source = _sanitize_id(source)
        if source not in self._nodes:
            return []

        results: List[Tuple[KGNode, KGEdge]] = []
        for eid in self._outgoing_edges.get(source, set()):
            edge = self._edges[eid]
            if relation and edge.relation != relation:
                continue
            target_node = self._nodes.get(edge.target)
            if target_node:
                results.append((target_node, edge))

        return results

    def two_hop_targets(
        self,
        source: str,
        through_relation: Optional[str] = None,
    ) -> List[KGNode]:
        """
        Source से दो hop दूर वाले unique nodes return करता है।
        """
        source = _sanitize_id(source)
        if source not in self._nodes:
            return []

        first_hop: List[KGNode] = self.neighbors(
            source,
            direction="out",
            relation=through_relation,
        )

        second_hop_ids: Set[str] = set()
        for node in first_hop:
            for neighbor in self.neighbors(node.id, direction="out"):
                second_hop_ids.add(neighbor.id)

        results: List[KGNode] = []
        for nid in second_hop_ids:
            node = self._nodes.get(nid)
            if node:
                results.append(node)

        results.sort(key=lambda n: n.id)
        return results

    # --------------------------------------------------------
    # Public API: Simple Introspection
    # --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """
        Graph की basic stats देता है।
        """
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "max_nodes": self.config.max_nodes,
            "max_edges": self.config.max_edges,
        }

    def debug_snapshot(self, limit: int = 20) -> Dict[str, Any]:
        """
        Debug के लिए छोटा snapshot (कुछ nodes/edges) देता है।
        """
        nodes_list = self.list_nodes()[: max(1, limit)]
        edges_list = self.list_edges()[: max(1, limit)]

        nodes_preview = [
            {
                "id": n.id,
                "label": n.label,
                "type": n.type,
                "properties": dict(n.properties),
            }
            for n in nodes_list
        ]

        edges_preview = [
            {
                "id": e.id,
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "properties": dict(e.properties),
            }
            for e in edges_list
        ]

        return {
            "statistics": self.statistics(),
            "nodes_preview": nodes_preview,
            "edges_preview": edges_preview,
        }


# ============================================================
# Public Factory Helper
# ============================================================

def create_default_knowledge_graph() -> KnowledgeGraph:
    """
    Default KGConfig के साथ simple knowledge graph instance देता है।
    """
    config = KGConfig()
    return KnowledgeGraph(config=config)


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "KGNode",
    "KGEdge",
    "KGConfig",
    "KnowledgeGraph",
    "create_default_knowledge_graph",
]

# End of File: memory/knowledge_graph.py (Phase‑1, ~500 lines with comments)
