# =========================================
# Memory Visualizer Module - AumCore_AI
# File: core/memory/memory_visualizer.py
# Phase-1+: Production-ready, rule-based
# Phase-2+: Ready for Mistral-7B embeddings
# =========================================

import asyncio
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
import re
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class VisualizerConfig:
    storage_path: str = "./data/memory_visualizer"
    auto_save: bool = True
    enable_graph: bool = True
    enable_dashboard: bool = True
    max_nodes: int = 5000
    relevance_threshold: float = 0.3
    enable_async: bool = True
    log_level: int = logging.INFO

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MemoryNode:
    """Represents a node in memory visualizer graph."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    category: str = "general"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "memory"
    context: Dict[str, Any] = field(default_factory=dict)
    edges: Set[str] = field(default_factory=set)

    def add_tag(self, tag: str) -> None:
        tag_lower = tag.lower().strip()
        if tag_lower and tag_lower not in self.tags:
            self.tags.append(tag_lower)

    def add_edge(self, node_id: str) -> None:
        if node_id and node_id != self.id:
            self.edges.add(node_id)

# ============================================================================
# MEMORY VISUALIZER CORE
# ============================================================================

class MemoryVisualizer:
    """
    Production-ready memory visualizer module.

    Features:
    - Node management (add, update, delete)
    - Edge relationships
    - Importance scoring
    - Category & tag indexing
    - Graph visualization using networkx + matplotlib
    - Dashboard statistics
    - Async APIs for large-scale processing
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Node storage
        self.nodes: Dict[str, MemoryNode] = {}

        # Indexes
        self.category_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}

        # Logger
        self.logger = self._setup_logger()
        self._load_from_disk()
        self.logger.info(f"MemoryVisualizer initialized: {len(self.nodes)} nodes loaded")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("aumcore.memory.visualizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(self.config.log_level)
        return logger

    # ============================================================================
    # NODE MANAGEMENT
    # ============================================================================

    def add_node(self, content: str, category: str = "general", importance: float = 0.5,
                 tags: Optional[List[str]] = None, source: str = "manual") -> str:
        node = MemoryNode(
            content=content,
            category=category,
            importance=max(0.0, min(1.0, importance)),
            tags=tags or [],
            source=source
        )
        self.nodes[node.id] = node
        self._update_indexes(node)
        if self.config.auto_save:
            self._save_to_disk()
        self.logger.debug(f"Node added: {node.id} ({category})")
        return node.id

    def update_node(self, node_id: str, content: Optional[str] = None, category: Optional[str] = None,
                    importance: Optional[float] = None, tags: Optional[List[str]] = None) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        self._remove_from_indexes(node)
        if content is not None:
            node.content = content
        if category is not None:
            node.category = category
        if importance is not None:
            node.importance = max(0.0, min(1.0, importance))
        if tags is not None:
            node.tags = [t.lower() for t in tags]
        self._update_indexes(node)
        if self.config.auto_save:
            self._save_to_disk()
        self.logger.debug(f"Node updated: {node_id}")
        return True

    def delete_node(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        self._remove_from_indexes(node)
        # Remove edges pointing to this node
        for n in self.nodes.values():
            n.edges.discard(node_id)
        del self.nodes[node_id]
        if self.config.auto_save:
            self._save_to_disk()
        self.logger.debug(f"Node deleted: {node_id}")
        return True

    # ============================================================================
    # INDEX MANAGEMENT
    # ============================================================================

    def _update_indexes(self, node: MemoryNode) -> None:
        if node.category not in self.category_index:
            self.category_index[node.category] = set()
        self.category_index[node.category].add(node.id)
        for tag in node.tags:
            t = tag.lower()
            if t not in self.tag_index:
                self.tag_index[t] = set()
            self.tag_index[t].add(node.id)

    def _remove_from_indexes(self, node: MemoryNode) -> None:
        if node.category in self.category_index:
            self.category_index[node.category].discard(node.id)
        for tag in node.tags:
            t = tag.lower()
            if t in self.tag_index:
                self.tag_index[t].discard(node.id)

    # ============================================================================
    # NODE RETRIEVAL
    # ============================================================================

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def search_nodes(self, query: str, category: Optional[str] = None,
                     min_importance: float = 0.0, limit: int = 10) -> List[MemoryNode]:
        query_lower = query.lower()
        matches = []
        for node in self.nodes.values():
            if category and node.category != category:
                continue
            if node.importance < min_importance:
                continue
            if query_lower in node.content.lower():
                matches.append(node)
        matches.sort(key=lambda n: n.importance, reverse=True)
        return matches[:limit]

    def get_nodes_by_category(self, category: str) -> List[MemoryNode]:
        ids = self.category_index.get(category, set())
        nodes = [self.nodes[nid] for nid in ids if nid in self.nodes]
        nodes.sort(key=lambda n: n.timestamp, reverse=True)
        return nodes

    def get_nodes_by_tag(self, tag: str) -> List[MemoryNode]:
        ids = self.tag_index.get(tag.lower(), set())
        nodes = [self.nodes[nid] for nid in ids if nid in self.nodes]
        nodes.sort(key=lambda n: n.importance, reverse=True)
        return nodes

    def get_most_important_nodes(self, limit: int = 10) -> List[MemoryNode]:
        return sorted(self.nodes.values(), key=lambda n: n.importance, reverse=True)[:limit]

    # ============================================================================
    # EDGE MANAGEMENT
    # ============================================================================

    def add_edge(self, from_id: str, to_id: str) -> bool:
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].add_edge(to_id)
            self.nodes[to_id].add_edge(from_id)
            if self.config.auto_save:
                self._save_to_disk()
            return True
        return False

    # ============================================================================
    # GRAPH VISUALIZATION
    # ============================================================================

    def visualize_graph(self, highlight_nodes: Optional[List[str]] = None) -> None:
        if not self.config.enable_graph:
            self.logger.warning("Graph visualization is disabled in config.")
            return
        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.id, label=node.content[:15])
            for e in node.edges:
                if e in self.nodes:
                    G.add_edge(node.id, e)
        pos = nx.spring_layout(G, k=0.5)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color='skyblue', edge_color='gray', node_size=1000, font_size=8)
        if highlight_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='orange')
        plt.title("Memory Visualizer Graph")
        plt.show()

    # ============================================================================
    # STATISTICS
    # ============================================================================

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_nodes": len(self.nodes),
            "by_category": {k: len(v) for k, v in self.category_index.items()},
            "by_tag": {k: len(v) for k, v in self.tag_index.items()},
            "avg_importance": sum(n.importance for n in self.nodes.values()) / max(1, len(self.nodes)),
        }
        return stats

    # ============================================================================
    # PERSISTENCE
    # ============================================================================

    def _save_to_disk(self) -> None:
        path = self.storage_path / "nodes.json"
        data = {
            "nodes": {nid: self._serialize_node(n) for nid, n in self.nodes.items()},
            "saved_at": datetime.now().isoformat()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self.logger.debug(f"Saved {len(self.nodes)} nodes to disk")

    def _load_from_disk(self) -> None:
        path = self.storage_path / "nodes.json"
        if not path.exists():
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for nid, ndata in data.get("nodes", {}).items():
            node = MemoryNode(
                id=ndata["id"],
                content=ndata["content"],
                category=ndata["category"],
                importance=ndata["importance"],
                tags=ndata.get("tags", []),
                timestamp=datetime.fromisoformat(ndata["timestamp"]),
                source=ndata.get("source", "memory"),
                context=ndata.get("context", {}),
                edges=set(ndata.get("edges", []))
            )
            self.nodes[nid] = node
            self._update_indexes(node)
        self.logger.debug(f"Loaded {len(self.nodes)} nodes from disk")

    def _serialize_node(self, node: MemoryNode) -> Dict[str, Any]:
        return {
            "id": node.id,
            "content": node.content,
            "category": node.category,
            "importance": node.importance,
            "tags": node.tags,
            "timestamp": node.timestamp.isoformat(),
            "source": node.source,
            "context": node.context,
            "edges": list(node.edges)
        }

    # ============================================================================
    # ASYNC SUPPORT
    # ============================================================================

    async def add_node_async(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.add_node, *args, **kwargs)

    async def search_nodes_async(self, *args, **kwargs) -> List[MemoryNode]:
        return await asyncio.to_thread(self.search_nodes, *args, **kwargs)

    async def visualize_graph_async(self, *args, **kwargs) -> None:
        await asyncio.to_thread(self.visualize_graph, *args, **kwargs)

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_memory_visualizer(config: Optional[VisualizerConfig] = None) -> MemoryVisualizer:
    return MemoryVisualizer(config=config)

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "MemoryVisualizer",
    "MemoryNode",
    "VisualizerConfig",
    "create_memory_visualizer"
]

# ============================================================================
# TODO: Phase-2+ Integrations
# - Mistral-7B embeddings
# - Semantic relevance scoring
# - Auto clustering & graph layout optimization
# - Interactive dashboards
# - Cross-memory linking
# - Collaborative editing
# ============================================================================

