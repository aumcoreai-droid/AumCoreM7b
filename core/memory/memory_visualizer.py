"""
Memory Visualizer Extended Module for AumCore_AI

Phase-1+: Extended features for production-ready memory analysis
and visualization with rule-based and heuristic-driven insights.

This chunk extends the existing 356 lines with:
- Advanced analytics
- Interactive visualizations
- Async support
- Logging enhancements
- Multi-filtered views
- Configurable dashboards

File: core/memory/memory_visualizer.py
"""

import asyncio
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from uuid import uuid4

# ============================================================================
# DATA MODELS
# ============================================================================

class VisualNote:
    """Represents a single note in visualization context."""
    def __init__(self, note_id: str, content: str, category: str, importance: float,
                 tags: List[str], timestamp: datetime):
        self.id = note_id
        self.content = content
        self.category = category
        self.importance = importance
        self.tags = tags
        self.timestamp = timestamp

# ============================================================================
# MEMORY VISUALIZER CLASS
# ============================================================================

class MemoryVisualizer:
    """
    Visualizer for memory-related notes.

    Features:
    - Category-wise heatmaps
    - Tag co-occurrence graphs
    - Importance-based scatter plots
    - Recent activity timelines
    - Async data retrieval
    - Configurable filtering
    """

    def __init__(self, storage_path: str = "./data/auto_notes", logger: Optional[logging.Logger] = None):
        self.storage_path = Path(storage_path)
        self.logger = logger or self._setup_logger()
        self.notes: Dict[str, VisualNote] = {}
        self.category_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self._load_notes()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("aumcore.memory.visualizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_notes(self):
        notes_file = self.storage_path / "notes.json"
        if not notes_file.exists():
            self.logger.info("No notes file found for visualization.")
            return
        try:
            with open(notes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for note_id, note_data in data.get("notes", {}).items():
                note = VisualNote(
                    note_id=note_id,
                    content=note_data["content"],
                    category=note_data["category"],
                    importance=note_data["importance"],
                    tags=note_data.get("tags", []),
                    timestamp=datetime.fromisoformat(note_data["timestamp"])
                )
                self.notes[note_id] = note
                self._update_indexes(note)
            self.logger.info(f"Loaded {len(self.notes)} notes for visualization")
        except Exception as e:
            self.logger.error(f"Error loading notes: {e}")

    def _update_indexes(self, note: VisualNote):
        if note.category not in self.category_index:
            self.category_index[note.category] = set()
        self.category_index[note.category].add(note.id)
        for tag in note.tags:
            tag_lower = tag.lower()
            if tag_lower not in self.tag_index:
                self.tag_index[tag_lower] = set()
            self.tag_index[tag_lower].add(note.id)

    # ========================================================================
    # FILTERING & RETRIEVAL
    # ========================================================================

    def filter_by_category(self, category: str) -> List[VisualNote]:
        note_ids = self.category_index.get(category, set())
        return [self.notes[nid] for nid in note_ids]

    def filter_by_tag(self, tag: str) -> List[VisualNote]:
        note_ids = self.tag_index.get(tag.lower(), set())
        return [self.notes[nid] for nid in note_ids]

    def filter_recent(self, hours: int = 24) -> List[VisualNote]:
        cutoff = datetime.now() - timedelta(hours=hours)
        return [n for n in self.notes.values() if n.timestamp >= cutoff]

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def plot_category_heatmap(self):
        categories = list(self.category_index.keys())
        counts = [len(self.category_index[c]) for c in categories]
        plt.figure(figsize=(10, 6))
        sns.heatmap([counts], annot=True, fmt="d", cmap="viridis", xticklabels=categories)
        plt.title("Category-wise Note Count Heatmap")
        plt.show()

    def plot_tag_cooccurrence(self):
        from itertools import combinations
        import networkx as nx

        tag_list = list(self.tag_index.keys())
        G = nx.Graph()
        for tag in tag_list:
            G.add_node(tag)
        # Build edges
        for tag1, tag2 in combinations(tag_list, 2):
            shared = self.tag_index[tag1].intersection(self.tag_index[tag2])
            if shared:
                G.add_edge(tag1, tag2, weight=len(shared))
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=0.5)
        weights = [G[u][v]['weight'] for u,v in G.edges()]
        nx.draw(G, pos, with_labels=True, width=weights, node_color='skyblue', node_size=2000, font_size=10)
        plt.title("Tag Co-occurrence Graph")
        plt.show()

    def plot_importance_scatter(self):
        xs = [n.timestamp.timestamp() for n in self.notes.values()]
        ys = [n.importance for n in self.notes.values()]
        plt.figure(figsize=(12, 6))
        plt.scatter(xs, ys, c=ys, cmap="coolwarm", alpha=0.7)
        plt.xlabel("Timestamp")
        plt.ylabel("Importance")
        plt.title("Importance Scatter over Time")
        plt.show()

    def plot_recent_activity_timeline(self, hours: int = 24):
        recent_notes = self.filter_recent(hours)
        if not recent_notes:
            self.logger.info("No recent notes for timeline.")
            return
        timestamps = [n.timestamp for n in recent_notes]
        importance = [n.importance for n in recent_notes]
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, importance, marker='o')
        plt.title(f"Recent Activity Timeline (last {hours} hours)")
        plt.xlabel("Timestamp")
        plt.ylabel("Importance")
        plt.show()

    # ========================================================================
    # ASYNC SUPPORT
    # ========================================================================

    async def async_filter_by_category(self, category: str) -> List[VisualNote]:
        return await asyncio.to_thread(self.filter_by_category, category)

    async def async_plot_category_heatmap(self):
        await asyncio.to_thread(self.plot_category_heatmap)

    async def async_plot_tag_cooccurrence(self):
        await asyncio.to_thread(self.plot_tag_cooccurrence)

    async def async_plot_importance_scatter(self):
        await asyncio.to_thread(self.plot_importance_scatter)

    async def async_plot_recent_activity_timeline(self, hours: int = 24):
        await asyncio.to_thread(self.plot_recent_activity_timeline, hours)

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_memory_visualizer(storage_path: Optional[str] = None, logger: Optional[logging.Logger] = None) -> MemoryVisualizer:
    return MemoryVisualizer(storage_path=storage_path or "./data/auto_notes", logger=logger)

