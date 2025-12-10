# Reason: Phase-1+ Production-ready Memory Visualizer for AumCoreM7b
# File: core/memory/memory_visualizer.py
# Author: aumcoreai-droid
# Description: Comprehensive visualization module for memory system
#              with category, tag, temporal, importance heatmaps, 
#              graph representation, interactive plots, async support, 
#              and metadata management.

import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
import numpy as np

from core.memory.auto_notes import AutoNotes, Note

# ============================================================================
# MEMORY VISUALIZER CLASS - PROFESSIONAL VERSION
# ============================================================================

class MemoryVisualizer:
    """
    Full-featured Memory Visualizer for AumCoreM7b.

    Features:
    - Category and tag distribution
    - Importance heatmaps
    - Temporal trend analysis
    - Interactive Network Graphs
    - Async visualization support
    - Metadata and logging
    """

    def __init__(
        self,
        notes: Dict[str, Note],
        storage_path: str = "./data/memory_visuals",
        logger: Optional[logging.Logger] = None,
        max_graph_nodes: int = 100
    ) -> None:
        self.notes = notes
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_graph_nodes = max_graph_nodes
        self.logger = logger or self._setup_logger()
        self.logger.info(f"MemoryVisualizer initialized with {len(notes)} notes")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("aumcore.memory.memory_visualizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # =========================================================================
    # CATEGORY DISTRIBUTION
    # =========================================================================
    def plot_category_distribution(self, save: bool = True) -> Optional[plt.Figure]:
        from collections import Counter
        categories = [note.category for note in self.notes.values()]
        counter = Counter(categories)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=list(counter.keys()), y=list(counter.values()), palette="Blues_d", ax=ax)
        ax.set_title("Notes per Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            file_path = self.storage_path / "category_distribution.png"
            plt.savefig(file_path)
            self.logger.info(f"Category distribution saved to {file_path}")
        return fig

    # =========================================================================
    # TAG DISTRIBUTION
    # =========================================================================
    def plot_tag_frequency(self, top_n: int = 20, save: bool = True) -> Optional[plt.Figure]:
        from collections import Counter
        tags = [tag for note in self.notes.values() for tag in note.tags]
        counter = Counter(tags)
        top_tags = dict(counter.most_common(top_n))
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=list(top_tags.keys()), y=list(top_tags.values()), palette="Greens_d", ax=ax)
        ax.set_title(f"Top {top_n} Tags")
        ax.set_xlabel("Tags")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            file_path = self.storage_path / "tag_frequency.png"
            plt.savefig(file_path)
            self.logger.info(f"Tag frequency saved to {file_path}")
        return fig

    # =========================================================================
    # IMPORTANCE HEATMAP
    # =========================================================================
    def plot_importance_heatmap(self, save: bool = True) -> Optional[plt.Figure]:
        sorted_notes = sorted(self.notes.values(), key=lambda n: n.importance, reverse=True)
        importance_values = np.array([n.importance for n in sorted_notes]).reshape(-1,1)
        fig, ax = plt.subplots(figsize=(6, len(importance_values)//5+2))
        sns.heatmap(importance_values, cmap='Reds', cbar_kws={'label':'Importance'}, ax=ax)
        ax.set_ylabel("Notes (sorted)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        if save:
            file_path = self.storage_path / "importance_heatmap.png"
            plt.savefig(file_path)
            self.logger.info(f"Importance heatmap saved to {file_path}")
        return fig

    # =========================================================================
    # TEMPORAL TREND PLOTTING
    # =========================================================================
    def plot_temporal_trends(self, interval: str = 'day', save: bool = True) -> Optional[plt.Figure]:
        """
        Plot the number of notes over time. Interval can be 'day', 'week', or 'month'.
        """
        from collections import Counter
        if not self.notes:
            self.logger.warning("No notes for temporal plotting")
            return None
        dates = []
        for note in self.notes.values():
            if interval == 'day':
                dates.append(note.timestamp.date())
            elif interval == 'week':
                year, week, _ = note.timestamp.isocalendar()
                dates.append(f"{year}-W{week}")
            elif interval == 'month':
                dates.append(f"{note.timestamp.year}-{note.timestamp.month}")
        counter = Counter(dates)
        x = list(counter.keys())
        y = list(counter.values())
        fig, ax = plt.subplots(figsize=(14,6))
        sns.lineplot(x=x, y=y, marker='o', ax=ax)
        ax.set_title(f"Notes over Time ({interval})")
        ax.set_xlabel(interval.capitalize())
        ax.set_ylabel("Number of Notes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            file_path = self.storage_path / f"temporal_trends_{interval}.png"
            plt.savefig(file_path)
            self.logger.info(f"Temporal trend saved to {file_path}")
        return fig

    # =========================================================================
    # GRAPH REPRESENTATION OF NOTES
    # =========================================================================
    def plot_memory_graph(self, max_nodes: Optional[int] = None, save: bool = True) -> Optional[plt.Figure]:
        G = nx.Graph()
        max_nodes = max_nodes or self.max_graph_nodes
        sorted_notes = sorted(self.notes.values(), key=lambda n: n.importance, reverse=True)[:max_nodes]
        for note in sorted_notes:
            G.add_node(note.id, label=note.category, importance=note.importance)
            for tag in note.tags:
                tag_node = f"tag:{tag}"
                G.add_node(tag_node)
                G.add_edge(note.id, tag_node)
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(12,12))
        nx.draw(G, pos, with_labels=False, node_size=200, node_color='skyblue', edge_color='gray', ax=ax)
        labels = {n: G.nodes[n]['label'] if 'label' in G.nodes[n] else n for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        ax.set_title("Memory Graph Visualization")
        plt.tight_layout()
        if save:
            file_path = self.storage_path / "memory_graph.png"
            plt.savefig(file_path)
            self.logger.info(f"Memory graph saved to {file_path}")
        return fig

    # =========================================================================
    # ASYNC SUPPORT
    # =========================================================================
    async def plot_category_distribution_async(self):
        return await asyncio.to_thread(self.plot_category_distribution)

    async def plot_tag_frequency_async(self):
        return await asyncio.to_thread(self.plot_tag_frequency)

    async def plot_importance_heatmap_async(self):
        return await asyncio.to_thread(self.plot_importance_heatmap)

    async def plot_temporal_trends_async(self, interval: str = 'day'):
        return await asyncio.to_thread(self.plot_temporal_trends, interval)

    async def plot_memory_graph_async(self, max_nodes: Optional[int] = None):
        return await asyncio.to_thread(self.plot_memory_graph, max_nodes)

    # =========================================================================
    # METADATA MANAGEMENT
    # =========================================================================
    def save_metadata(self) -> None:
        metadata_file = self.storage_path / "metadata.json"
        metadata = {
            "saved_at": datetime.now().isoformat(),
            "visualizations": [f.name for f in self.storage_path.glob("*.png")],
            "note_count": len(self.notes),
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Metadata saved to {metadata_file}")

# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def create_memory_visualizer(notes: Dict[str, Note], storage_path: str = "./data/memory_visuals") -> MemoryVisualizer:
    return MemoryVisualizer(notes=notes, storage_path=storage_path)

# ============================================================================
# MODULE EXPORTS
# ============================================================================
__all__ = ["MemoryVisualizer", "create_memory_visualizer"]
