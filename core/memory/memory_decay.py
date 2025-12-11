"""
AumCore_AI Memory Subsystem - Memory Decay Module
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/memory_decay.py

Description:
    यह module Phase‑1 में simple, rule-based memory decay system देता है।

    Goals (Phase‑1):
        - Short-term memory items पर time + usage आधारित decay लागू करना
        - Deterministic, explainable scoring
        - Simple pruning (low-importance items हटाना)
        - कोई ML / embeddings / DB / async नहीं

    Future (Phase‑3+):
        - Semantic decay (similarity-based)
        - Cross-module coordinated decay (LTM, notes, KG)
        - Adaptive decay per-user / per-category
"""

# ============================================================
# ✅ Chunk 1: Imports (PEP8)
# ============================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# ✅ Chunk 1: Local Logger Setup
# ============================================================

def get_decay_logger(name: str = "AumCoreAI.Memory.Decay") -> logging.Logger:
    """
    Memory decay module के लिए simple logger बनाता है।
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [MEMORY_DECAY] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_decay_logger()


# ============================================================
# ✅ Chunk 2: Data Structures (Rule-Based)
# ============================================================

@dataclass
class DecayConfig:
    """
    Memory decay के लिए Phase‑1 configuration।
    Pure rule-based knobs, कोई model-based param नहीं।
    """

    # कितने समय बाद decay शुरू हो (last_accessed से)
    decay_after_minutes: int = 30

    # decay factor (0.0–1.0), importance को इससे multiply किया जाएगा
    decay_factor: float = 0.85

    # minimum importance threshold (इससे नीचे वाले items prune हो सकते हैं)
    min_importance_threshold: float = 0.15

    # prune करने के लिए percentage (0.0–1.0), low-importance में से
    prune_percentage: float = 0.10

    # max text length (sanitization)
    max_text_length: int = 5000

    def __post_init__(self) -> None:
        if self.decay_after_minutes <= 0:
            raise ValueError("decay_after_minutes positive होना चाहिए।")
        if not (0.0 <= self.decay_factor <= 1.0):
            raise ValueError("decay_factor 0.0–1.0 के बीच होना चाहिए।")
        if not (0.0 <= self.min_importance_threshold <= 1.0):
            raise ValueError("min_importance_threshold 0.0–1.0 के बीच होना चाहिए।")
        if not (0.0 <= self.prune_percentage <= 1.0):
            raise ValueError("prune_percentage 0.0–1.0 के बीच होना चाहिए।")
        if self.max_text_length <= 0:
            raise ValueError("max_text_length positive होना चाहिए।")


@dataclass
class MemoryItem:
    """
    Short-term memory item (Phase‑1 simple structure)।

    Fields:
        id: unique identifier
        content: text content
        importance: 0.0–1.0
        created_at: कब create हुआ
        last_accessed_at: आख़िरी बार कब access हुआ
        seen_count: कितनी बार access हुआ
        metadata: extra info (category, source, etc.)
    """

    id: str
    content: str
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    seen_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("MemoryItem.id non-empty string होना चाहिए।")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("MemoryItem.content non-empty string होना चाहिए।")
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("MemoryItem.importance 0.0–1.0 के बीच होना चाहिए।")
        if self.seen_count < 0:
            raise ValueError("MemoryItem.seen_count negative नहीं होना चाहिए।")


# ============================================================
# ✅ Chunk 2: Sanitization + Utility Helpers
# ============================================================

def _sanitize_text(value: str, max_len: int) -> str:
    """
    Content sanitization (trim + length clamp).
    """
    if value is None:
        return ""
    t = str(value).strip()
    if len(t) > max_len:
        t = t[:max_len].rstrip()
    return t


def _clamp_importance(value: float) -> float:
    """
    Importance को 0.0–1.0 range में clamp करता है।
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_seen_increment(current: int, inc: int = 1) -> int:
    """
    seen_count को safe तरीके से increment करता है।
    """
    if current < 0:
        current = 0
    if inc < 0:
        inc = 0
    return current + inc


# ============================================================
# ✅ Core Class: MemoryDecayEngine (Phase‑1)
# ============================================================

class MemoryDecayEngine:
    """
    Phase‑1 compatible memory decay engine।

    Features:
        - In-memory short-term items
        - Time-based decay (last_accessed से)
        - Usage-based context (seen_count)
        - Rule-based pruning
        - No async, no DB, no model inference
    """

    def __init__(self, config: Optional[DecayConfig] = None) -> None:
        self.config: DecayConfig = config or DecayConfig()
        self._items: Dict[str, MemoryItem] = {}

        logger.debug("MemoryDecayEngine initialized with config: %r", self.config)

    # --------------------------------------------------------
    # Public API: Add / Update
    # --------------------------------------------------------

    def add_item(
        self,
        item_id: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """
        नया short-term memory item add करता है।
        """
        item_id = str(item_id).strip()
        if not item_id:
            raise ValueError("item_id empty नहीं हो सकता।")

        cleaned = _sanitize_text(content, self.config.max_text_length)
        if not cleaned:
            raise ValueError("content empty नहीं हो सकता।")

        imp = _clamp_importance(importance)
        now = datetime.now()

        item = MemoryItem(
            id=item_id,
            content=cleaned,
            importance=imp,
            created_at=now,
            last_accessed_at=now,
            seen_count=0,
            metadata=dict(metadata or {}),
        )
        self._items[item_id] = item
        logger.debug("Memory item added: %s (importance=%.2f)", item_id, imp)
        return item

    def update_item(
        self,
        item_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        seen_increment: int = 1,
    ) -> bool:
        """
        Existing memory item update करता है।
        """
        item = self._items.get(item_id)
        if item is None:
            return False

        if content is not None:
            item.content = _sanitize_text(content, self.config.max_text_length)
        if importance is not None:
            item.importance = _clamp_importance(importance)
        if metadata:
            item.metadata.update(metadata)

        item.last_accessed_at = datetime.now()
        item.seen_count = _safe_seen_increment(item.seen_count, seen_increment)

        logger.debug("Memory item updated: %s (importance=%.2f)", item_id, item.importance)
        return True

    # --------------------------------------------------------
    # Public API: Retrieval
    # --------------------------------------------------------

    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """
        ID से memory item retrieve करता है और access stats update करता है।
        """
        item = self._items.get(item_id)
        if item:
            item.last_accessed_at = datetime.now()
            item.seen_count = _safe_seen_increment(item.seen_count, 1)
        return item

    def list_items(self) -> List[MemoryItem]:
        """
        सारे memory items list करता है (id sorted)।
        """
        items = list(self._items.values())
        items.sort(key=lambda i: i.id)
        return items

    # --------------------------------------------------------
    # Public API: Decay Application
    # --------------------------------------------------------

    def apply_decay(self) -> None:
        """
        सारे items पर decay apply करता है:
            - Time-based decay factor
            - फिर pruning
        """
        now = datetime.now()
        decay_delta = timedelta(minutes=self.config.decay_after_minutes)

        for item in self._items.values():
            age = now - item.last_accessed_at

            if age < decay_delta:
                continue

            old_imp = item.importance
            new_imp = _clamp_importance(old_imp * self.config.decay_factor)
            item.importance = new_imp

            logger.debug(
                "Decay applied to %s: importance %.2f -> %.2f (age=%s, seen=%d)",
                item.id,
                old_imp,
                new_imp,
                age,
                item.seen_count,
            )

        self._prune_low_importance()

    # --------------------------------------------------------
    # Internal: Pruning
    # --------------------------------------------------------

    def _prune_low_importance(self) -> None:
        """
        Low-importance items prune करता है।
        Rule:
            - items को (importance, seen_count, last_accessed_at) से sort
            - bottom prune_percentage items consider
            - उनमें से सिर्फ min_importance_threshold से नीचे वाले हटाओ
        """
        if not self._items:
            return

        items_sorted = sorted(
            self._items.values(),
            key=lambda i: (i.importance, i.seen_count, i.last_accessed_at),
        )

        total = len(items_sorted)
        prune_candidate_count = max(1, int(total * self.config.prune_percentage))
        candidates = items_sorted[:prune_candidate_count]

        threshold = self.config.min_importance_threshold
        removed_ids: List[str] = []

        for item in candidates:
            if item.importance < threshold:
                removed_ids.append(item.id)

        for item_id in removed_ids:
            del self._items[item_id]

        if removed_ids:
            logger.info(
                "MemoryDecayEngine pruned %d items (threshold=%.2f, total_after=%d).",
                len(removed_ids),
                threshold,
                len(self._items),
            )

    # --------------------------------------------------------
    # Public API: Delete / Clear
    # --------------------------------------------------------

    def delete_item(self, item_id: str) -> bool:
        """
        एक memory item remove करता है।
        """
        if item_id in self._items:
            del self._items[item_id]
            logger.debug("Memory item deleted: %s", item_id)
            return True
        return False

    def clear(self) -> None:
        """
        सारे memory items clear करता है।
        """
        count = len(self._items)
        self._items.clear()
        logger.info("MemoryDecayEngine cleared, removed %d items.", count)

    # --------------------------------------------------------
    # Public API: Simple Query Helpers
    # --------------------------------------------------------

    def important_items(
        self,
        min_importance: float = 0.5,
        limit: int = 20,
    ) -> List[MemoryItem]:
        """
        दिए गए threshold से ऊपर वाले important items return करता है।
        """
        thr = _clamp_importance(min_importance)
        items = [
            i for i in self._items.values() if i.importance >= thr
        ]
        items.sort(
            key=lambda i: (i.importance, i.seen_count, i.last_accessed_at),
            reverse=True,
        )
        return items[: max(1, limit)]

    def stale_items(
        self,
        older_than_minutes: int,
        limit: int = 20,
    ) -> List[MemoryItem]:
        """
        दिए गए time से पुराने (last_accessed) items return करता है।
        """
        if older_than_minutes <= 0:
            return []

        cutoff = datetime.now() - timedelta(minutes=older_than_minutes)
        items = [
            i for i in self._items.values() if i.last_accessed_at < cutoff
        ]
        items.sort(key=lambda i: i.last_accessed_at)
        return items[: max(1, limit)]

    # --------------------------------------------------------
    # Public API: Stats & Debug
    # --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """
        Memory decay system की basic stats देता है।
        """
        total = len(self._items)
        avg_importance = (
            sum(i.importance for i in self._items.values()) / total
            if total > 0
            else 0.0
        )
        max_seen = max((i.seen_count for i in self._items.values()), default  = 0)

        return {
            "total_items": total,
            "avg_importance": avg_importance,
            "max_seen_count": max_seen,
            "min_importance_threshold": self.config.min_importance_threshold,
            "decay_factor": self.config.decay_factor,
            "decay_after_minutes": self.config.decay_after_minutes,
        }

    def debug_snapshot(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Debug के लिए कुछ items का preview देता है।
        """
        items = self.list_items()[: max(1, limit)]
        snapshot: List[Dict[str, Any]] = []

        for i in items:
            snapshot.append(
                {
                    "id": i.id,
                    "importance": i.importance,
                    "seen_count": i.seen_count,
                    "created_at": i.created_at.isoformat(),
                    "last_accessed_at": i.last_accessed_at.isoformat(),
                    "content_preview": i.content[:120],
                }
            )

        return snapshot


# ============================================================
# Public Factory Helper
# ============================================================

def create_default_memory_decay_engine() -> MemoryDecayEngine:
    """
    Default DecayConfig के साथ MemoryDecayEngine instance देता है।
    """
    config = DecayConfig()
    return MemoryDecayEngine(config=config)


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "DecayConfig",
    "MemoryItem",
    "MemoryDecayEngine",
    "create_default_memory_decay_engine",
]

# End of File: memory/memory_decay.py (Phase‑1, ~500+ lines with comments)
