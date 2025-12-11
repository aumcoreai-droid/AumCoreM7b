"""
AumCore_AI Memory Subsystem - Long Term Memory Module
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/long_term_memory.py

Description:
    यह module Phase‑1 में simple, rule-based, in-memory
    long-term memory storage देता है।

    Core Idea (Phase‑1):
        - Important, reusable information को structured entries के रूप में store करना।
        - कोई DB, vector store, या model-based retention नहीं।
        - Deterministic rules से decide करना कि क्या long-term memory बनना चाहिए।

    Future (Phase‑3+):
        - Persistent DB / KV store integration
        - Priority-based decay and promotion
        - Semantic grouping and linking
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

def get_ltm_logger(name: str = "AumCoreAI.Memory.LongTerm") -> logging.Logger:
    """
    Long-term memory module के लिए local logger।
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [LONG_TERM_MEMORY] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_ltm_logger()


# ============================================================
# ✅ Chunk 2: Data Structures (Rule-Based Models)
# ============================================================

@dataclass
class LongTermMemoryConfig:
    """
    Long-term memory के लिए Phase‑1 configuration।
    """

    max_entries: int = 2_000
    min_importance_for_promotion: float = 0.6
    min_seen_count_for_promotion: int = 2
    default_ttl_days: int = 365  # सिर्फ informational, hard enforcement नहीं
    max_text_length: int = 5_000

    def __post_init__(self) -> None:
        if self.max_entries <= 0:
            raise ValueError("max_entries positive होना चाहिए।")
        if not (0.0 <= self.min_importance_for_promotion <= 1.0):
            raise ValueError("min_importance_for_promotion 0.0–1.0 के बीच होना चाहिए।")
        if self.min_seen_count_for_promotion <= 0:
            raise ValueError("min_seen_count_for_promotion positive होना चाहिए।")
        if self.default_ttl_days <= 0:
            raise ValueError("default_ttl_days positive होना चाहिए।")
        if self.max_text_length <= 0:
            raise ValueError("max_text_length positive होना चाहिए।")


@dataclass
class LongTermMemoryEntry:
    """
    Long-term memory में store होने वाली एक entry।

    Fields:
        id: unique identifier (string, external or auto-generated)
        content: human-readable text content
        category: optional category (e.g., "user_preference", "fact", "rule")
        importance: 0.0–1.0 (Phase‑1 rule-based मान)
        created_at: कब create हुआ
        last_accessed_at: last retrieval/update समय
        seen_count: कितनी बार access/observe हुआ
        metadata: अतिरिक्त key-value pairs (Phase‑1 simple)
    """

    id: str
    content: str
    category: str = "general"
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    seen_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("LongTermMemoryEntry.id non-empty string होना चाहिए।")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("LongTermMemoryEntry.content non-empty string होना चाहिए।")
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("importance 0.0–1.0 के बीच होना चाहिए।")
        if self.seen_count < 0:
            raise ValueError("seen_count negative नहीं होना चाहिए।")


# ============================================================
# ✅ Chunk 2: Sanitization + Validation Utilities
# ============================================================

def _sanitize_text(text: str, max_len: int) -> str:
    """
    Content के लिए simple sanitization।
    """
    if text is None:
        return ""
    t = str(text).strip()
    if len(t) > max_len:
        t = t[:max_len].rstrip()
    return t


def _sanitize_category(category: Optional[str]) -> str:
    """
    Category field normalization।
    """
    if category is None:
        return "general"
    c = str(category).strip()
    return c if c else "general"


def _clamp_importance(value: float) -> float:
    """
    Importance को 0.0–1.0 में clamp करता है।
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ============================================================
# ✅ Core Class: LongTermMemoryStore (Phase‑1)
# ============================================================

class LongTermMemoryStore:
    """
    Phase‑1 compatible long-term memory store।

    Features:
        - In-memory dict-based storage
        - Deterministic promotion rules (importance + seen_count)
        - Simple retrieval filters
        - Basic maintenance (pruning by importance)
    """

    def __init__(self, config: Optional[LongTermMemoryConfig] = None) -> None:
        self.config: LongTermMemoryConfig = config or LongTermMemoryConfig()
        self._entries: Dict[str, LongTermMemoryEntry] = {}

        logger.debug("LongTermMemoryStore initialized with config: %r", self.config)

    # --------------------------------------------------------
    # Public API: Promotion Decision
    # --------------------------------------------------------

    def should_promote(
        self,
        content: str,
        importance: float,
        seen_count: int,
    ) -> bool:
        """
        Rule-based decision कि दिया हुआ content long-term memory के लायक है या नहीं।

        Rules (Phase‑1):
            - importance >= min_importance_for_promotion
            - seen_count >= min_seen_count_for_promotion
        """
        imp = _clamp_importance(importance)
        return (
            imp >= self.config.min_importance_for_promotion
            and seen_count >= self.config.min_seen_count_for_promotion
        )

    # --------------------------------------------------------
    # Public API: Add / Update Entries
    # --------------------------------------------------------

    def upsert_entry(
        self,
        entry_id: str,
        content: str,
        category: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        seen_increment: int = 1,
    ) -> LongTermMemoryEntry:
        """
        Long-term memory entry create या update करता है।

        Behavior:
            - अगर ID exist नहीं करता => नया entry create
            - अगर ID already exist => content/importance/category/metadata update
            - seen_count += seen_increment
        """
        entry_id = str(entry_id).strip()
        if not entry_id:
            raise ValueError("entry_id empty नहीं हो सकता।")

        cleaned_content = _sanitize_text(content, self.config.max_text_length)
        if not cleaned_content:
            raise ValueError("content empty नहीं हो सकता।")

        cat = _sanitize_category(category)
        imp = _clamp_importance(importance)
        now = datetime.now()

        existing = self._entries.get(entry_id)
        if existing is None:
            if len(self._entries) >= self.config.max_entries:
                self._prune_low_importance()

            entry = LongTermMemoryEntry(
                id=entry_id,
                content=cleaned_content,
                category=cat,
                importance=imp,
                created_at=now,
                last_accessed_at=now,
                seen_count=max(0, seen_increment),
                metadata=dict(metadata or {}),
            )
            self._entries[entry_id] = entry
            logger.debug("LTM entry created: %s", entry_id)
        else:
            existing.content = cleaned_content
            existing.category = cat
            existing.importance = imp
            existing.last_accessed_at = now
            existing.seen_count = max(0, existing.seen_count + max(0, seen_increment))
            if metadata:
                existing.metadata.update(metadata)
            entry = existing
            logger.debug("LTM entry updated: %s", entry_id)

        return entry

    # --------------------------------------------------------
    # Public API: Retrieval
    # --------------------------------------------------------

    def get_entry(self, entry_id: str) -> Optional[LongTermMemoryEntry]:
        """
        ID से entry retrieve करता है और last_accessed_at + seen_count update करता है।
        """
        entry = self._entries.get(entry_id)
        if entry is None:
            return None

        entry.last_accessed_at = datetime.now()
        entry.seen_count = max(0, entry.seen_count + 1)
        return entry

    def list_entries(self) -> List[LongTermMemoryEntry]:
        """
        सारे entries को list के रूप में return करता है (id sorted)।
        """
        entries = list(self._entries.values())
        entries.sort(key=lambda e: e.id)
        return entries

    def find_by_category(
        self,
        category: str,
        min_importance: float = 0.0,
    ) -> List[LongTermMemoryEntry]:
        """
        Category और min importance के आधार पर entries filter करता है।
        """
        cat = _sanitize_category(category)
        threshold = _clamp_importance(min_importance)

        results: List[LongTermMemoryEntry] = []
        for entry in self._entries.values():
            if entry.category != cat:
                continue
            if entry.importance < threshold:
                continue
            results.append(entry)

        results.sort(
            key=lambda e: (e.importance, e.seen_count, e.last_accessed_at),
            reverse=True,
        )
        return results

    def search_contains(
        self,
        text: str,
        min_importance: float = 0.0,
        limit: int = 20,
    ) -> List[LongTermMemoryEntry]:
        """
        Simple substring-based search (Phase‑1, no semantics)।
        """
        query = _sanitize_text(text, self.config.max_text_length)
        if not query:
            return []

        q_lower = query.lower()
        threshold = _clamp_importance(min_importance)

        matches: List[LongTermMemoryEntry] = []
        for entry in self._entries.values():
            if entry.importance < threshold:
                continue
            if q_lower in entry.content.lower():
                matches.append(entry)

        matches.sort(
            key=lambda e: (e.importance, e.seen_count, e.last_accessed_at),
            reverse=True,
        )
        return matches[: max(1, limit)]

    # --------------------------------------------------------
    # Public API: Delete / Maintenance
    # --------------------------------------------------------

    def delete_entry(self, entry_id: str) -> bool:
        """
        Entry remove करता है।
        """
        if entry_id in self._entries:
            del self._entries[entry_id]
            logger.debug("LTM entry deleted: %s", entry_id)
            return True
        return False

    def clear(self) -> None:
        """
        सारे entries clear करता है।
        """
        count = len(self._entries)
        self._entries.clear()
        logger.info("LongTermMemoryStore cleared, removed %d entries.", count)

    def _prune_low_importance(self) -> None:
        """
        Max entries limit cross होने पर low-importance entries prune करता है।
        Phase‑1 में basic rule:
            - importance asc
            - फिर seen_count asc
        """
        if not self._entries:
            return

        entries_sorted = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.seen_count, e.last_accessed_at),
        )

        remove_count = max(1, len(entries_sorted) // 10)
        to_remove = entries_sorted[:remove_count]

        for entry in to_remove:
            del self._entries[entry.id]

        logger.info(
            "LTM pruned %d low-importance entries (size=%d).",
            remove_count,
            len(self._entries),
        )

    # --------------------------------------------------------
    # Public API: Simple Stats & Debug
    # --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """
        Store के बारे में basic statistics देता है।
        """
        total = len(self._entries)
        by_category: Dict[str, int] = {}

        for entry in self._entries.values():
            by_category[entry.category] = by_category.get(entry.category, 0) + 1

        avg_importance = (
            sum(e.importance for e in self._entries.values()) / total
            if total > 0
            else 0.0
        )

        return {
            "total_entries": total,
            "by_category": by_category,
            "avg_importance": avg_importance,
            "max_entries": self.config.max_entries,
        }

    def debug_snapshot(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Debug के लिए कुछ entries का preview देता है।
        """
        entries = self.list_entries()[: max(1, limit)]
        snapshot: List[Dict[str, Any]] = []

        for e in entries:
            snapshot.append(
                {
                    "id": e.id,
                    "category": e.category,
                    "importance": e.importance,
                    "seen_count": e.seen_count,
                    "created_at": e.created_at.isoformat(),
                    "last_accessed_at": e.last_accessed_at.isoformat(),
                    "content_preview": e.content[:120],
                }
            )

        return snapshot

    # --------------------------------------------------------
    # Helper: TTL / Age check (Phase‑1 informational)
    # --------------------------------------------------------

    def is_entry_old(self, entry: LongTermMemoryEntry) -> bool:
        """
        Check करता है कि entry default TTL से पुरानी है या नहीं।
        (Phase‑1 में सिर्फ informational, auto-delete नहीं करता)
        """
        ttl_delta = timedelta(days=self.config.default_ttl_days)
        return datetime.now() - entry.created_at > ttl_delta

    def list_old_entries(self) -> List[LongTermMemoryEntry]:
        """
        Default TTL से पुरानी entries की list देता है (informational).
        """
        return [
            e for e in self._entries.values() if self.is_entry_old(e)
        ]


# ============================================================
# Public Factory Helper
# ============================================================

def create_default_long_term_memory_store() -> LongTermMemoryStore:
    """
    Default config के साथ LongTermMemoryStore instance देता है।
    """
    config = LongTermMemoryConfig()
    return LongTermMemoryStore(config=config)


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "LongTermMemoryConfig",
    "LongTermMemoryEntry",
    "LongTermMemoryStore",
    "create_default_long_term_memory_store",
]

# End of File: memory/long_term_memory.py (Phase‑1, ~500 lines with comments)
