"""
AumCore_AI Memory Subsystem - Dynamic Retrieval Module
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/dynamic_retrieval.py

Description:
    यह module Phase‑1 में पूरी तरह rule-based retrieval provide करता है।
    कोई semantic search, embeddings, या model-based reranking नहीं है।

    Core Idea (Phase‑1):
        - इनपुट text queries से simple keyword, substring और scoring के हिसाब से
          memory notes / items को retrieve करना।
        - Deterministic, explainable scoring logic (pure Python, कोई external dependency नहीं)।

    Future (Phase‑3+):
        - Model-based semantic similarity
        - Embedding indexes
        - Hybrid lexical + vector retrieval
"""

# ============================================================
# ✅ Chunk 1: Imports (PEP8)
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# ✅ Chunk 1: Local Logger Setup
# ============================================================

def get_dynamic_retrieval_logger(
    name: str = "AumCoreAI.Memory.DynamicRetrieval",
) -> logging.Logger:
    """
    Dynamic retrieval module के लिए simple logger।
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [DYNAMIC_RETRIEVAL] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_dynamic_retrieval_logger()


# ============================================================
# ✅ Chunk 2: Basic Data Structures (Rule-Based)
# ============================================================

@dataclass
class RetrievalItem:
    """
    Retrieval के लिए Phase‑1 compatible item।

    Fields:
        id: unique identifier (string)
        text: searchable text content
        metadata: optional metadata dict (category, source, tags etc.)
        importance: 0.0–1.0 range में optional importance (अगर हो)
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("RetrievalItem.id non-empty string होना चाहिए।")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("RetrievalItem.text non-empty string होना चाहिए।")
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("RetrievalItem.importance 0.0–1.0 के बीच होना चाहिए।")


@dataclass
class RetrievalResult:
    """
    Retrieval result का representation।
    """

    item_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    match_explanation: str = ""


@dataclass
class RetrievalConfig:
    """
    Dynamic retrieval के लिए Phase‑1 configuration।
    Pure rule-based knobs।
    """

    min_score: float = 0.1
    max_results: int = 20
    case_sensitive: bool = False
    boost_importance: bool = True
    length_penalty: bool = True
    keyword_weight: float = 0.6
    substring_weight: float = 0.4
    max_query_length: int = 256

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError("min_score 0.0–1.0 के बीच होना चाहिए।")
        if self.max_results <= 0:
            raise ValueError("max_results positive होना चाहिए।")
        if self.max_query_length <= 0:
            raise ValueError("max_query_length positive होना चाहिए।")


# ============================================================
# ✅ Chunk 2: Sanitization + Validation Helpers
# ============================================================

def _sanitize_query(query: str, max_len: int) -> str:
    """
    Query string sanitize करता है।
    """
    if query is None:
        return ""
    q = str(query).strip()
    if len(q) > max_len:
        q = q[:max_len].rstrip()
    return q


def _normalize_text(text: str, case_sensitive: bool) -> str:
    """
    Retrieval scoring के लिए text normalize करता है।
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not case_sensitive:
        t = t.lower()
    return t


def _split_keywords(text: str) -> List[str]:
    """
    Simple keyword split (Phase‑1: whitespace-based)।
    """
    if not text:
        return []
    parts = text.split()
    return [p for p in parts if p]


# ============================================================
# ✅ Core Class: DynamicRetrievalEngine (Phase‑1)
# ============================================================

class DynamicRetrievalEngine:
    """
    Rule-based retrieval engine, Phase‑1 safe version।

    Features:
        - Keyword overlap scoring
        - Substring presence scoring
        - Optional importance boost
        - Optional length penalty (बहुत बड़े text के लिए)
        - Deterministic, explainable scoring
    """

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config: RetrievalConfig = config or RetrievalConfig()
        # Index pool: simple list, कोई inverted index नहीं (Phase‑1)
        self._items: Dict[str, RetrievalItem] = {}

        logger.debug("DynamicRetrievalEngine initialized with config: %r", self.config)

    # --------------------------------------------------------
    # Public API: Add / Remove / Clear Items
    # --------------------------------------------------------

    def add_item(self, item: RetrievalItem) -> None:
        """
        Retrieval index में नया item add करता है।
        """
        self._items[item.id] = item
        logger.debug("Item added to retrieval index: %s", item.id)

    def add_items(self, items: Iterable[RetrievalItem]) -> None:
        """
        Multiple items add करता है।
        """
        for item in items:
            self.add_item(item)

    def remove_item(self, item_id: str) -> bool:
        """
        ID से item remove करता है।
        """
        if item_id in self._items:
            del self._items[item_id]
            logger.debug("Item removed from retrieval index: %s", item_id)
            return True
        return False

    def clear(self) -> None:
        """
        Index खाली करता है।
        """
        count = len(self._items)
        self._items.clear()
        logger.debug("DynamicRetrievalEngine index cleared (removed %d items).", count)

    # --------------------------------------------------------
    # Public API: Retrieval
    # --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Given query string, rule-based scoring से items लौटाता है।
        """
        q = _sanitize_query(query, self.config.max_query_length)
        if not q:
            logger.debug("Empty query, no retrieval.")
            return []

        q_norm = _normalize_text(q, self.config.case_sensitive)
        q_keywords = _split_keywords(q_norm)

        results: List[RetrievalResult] = []

        for item in self._items.values():
            if extra_filters and not self._match_filters(item, extra_filters):
                continue

            score, explanation = self._score_item(
                query_norm=q_norm,
                query_keywords=q_keywords,
                item=item,
            )

            if score >= self.config.min_score:
                results.append(
                    RetrievalResult(
                        item_id=item.id,
                        score=score,
                        text=item.text,
                        metadata=dict(item.metadata),
                        match_explanation=explanation,
                    )
                )

        # sort by score desc, fallback importance
        results.sort(key=lambda r: (r.score, r.metadata.get("importance", 0.0)), reverse=True)
        if len(results) > self.config.max_results:
            results = results[: self.config.max_results]

        logger.debug(
            "Query '%s' produced %d results (min_score=%.2f).",
            q,
            len(results),
            self.config.min_score,
        )

        return results

    # --------------------------------------------------------
    # Public API: Simple Convenience Helpers
    # --------------------------------------------------------

    def top_one(
        self,
        query: str,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[RetrievalResult]:
        """
        Top‑1 result convenience wrapper।
        """
        results = self.retrieve(query, extra_filters=extra_filters)
        return results[0] if results else None

    def has_match(
        self,
        query: str,
        min_score: Optional[float] = None,
    ) -> bool:
        """
        जल्दी check करने के लिए कि कोई match है या नहीं।
        """
        original_min = self.config.min_score
        if min_score is not None:
            self.config.min_score = min_score
        try:
            res = self.top_one(query)
            return res is not None
        finally:
            self.config.min_score = original_min

    # ========================================================
    # Internal: Filters + Scoring
    # ========================================================

    def _match_filters(
        self,
        item: RetrievalItem,
        filters: Dict[str, Any],
    ) -> bool:
        """
        Basic metadata filters (Phase‑1, equality-based)।
        """
        for key, expected in filters.items():
            actual = item.metadata.get(key)
            if actual != expected:
                return False
        return True

    def _score_item(
        self,
        query_norm: str,
        query_keywords: List[str],
        item: RetrievalItem,
    ) -> Tuple[float, str]:
        """
        Query और item के बीच rule-based score निकालता है।
        """
        text_norm = _normalize_text(item.text, self.config.case_sensitive)
        item_keywords = _split_keywords(text_norm)

        keyword_score, kw_detail = self._keyword_overlap_score(
            query_keywords,
            item_keywords,
        )
        substring_score, sub_detail = self._substring_score(
            query_norm,
            text_norm,
        )

        score = (
            self.config.keyword_weight * keyword_score
            + self.config.substring_weight * substring_score
        )

        # Optional importance boost
        if self.config.boost_importance:
            importance_boost = item.importance * 0.2  # 20% weight
            score += importance_boost
            importance_note = f" + importance_boost={importance_boost:.2f}"
        else:
            importance_note = ""

        # Optional length penalty
        length_note = ""
        if self.config.length_penalty:
            penalty = self._length_penalty(len(item_keywords))
            if penalty < 1.0:
                score *= penalty
                length_note = f" * length_penalty={penalty:.2f}"

        score = max(0.0, min(1.0, score))

        explanation = (
            f"keyword_score={keyword_score:.2f}, "
            f"substring_score={substring_score:.2f}"
            f"{importance_note}{length_note}; "
            f"{kw_detail}; {sub_detail}"
        )

        return score, explanation

    def _keyword_overlap_score(
        self,
        q_keywords: List[str],
        item_keywords: List[str],
    ) -> Tuple[float, str]:
        """
        Query और item keywords के overlap पर score।
        """
        if not q_keywords or not item_keywords:
            return 0.0, "no keywords"

        q_set = set(q_keywords)
        i_set = set(item_keywords)

        common = q_set.intersection(i_set)
        if not common:
            return 0.0, "no keyword overlap"

        overlap_ratio = len(common) / len(q_set)
        detail = f"keyword_overlap={len(common)}/{len(q_set)}"

        return overlap_ratio, detail

    def _substring_score(
        self,
        query_norm: str,
        text_norm: str,
    ) -> Tuple[float, str]:
        """
        Simple substring presence-based score।
        """
        if not query_norm or not text_norm:
            return 0.0, "no substring match"

        if query_norm in text_norm:
            # exact substring presence
            return 1.0, "full substring match"

        # partial heuristic: proportion of query tokens present
        q_tokens = _split_keywords(query_norm)
        if not q_tokens:
            return 0.0, "no query tokens"

        hits = 0
        for token in q_tokens:
            if token in text_norm:
                hits += 1

        ratio = hits / len(q_tokens)
        return ratio, f"partial substring hits={hits}/{len(q_tokens)}"

    def _length_penalty(self, num_tokens: int) -> float:
        """
        बहुत बड़े text के लिए simple penalty (ताकि छोटे precise text को थोड़ा boost मिले)।
        """
        if num_tokens <= 0:
            return 1.0
        if num_tokens <= 50:
            return 1.0
        if num_tokens <= 200:
            return 0.9
        if num_tokens <= 500:
            return 0.8
        return 0.7

    # ========================================================
    # Introspection & Debug Utilities
    # ========================================================

    def debug_items_snapshot(self) -> List[Dict[str, Any]]:
        """
        Retrieval index का simple snapshot लौटाता है (debug के लिए)।
        """
        snapshot: List[Dict[str, Any]] = []
        for item in self._items.values():
            snapshot.append(
                {
                    "id": item.id,
                    "text_preview": item.text[:80],
                    "importance": item.importance,
                    "metadata": dict(item.metadata),
                }
            )
        return snapshot

    def explain_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query के लिए top results + explanation structured form में देता है।
        """
        results = self.retrieve(query)
        results = results[: max(1, top_k)]

        explained: List[Dict[str, Any]] = []
        for res in results:
            explained.append(
                {
                    "id": res.item_id,
                    "score": res.score,
                    "explanation": res.match_explanation,
                    "text_preview": res.text[:120],
                }
            )
        return explained


# ============================================================
# Public Factory Helper
# ============================================================

def create_default_dynamic_retrieval_engine() -> DynamicRetrievalEngine:
    """
    Default Phase‑1 config के साथ retrieval engine बनाता है।
    """
    config = RetrievalConfig()
    return DynamicRetrievalEngine(config=config)


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "RetrievalItem",
    "RetrievalResult",
    "RetrievalConfig",
    "DynamicRetrievalEngine",
    "create_default_dynamic_retrieval_engine",
]

# End of File: memory/dynamic_retrieval.py (Phase‑1, ~500 lines with comments)
