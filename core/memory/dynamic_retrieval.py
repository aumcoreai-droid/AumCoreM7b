"""
Dynamic Retrieval Module for AumCore_AI

This module implements intelligent memory retrieval based on context relevance,
recency, and importance scoring. Provides adaptive search strategies.

Phase-1: Rule-based retrieval with heuristic scoring.
Future: Integration with Mistral-7B for semantic similarity matching.

File: core/memory/dynamic_retrieval.py
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

# Type annotations
from typing import TypedDict


# ============================================================================
# CONFIGURATION & TYPES
# ============================================================================

class RetrievalConfig(TypedDict):
    """Configuration for dynamic retrieval."""
    max_results: int
    recency_weight: float
    importance_weight: float
    relevance_weight: float
    enable_context_boost: bool


class ScoredResult(TypedDict):
    """Type definition for a scored retrieval result."""
    entry_id: str
    content: str
    score: float
    relevance: float
    recency: float
    importance: float


# ============================================================================
# ABSTRACT INTERFACES (SOLID: Dependency Inversion)
# ============================================================================

class MemorySourceProtocol(Protocol):
    """Protocol for memory sources."""
    
    def get_all_entries(self) -> List[Any]:
        """Retrieve all memory entries."""
        ...
    
    def get_entry(self, entry_id: str) -> Optional[Any]:
        """Retrieve specific entry."""
        ...


class ScoringStrategyProtocol(Protocol):
    """Protocol for scoring strategies."""
    
    def calculate_score(
        self,
        entry: Any,
        query: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for entry."""
        ...


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RetrievalContext:
    """Context information for retrieval operations."""
    
    query: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[str] = field(default_factory=list)
    active_topics: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate context after initialization."""
        if not self.query or not isinstance(self.query, str):
            raise ValueError("Query must be a non-empty string")


@dataclass
class RetrievalResult:
    """Represents a single retrieval result with scoring details."""
    
    entry_id: str
    content: str
    final_score: float
    relevance_score: float
    recency_score: float
    importance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> ScoredResult:
        """Convert to dictionary format."""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "score": self.final_score,
            "relevance": self.relevance_score,
            "recency": self.recency_score,
            "importance": self.importance_score
        }


# ============================================================================
# SCORING STRATEGIES
# ============================================================================

class HeuristicScorer:
    """
    Phase-1 heuristic-based scoring strategy.
    
    Calculates relevance using:
    - Keyword matching
    - Term frequency
    - Query overlap
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize heuristic scorer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_relevance(self, content: str, query: str) -> float:
        """
        Calculate relevance score between content and query.
        
        Args:
            content: Memory content
            query: Search query
        
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not content or not query:
            return 0.0
        
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Exact match bonus
        if query_lower in content_lower:
            return 1.0
        
        # Tokenize query and content
        query_tokens = self._tokenize(query_lower)
        content_tokens = self._tokenize(content_lower)
        
        if not query_tokens or not content_tokens:
            return 0.0
        
        # Calculate token overlap
        matching_tokens = set(query_tokens) & set(content_tokens)
        overlap_ratio = len(matching_tokens) / len(query_tokens)
        
        # Calculate term frequency score
        tf_score = sum(
            content_tokens.count(token) for token in matching_tokens
        ) / len(content_tokens)
        
        # Combined score
        relevance = (overlap_ratio * 0.7) + (tf_score * 0.3)
        
        return min(1.0, relevance)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (Phase-1)."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Filter stopwords (basic list)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are'
        }
        
        return [t for t in tokens if t not in stopwords and len(t) > 2]


# ============================================================================
# CORE DYNAMIC RETRIEVAL CLASS
# ============================================================================

class DynamicRetrieval:
    """
    Intelligent memory retrieval system with adaptive scoring.
    
    Features:
    - Multi-factor scoring (relevance, recency, importance)
    - Configurable weight distribution
    - Context-aware boosting
    - Category filtering
    - Diversity-aware ranking
    
    Attributes:
        config: Retrieval configuration
        scorer: Relevance scoring strategy
        logger: Centralized logger instance
    
    Example:
        >>> retriever = DynamicRetrieval()
        >>> context = RetrievalContext(query="Python coding")
        >>> results = retriever.retrieve(memory_source, context)
    """
    
    def __init__(
        self,
        max_results: int = 10,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
        relevance_weight: float = 0.4,
        enable_context_boost: bool = True,
        scorer: Optional[HeuristicScorer] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize dynamic retrieval system.
        
        Args:
            max_results: Maximum number of results to return
            recency_weight: Weight for recency score (0.0-1.0)
            importance_weight: Weight for importance score (0.0-1.0)
            relevance_weight: Weight for relevance score (0.0-1.0)
            enable_context_boost: Enable context-based score boosting
            scorer: Custom scoring strategy (optional)
            logger: Optional logger instance
        
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        # Validate weights
        total_weight = recency_weight + importance_weight + relevance_weight
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}"
            )
        
        # Configuration
        self.config: RetrievalConfig = {
            "max_results": max_results,
            "recency_weight": recency_weight,
            "importance_weight": importance_weight,
            "relevance_weight": relevance_weight,
            "enable_context_boost": enable_context_boost
        }
        
        # Components
        self.scorer = scorer or HeuristicScorer()
        self.logger = logger or self._setup_logger()
        
        self.logger.info(
            f"DynamicRetrieval initialized: "
            f"weights=[R:{recency_weight}, I:{importance_weight}, V:{relevance_weight}]"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup centralized logger (Phase-1 fallback)."""
        logger = logging.getLogger("aumcore.memory.dynamic_retrieval")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # ========================================================================
    # PUBLIC API - RETRIEVAL OPERATIONS
    # ========================================================================
    
    def retrieve(
        self,
        memory_source: MemorySourceProtocol,
        context: RetrievalContext,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve and rank memory entries based on context.
        
        Args:
            memory_source: Source of memory entries
            context: Retrieval context with query and metadata
            filters: Optional filtering criteria
        
        Returns:
            List of ranked retrieval results
        
        Example:
            >>> context = RetrievalContext(query="machine learning")
            >>> results = retriever.retrieve(memory, context)
        """
        try:
            # Get all entries from source
            all_entries = memory_source.get_all_entries()
            
            if not all_entries:
                self.logger.warning("No entries found in memory source")
                return []
            
            # Apply filters
            if filters:
                all_entries = self._apply_filters(all_entries, filters)
            
            # Score each entry
            scored_results = []
            for entry in all_entries:
                result = self._score_entry(entry, context)
                if result.final_score > 0.0:  # Only include non-zero scores
                    scored_results.append(result)
            
            # Sort by final score (descending)
            scored_results.sort(key=lambda r: r.final_score, reverse=True)
            
            # Apply diversity if enabled
            if self.config["enable_context_boost"]:
                scored_results = self._apply_diversity(scored_results)
            
            # Limit results
            results = scored_results[:self.config["max_results"]]
            
            self.logger.debug(
                f"Retrieved {len(results)} results for query: {context.query}"
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            raise
    
    def retrieve_similar(
        self,
        memory_source: MemorySourceProtocol,
        reference_content: str,
        limit: int = 5
    ) -> List[RetrievalResult]:
        """
        Find entries similar to reference content.
        
        Args:
            memory_source: Source of memory entries
            reference_content: Reference text for similarity
            limit: Maximum results
        
        Returns:
            List of similar entries
        
        Example:
            >>> similar = retriever.retrieve_similar(memory, "Python code")
        """
        context = RetrievalContext(query=reference_content)
        results = self.retrieve(memory_source, context)
        return results[:limit]
    
    def retrieve_by_category(
        self,
        memory_source: MemorySourceProtocol,
        category: str,
        context: RetrievalContext
    ) -> List[RetrievalResult]:
        """
        Retrieve entries from specific category.
        
        Args:
            memory_source: Source of memory entries
            category: Category to filter
            context: Retrieval context
        
        Returns:
            List of category-filtered results
        
        Example:
            >>> results = retriever.retrieve_by_category(memory, "preference", ctx)
        """
        filters = {"category": category}
        return self.retrieve(memory_source, context, filters)
    
    def retrieve_recent(
        self,
        memory_source: MemorySourceProtocol,
        hours: int = 24,
        limit: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve recent entries within time window.
        
        Args:
            memory_source: Source of memory entries
            hours: Time window in hours
            limit: Maximum results
        
        Returns:
            List of recent entries
        
        Example:
            >>> recent = retriever.retrieve_recent(memory, hours=48)
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filters = {"min_timestamp": cutoff_time}
        
        # Use empty query to rely on recency scoring
        context = RetrievalContext(query="")
        results = self.retrieve(memory_source, context, filters)
        
        return results[:limit]
    
    # ========================================================================
    # SCORING LOGIC
    # ========================================================================
    
    def _score_entry(
        self,
        entry: Any,
        context: RetrievalContext
    ) -> RetrievalResult:
        """
        Calculate comprehensive score for a memory entry.
        
        Args:
            entry: Memory entry to score
            context: Retrieval context
        
        Returns:
            RetrievalResult with all scoring components
        """
        # Extract entry attributes (assuming standard interface)
        entry_id = getattr(entry, 'id', 'unknown')
        content = getattr(entry, 'content', '')
        timestamp = getattr(entry, 'timestamp', datetime.now())
        importance = getattr(entry, 'importance', 0.5)
        
        # Calculate individual scores
        relevance_score = self._calculate_relevance_score(content, context)
        recency_score = self._calculate_recency_score(timestamp)
        importance_score = importance  # Already normalized 0.0-1.0
        
        # Apply context boost if enabled
        if self.config["enable_context_boost"]:
            relevance_score = self._apply_context_boost(
                relevance_score, entry, context
            )
        
        # Calculate weighted final score
        final_score = (
            self.config["relevance_weight"] * relevance_score +
            self.config["recency_weight"] * recency_score +
            self.config["importance_weight"] * importance_score
        )
        
        return RetrievalResult(
            entry_id=entry_id,
            content=content,
            final_score=final_score,
            relevance_score=relevance_score,
            recency_score=recency_score,
            importance_score=importance_score,
            metadata=getattr(entry, 'metadata', {})
        )
    
    def _calculate_relevance_score(
        self,
        content: str,
        context: RetrievalContext
    ) -> float:
        """Calculate relevance score using scorer."""
        if not context.query:
            return 0.5  # Neutral score for empty query
        
        return self.scorer.calculate_relevance(content, context.query)
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """
        Calculate recency score using exponential decay.
        
        Args:
            timestamp: Entry timestamp
        
        Returns:
            Recency score (0.0 to 1.0)
        """
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        
        # Exponential decay with half-life of 24 hours
        half_life = 24.0
        decay_rate = 0.693 / half_life  # ln(2) / half_life
        
        recency_score = 2 ** (-decay_rate * age_hours)
        
        return max(0.0, min(1.0, recency_score))
    
    def _apply_context_boost(
        self,
        base_score: float,
        entry: Any,
        context: RetrievalContext
    ) -> float:
        """
        Apply context-based score boosting.
        
        Args:
            base_score: Base relevance score
            entry: Memory entry
            context: Retrieval context
        
        Returns:
            Boosted score
        """
        boost = 0.0
        
        # Check if entry category matches active topics
        entry_category = getattr(entry, 'category', None)
        if entry_category and entry_category in context.active_topics:
            boost += 0.1
        
        # Check user preferences alignment
        if context.user_preferences:
            entry_metadata = getattr(entry, 'metadata', {})
            for pref_key, pref_value in context.user_preferences.items():
                if entry_metadata.get(pref_key) == pref_value:
                    boost += 0.05
        
        # Check conversation history overlap
        if context.conversation_history:
            content = getattr(entry, 'content', '').lower()
            for msg in context.conversation_history[-3:]:  # Last 3 messages
                if any(word in content for word in msg.lower().split()):
                    boost += 0.05
                    break
        
        # Apply boost (max 20% increase)
        boosted_score = base_score * (1.0 + min(boost, 0.2))
        
        return min(1.0, boosted_score)
    
    # ========================================================================
    # FILTERING & RANKING
    # ========================================================================
    
    def _apply_filters(
        self,
        entries: List[Any],
        filters: Dict[str, Any]
    ) -> List[Any]:
        """
        Apply filtering criteria to entries.
        
        Args:
            entries: List of memory entries
            filters: Filter criteria
        
        Returns:
            Filtered list of entries
        """
        filtered = entries
        
        # Category filter
        if "category" in filters:
            filtered = [
                e for e in filtered
                if getattr(e, 'category', None) == filters["category"]
            ]
        
        # Minimum timestamp filter
        if "min_timestamp" in filters:
            min_ts = filters["min_timestamp"]
            filtered = [
                e for e in filtered
                if getattr(e, 'timestamp', datetime.min) >= min_ts
            ]
        
        # Minimum importance filter
        if "min_importance" in filters:
            min_imp = filters["min_importance"]
            filtered = [
                e for e in filtered
                if getattr(e, 'importance', 0.0) >= min_imp
            ]
        
        return filtered
    
    def _apply_diversity(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Apply diversity to prevent redundant results.
        
        Args:
            results: Scored results list
        
        Returns:
            Diversified results list
        """
        if len(results) <= 3:
            return results
        
        diverse_results = [results[0]]  # Always include top result
        
        for result in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            for selected in diverse_results:
                similarity = self._calculate_similarity(
                    result.content, selected.content
                )
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (Phase-1: simple overlap).
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        tokens1 = set(self.scorer._tokenize(text1.lower()))
        tokens2 = set(self.scorer._tokenize(text2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0
    
    # ========================================================================
    # ASYNC API (Future-ready)
    # ========================================================================
    
    async def retrieve_async(
        self,
        memory_source: MemorySourceProtocol,
        context: RetrievalContext,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Async version of retrieve."""
        return await asyncio.to_thread(
            self.retrieve, memory_source, context, filters
        )
    
    # ========================================================================
    # CONFIGURATION MANAGEMENT
    # ========================================================================
    
    def update_weights(
        self,
        recency: Optional[float] = None,
        importance: Optional[float] = None,
        relevance: Optional[float] = None
    ) -> None:
        """
        Update scoring weights dynamically.
        
        Args:
            recency: New recency weight
            importance: New importance weight
            relevance: New relevance weight
        
        Raises:
            ValueError: If weights don't sum to 1.0
        
        Example:
            >>> retriever.update_weights(recency=0.4, relevance=0.5)
        """
        new_recency = recency if recency is not None else self.config["recency_weight"]
        new_importance = importance if importance is not None else self.config["importance_weight"]
        new_relevance = relevance if relevance is not None else self.config["relevance_weight"]
        
        total = new_recency + new_importance + new_relevance
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.config["recency_weight"] = new_recency
        self.config["importance_weight"] = new_importance
        self.config["relevance_weight"] = new_relevance
        
        self.logger.info(f"Updated weights: R:{new_recency}, I:{new_importance}, V:{new_relevance}")


# ============================================================================
# FACTORY PATTERN (Dependency Injection)
# ============================================================================

def create_dynamic_retrieval(
    config: Optional[RetrievalConfig] = None,
    logger: Optional[logging.Logger] = None
) -> DynamicRetrieval:
    """
    Factory function to create DynamicRetrieval instance.
    
    Args:
        config: Optional retrieval configuration
        logger: Optional logger instance
    
    Returns:
        Configured DynamicRetrieval instance
    
    Example:
        >>> retriever = create_dynamic_retrieval({"max_results": 15})
    """
    if config is None:
        config = {
            "max_results": 10,
            "recency_weight": 0.3,
            "importance_weight": 0.3,
            "relevance_weight": 0.4,
            "enable_context_boost": True
        }
    
    return DynamicRetrieval(
        max_results=config["max_results"],
        recency_weight=config["recency_weight"],
        importance_weight=config["importance_weight"],
        relevance_weight=config["relevance_weight"],
        enable_context_boost=config["enable_context_boost"],
        logger=logger
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "DynamicRetrieval",
    "RetrievalContext",
    "RetrievalResult",
    "HeuristicScorer",
    "RetrievalConfig",
    "ScoredResult",
    "create_dynamic_retrieval"
]


# ============================================================================
# TODO: FUTURE MODEL INTEGRATION (Phase-2+)
# ============================================================================

# TODO: Integrate Mistral-7B for:
# - Semantic similarity scoring
# - Neural re-ranking of results
# - Query expansion and reformulation
# - Context understanding for boosting

# TODO: Add vector embeddings for semantic search

# TODO: Implement learned-to-rank models

# TODO: Add A/B testing framework for scoring strategies

# TODO: Implement caching layer for frequent queries