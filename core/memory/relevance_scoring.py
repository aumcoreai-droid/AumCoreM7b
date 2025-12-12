"""
AumCore_AI Memory Subsystem - Relevance Scoring Module
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/relevance_scoring.py

Description:
    यह module Phase‑1 में simple, rule-based, in-memory
    relevance scoring provide करता है long-term / short-term
    memory entries के लिए।

    Core Idea (Phase‑1):
        - Query + memory metadata (importance, seen_count, recency, source)
          के basis पर deterministic relevance score निकालना।
        - कोई model-based embeddings, vector store या stochastic scoring नहीं।
        - Pure rule-based, explainable, config-driven scoring logic।

    Future (Phase‑3+):
        - Embedding-based semantic similarity
        - Learned weights / adaptive scoring
        - Cross-source ranking + experimentation hooks
"""

# ============================================================
# ✅ Chunk 1: Imports (PEP8 + Strict Typing)
# ============================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Protocol


# ============================================================
# ✅ Chunk 1: Module Metadata
# ============================================================

MODULE_NAME: str = "AumCoreAI.Memory.RelevanceScoring"
MODULE_VERSION: str = "1.0.0"
MODULE_AUTHOR: str = "AumCore_AI"
MODULE_CREATED_ON: str = "2025-12-12"
MODULE_UPDATED_ON: str = "2025-12-12"


# ============================================================
# ✅ Chunk 1: Local Logger Setup (Structured-ish)
# ============================================================

def get_relevance_logger(name: str = MODULE_NAME) -> logging.Logger:
    """
    Relevance scoring module के लिए local logger बनाता/return करता है।

    Args:
        name: Logger नाम (namespace जैसा)।

    Returns:
        logging.Logger: Configured logger instance।
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [RELEVANCE_SCORING] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger: logging.Logger = get_relevance_logger()


# ============================================================
# ✅ Chunk 2: Custom Error Classes
# ============================================================

class RelevanceModuleError(Exception):
    """
    Base error type for relevance scoring module।
    """


class RelevanceValidationError(RelevanceModuleError):
    """
    Input validation या state inconsistencies के लिए error।
    """


class RelevanceConfigError(RelevanceModuleError):
    """
    Config missing/invalid होने पर error।
    """


# ============================================================
# ✅ Chunk 2: Config Loader Protocol (YAML-style Stub)
# ============================================================

class ConfigLoader(Protocol):
    """
    Config loader protocol (Phase‑1 stub)।

    Future integration:
        - Central YAML config service
        - Environment-based overrides
    """

    def load(self, module_name: str) -> Dict[str, Any]:
        """
        दिए हुए module के लिए config load करता है।

        Args:
            module_name: Module का canonical नाम।

        Returns:
            dict: Module-specific config mapping।
        """


@dataclass
class SimpleRelevanceConfigLoader:
    """
    Simple in-memory config loader (Phase‑1 stub)।

    Attributes:
        global_config: पूरे system का merged config dict।
    """

    global_config: Dict[str, Any] = field(default_factory=dict)

    def load(self, module_name: str) -> Dict[str, Any]:
        """
        Module-specific config return करता है, safe defaults के साथ।

        Args:
            module_name: Module का नाम।

        Returns:
            dict: Config mapping, या safe defaults।
        """
        config = self.global_config.get(module_name, {})
        if config:
            return config

        # Safe fallback config (Phase‑1, कोई external YAML नहीं)
        return {
            "logging": {
                "level": "DEBUG",
            },
            "relevance": {
                "max_score": 1.0,
                "min_score": 0.0,
                "default_importance_weight": 0.5,
                "default_seen_count_weight": 0.2,
                "default_recency_weight": 0.2,
                "default_source_weight": 0.1,
                "recency_half_life_days": 14,
                "max_age_days_for_full_score": 30,
            },
            "sources": {
                "default_weight": 1.0,
                "weights": {
                    "user": 1.2,
                    "assistant": 0.9,
                    "system": 1.0,
                    "profile": 1.1,
                    "other": 1.0,
                },
            },
        }


# ============================================================
# ✅ Chunk 2: Performance Timers + Telemetry Hooks (Simple)
# ============================================================

@dataclass
class TelemetryRecord:
    """
    Simple telemetry रिकॉर्ड (Phase‑1 in-memory only)।

    Fields:
        operation: Operation का नाम (e.g., "score_batch")।
        duration_ms: Execution time milliseconds में।
        timestamp: Event का time (epoch seconds)।
        meta: Extra metadata (e.g., item_count, mode)।
    """

    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryCollector:
    """
    Basic telemetry collector (Phase‑1, in-memory only)।
    """

    records: List[TelemetryRecord] = field(default_factory=list)

    def record(self, operation: str, duration_ms: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        नया telemetry record add करता है।

        Args:
            operation: Operation का नाम।
            duration_ms: Duration in milliseconds।
            meta: Optional metadata।
        """
        rec = TelemetryRecord(operation=operation, duration_ms=duration_ms, meta=dict(meta or {}))
        self.records.append(rec)

    def snapshot(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Latest N telemetry records का snapshot देता है।

        Args:
            limit: Maximum records count।

        Returns:
            list of dict: Telemetry records का serialized रूप।
        """
        recent = self.records[-max(1, limit):]
        return [
            {
                "operation": r.operation,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp,
                "meta": r.meta,
            }
            for r in recent
        ]


telemetry: TelemetryCollector = TelemetryCollector()


def timed_operation(operation_name: str) -> Any:
    """
    Decorator जो operation को measure करता है और telemetry में record करता है।

    Args:
        operation_name: Operation का नाम (string)।

    Returns:
        Wrapped function।
    """

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                telemetry.record(operation_name, duration_ms, meta={"sync": True})
                logger.debug(
                    "Timed operation '%s' duration_ms=%.3f",
                    operation_name,
                    duration_ms,
                )

        return wrapper

    return decorator


def timed_operation_async(operation_name: str) -> Any:
    """
    Async decorator version for timing operations।

    Args:
        operation_name: Operation का नाम।

    Returns:
        Wrapped async function।
    """

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                telemetry.record(operation_name, duration_ms, meta={"sync": False})
                logger.debug(
                    "Timed async operation '%s' duration_ms=%.3f",
                    operation_name,
                    duration_ms,
                )

        return wrapper

    return decorator


# ============================================================
# ✅ Chunk 2: Sanitization + Validation Utilities
# ============================================================

def _clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Floating-point value को given range में clamp करता है।

    Args:
        value: Input float।
        min_value: Minimum allowed।
        max_value: Maximum allowed।

    Returns:
        float: Clamped value।
    """
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _safe_str(text: Any) -> str:
    """
    Generic text को safe string में convert करता है।

    Args:
        text: कोई भी value।

    Returns:
        str: Trimmed string representation।
    """
    if text is None:
        return ""
    return str(text).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Any को safe float में convert करता है, error पर default।

    Args:
        value: Input value।
        default: Fallback value।

    Returns:
        float: Parsed या default।
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Any को safe int में convert करता है।

    Args:
        value: Input value।
        default: Fallback value।

    Returns:
        int: Parsed या default।
    """
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _age_in_days(created_at: datetime, now: Optional[datetime] = None) -> float:
    """
    Created_at और now के बीच age (days) return करता है।

    Args:
        created_at: Creation time।
        now: Optional override for current time।

    Returns:
        float: Age in days।
    """
    now_dt = now or datetime.now()
    delta: timedelta = now_dt - created_at
    return max(delta.total_seconds() / 86400.0, 0.0)


# ============================================================
# ✅ Chunk 2: Data Structures for Scoring
# ============================================================

@dataclass
class RelevanceConfig:
    """
    Relevance scoring के लिए Phase‑1 configuration।

    Fields:
        max_score: Upper bound for final score।
        min_score: Lower bound for final score।
        importance_weight: importance factor का weight।
        seen_count_weight: seen_count factor का weight।
        recency_weight: recency factor का weight।
        source_weight: source factor का weight।
        recency_half_life_days: Exponential decay half life।
        max_age_days_for_full_score: इस age तक full recency weight।
    """

    max_score: float = 1.0
    min_score: float = 0.0
    importance_weight: float = 0.5
    seen_count_weight: float = 0.2
    recency_weight: float = 0.2
    source_weight: float = 0.1
    recency_half_life_days: float = 14.0
    max_age_days_for_full_score: float = 30.0

    def __post_init__(self) -> None:
        """
        Config invariants validate करता है।
        """
        if self.max_score <= self.min_score:
            raise RelevanceConfigError("max_score > min_score होना चाहिए।")
        if self.importance_weight < 0.0:
            raise RelevanceConfigError("importance_weight negative नहीं हो सकता।")
        if self.seen_count_weight < 0.0:
            raise RelevanceConfigError("seen_count_weight negative नहीं हो सकता।")
        if self.recency_weight < 0.0:
            raise RelevanceConfigError("recency_weight negative नहीं हो सकता।")
        if self.source_weight < 0.0:
            raise RelevanceConfigError("source_weight negative नहीं हो सकता।")
        if self.recency_half_life_days <= 0.0:
            raise RelevanceConfigError("recency_half_life_days positive होना चाहिए।")
        if self.max_age_days_for_full_score <= 0.0:
            raise RelevanceConfigError("max_age_days_for_full_score positive होना चाहिए।")


@dataclass
class SourceWeightConfig:
    """
    Different memory sources के लिए weights।

    Fields:
        default_weight: Unknown sources के लिए weight।
        weights: Source -> weight mapping।
    """

    default_weight: float = 1.0
    weights: Dict[str, float] = field(default_factory=dict)

    def get_weight(self, source: str) -> float:
        """
        Source-specific weight return करता है।

        Args:
            source: Source name (e.g., "user", "system")।

        Returns:
            float: Weight value।
        """
        key = _safe_str(source).lower() or "other"
        return _safe_float(self.weights.get(key, self.default_weight), self.default_weight)


@dataclass
class MemoryForScoring:
    """
    Relevance scoring के लिए normalized memory representation।

    Fields:
        id: Unique identifier।
        text: Content text।
        importance: 0.0–1.0 scale।
        seen_count: कितनी बार observe/use हुआ।
        created_at: Creation time।
        source: Source string (user/system/profile/etc)।
        extra: Additional metadata।
    """

    id: str
    text: str
    importance: float
    seen_count: int
    created_at: datetime
    source: str = "other"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Basic validation।
        """
        if not self.id or not isinstance(self.id, str):
            raise RelevanceValidationError("MemoryForScoring.id non-empty string होना चाहिए।")
        if not isinstance(self.created_at, datetime):
            raise RelevanceValidationError("MemoryForScoring.created_at datetime होना चाहिए।")
        if not (0.0 <= self.importance <= 1.0):
            raise RelevanceValidationError("importance 0.0–1.0 के बीच होना चाहिए।")
        if self.seen_count < 0:
            raise RelevanceValidationError("seen_count negative नहीं होना चाहिए।")


@dataclass
class ScoredMemory:
    """
    Scoring के बाद memory item, detailed breakdown के साथ।

    Fields:
        memory: Original memory representation।
        score: Final relevance score।
        breakdown: Different components का per-factor score।
    """

    memory: MemoryForScoring
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


# ============================================================
# ✅ Chunk 2: Unified Response Schema Helper
# ============================================================

def make_response(
    status: str,
    data: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Standard response schema create करता है।

    Args:
        status: "success" या "error"।
        data: Actual payload data।
        meta: Metadata (e.g., timing, counts)।
        errors: Error objects की list।

    Returns:
        dict: Normalized response।
    """
    return {
        "status": status,
        "data": data or {},
        "meta": meta or {},
        "errors": errors or [],
    }


# ============================================================
# ✅ Core Class: Relevance Scoring Engine (Phase‑1)
# ============================================================

class RelevanceScoringEngine:
    """
    Phase‑1 compatible relevance scoring engine।

    Features:
        - Deterministic, rule-based scoring
        - Simple, explainable factors:
            - importance
            - seen_count
            - recency
            - source_weight
        - Config-driven weights और bounds
        - Safe defaults + validation
    """

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        module_name: str = MODULE_NAME,
    ) -> None:
        """
        Engine को config loader के साथ initialize करता है।

        Args:
            config_loader: Config loader (optional, default internal stub)।
            module_name: Module name for config।
        """
        self.module_name: str = module_name
        self._config_loader: ConfigLoader = config_loader or SimpleRelevanceConfigLoader()
        raw_config: Dict[str, Any] = self._config_loader.load(module_name)

        relevance_cfg = raw_config.get("relevance", {})
        sources_cfg = raw_config.get("sources", {})

        self.config: RelevanceConfig = RelevanceConfig(
            max_score=_safe_float(relevance_cfg.get("max_score", 1.0), 1.0),
            min_score=_safe_float(relevance_cfg.get("min_score", 0.0), 0.0),
            importance_weight=_safe_float(relevance_cfg.get("default_importance_weight", 0.5), 0.5),
            seen_count_weight=_safe_float(relevance_cfg.get("default_seen_count_weight", 0.2), 0.2),
            recency_weight=_safe_float(relevance_cfg.get("default_recency_weight", 0.2), 0.2),
            source_weight=_safe_float(relevance_cfg.get("default_source_weight", 0.1), 0.1),
            recency_half_life_days=_safe_float(relevance_cfg.get("recency_half_life_days", 14.0), 14.0),
            max_age_days_for_full_score=_safe_float(
                relevance_cfg.get("max_age_days_for_full_score", 30.0),
                30.0,
            ),
        )

        self.source_config: SourceWeightConfig = SourceWeightConfig(
            default_weight=_safe_float(sources_cfg.get("default_weight", 1.0), 1.0),
            weights=dict(sources_cfg.get("weights", {})),
        )

        logger.debug("RelevanceScoringEngine initialized with config: %r", self.config)

    # --------------------------------------------------------
    # Internal Scoring Helpers
    # --------------------------------------------------------

    def _compute_importance_component(self, importance: float) -> float:
        """
        Importance component compute करता है।

        Args:
            importance: 0.0–1.0।

        Returns:
            float: Component score।
        """
        imp = _clamp(importance, 0.0, 1.0)
        return imp * self.config.importance_weight

    def _compute_seen_component(self, seen_count: int) -> float:
        """
        Seen_count component compute करता है।

        Args:
            seen_count: Non-negative integer।

        Returns:
            float: Component score (log-like squashing)।
        """
        c = max(seen_count, 0)
        if c == 0:
            base = 0.0
        elif c == 1:
            base = 0.2
        elif c == 2:
            base = 0.4
        else:
            base = 0.6 + min((c - 2) * 0.05, 0.4)
        return base * self.config.seen_count_weight

    def _compute_recency_component(self, created_at: datetime, now: Optional[datetime] = None) -> float:
        """
        Recency component compute करता है, exponential decay से।

        Args:
            created_at: Memory का creation time।
            now: Optional override current time।

        Returns:
            float: Component score।
        """
        age_days = _age_in_days(created_at, now)
        if age_days <= 0.0:
            base = 1.0
        elif age_days <= self.config.max_age_days_for_full_score:
            base = 1.0
        else:
            decay_factor = 0.5 ** ((age_days - self.config.max_age_days_for_full_score) / self.config.recency_half_life_days)
            base = max(decay_factor, 0.0)
        return base * self.config.recency_weight

    def _compute_source_component(self, source: str) -> float:
        """
        Source weighting component compute करता है।

        Args:
            source: Source string।

        Returns:
            float: Component score।
        """
        weight = self.source_config.get_weight(source)
        normalized = _clamp(weight / 2.0, 0.0, 1.0)  # normalize approx, Phase‑1
        return normalized * self.config.source_weight

    def _normalize_score(self, raw_score: float) -> float:
        """
        Raw score को configured bounds में normalize करता है।

        Args:
            raw_score: Unbounded raw score।

        Returns:
            float: Final score [min_score, max_score] में।
        """
        return _clamp(raw_score, self.config.min_score, self.config.max_score)

    # --------------------------------------------------------
    # Public API: Single Memory Scoring (Sync)
    # --------------------------------------------------------

    @timed_operation("score_single")
    def score_memory(self, memory: MemoryForScoring, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Single memory item के लिए relevance score compute करता है।

        Args:
            memory: MemoryForScoring object।
            now: Optional current time override।

        Returns:
            dict: Standard response schema with score + breakdown।
        """
        try:
            now_dt = now or datetime.now()

            imp_component = self._compute_importance_component(memory.importance)
            seen_component = self._compute_seen_component(memory.seen_count)
            rec_component = self._compute_recency_component(memory.created_at, now_dt)
            src_component = self._compute_source_component(memory.source)

            raw_score = imp_component + seen_component + rec_component + src_component
            final_score = self._normalize_score(raw_score)

            breakdown = {
                "importance_component": imp_component,
                "seen_component": seen_component,
                "recency_component": rec_component,
                "source_component": src_component,
                "raw_score": raw_score,
                "final_score": final_score,
            }

            logger.debug(
                "Score computed for memory_id=%s final_score=%.4f breakdown=%r",
                memory.id,
                final_score,
                breakdown,
            )

            return make_response(
                status="success",
                data={
                    "memory_id": memory.id,
                    "score": final_score,
                    "breakdown": breakdown,
                },
                meta={
                    "module": self.module_name,
                    "version": MODULE_VERSION,
                },
            )
        except RelevanceModuleError as exc:
            logger.error("RelevanceModuleError in score_memory: %s", exc)
            return make_response(
                status="error",
                data={},
                meta={"module": self.module_name},
                errors=[{"type": "RelevanceModuleError", "message": str(exc)}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in score_memory")
            return make_response(
                status="error",
                data={},
                meta={"module": self.module_name},
                errors=[{"type": "UnexpectedError", "message": str(exc)}],
            )

    # --------------------------------------------------------
    # Public API: Batch Scoring (Sync)
    # --------------------------------------------------------

    @timed_operation("score_batch")
    def score_batch(
        self,
        memories: List[MemoryForScoring],
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Multiple memories के लिए relevance score compute करता है।

        Args:
            memories: MemoryForScoring की list।
            now: Optional current time override।

        Returns:
            dict: Standard response schema with scored list।
        """
        try:
            now_dt = now or datetime.now()
            scored: List[ScoredMemory] = []

            for mem in memories:
                single_resp = self.score_memory(mem, now=now_dt)
                if single_resp.get("status") != "success":
                    continue
                data = single_resp.get("data", {})
                scored_mem = ScoredMemory(
                    memory=mem,
                    score=_safe_float(data.get("score", 0.0), 0.0),
                    breakdown=dict(data.get("breakdown", {})),
                )
                scored.append(scored_mem)

            scored.sort(key=lambda x: x.score, reverse=True)

            result_data: List[Dict[str, Any]] = [
                {
                    "memory_id": sm.memory.id,
                    "score": sm.score,
                    "breakdown": sm.breakdown,
                }
                for sm in scored
            ]

            return make_response(
                status="success",
                data={"results": result_data},
                meta={
                    "module": self.module_name,
                    "version": MODULE_VERSION,
                    "count": len(result_data),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in score_batch")
            return make_response(
                status="error",
                data={},
                meta={"module": self.module_name},
                errors=[{"type": "UnexpectedError", "message": str(exc)}],
            )


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "RelevanceModuleError",
    "RelevanceValidationError",
    "RelevanceConfigError",
    "ConfigLoader",
    "SimpleRelevanceConfigLoader",
    "TelemetryRecord",
    "TelemetryCollector",
    "RelevanceConfig",
    "SourceWeightConfig",
    "MemoryForScoring",
    "ScoredMemory",
    "RelevanceScoringEngine",
    "make_response",
]

# End of File: memory/relevance_scoring.py (Phase‑1, Chunk‑1 + Chunk‑2)
