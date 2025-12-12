"""
AumCore_AI Memory Subsystem - Memory Maintenance Module
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/memory_maintenance.py

Description:
    यह module Phase‑1 में simple, rule-based, in-memory
    memory maintenance utilities provide करता है।

    Core Idea (Phase‑1):
        - Long-term / short-term memory stores के ऊपर
          housekeeping, pruning, compaction और basic checks चलाना।
        - Deterministic rules + config-driven thresholds से काम करना।
        - कोई background threads, scheduler या external DB नहीं।

    Future (Phase‑3+):
        - Persistent DB cleanup jobs
        - Priority-based decay और automatic summarization
        - Cross-store compaction + archiving
"""

# ============================================================
# ✅ Chunk 1: Imports (PEP8 + Strict Typing)
# ============================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple, Callable


# ============================================================
# ✅ Chunk 1: Module Metadata
# ============================================================

MODULE_NAME: str = "AumCoreAI.Memory.Maintenance"
MODULE_VERSION: str = "1.0.0"
MODULE_AUTHOR: str = "AumCore_AI"
MODULE_CREATED_ON: str = "2025-12-12"
MODULE_UPDATED_ON: str = "2025-12-12"


# ============================================================
# ✅ Chunk 1: Local Logger Setup
# ============================================================

def get_maintenance_logger(name: str = MODULE_NAME) -> logging.Logger:
    """
    Memory maintenance module के लिए local logger बनाता/return करता है।

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
            "[%(asctime)s] [MEMORY_MAINTENANCE] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger: logging.Logger = get_maintenance_logger()


# ============================================================
# ✅ Chunk 2: Custom Error Classes
# ============================================================

class MemoryMaintenanceError(Exception):
    """
    Base error type for memory maintenance module।
    """


class MemoryMaintenanceValidationError(MemoryMaintenanceError):
    """
    Input validation या state inconsistency के लिए error।
    """


class MemoryMaintenanceConfigError(MemoryMaintenanceError):
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
class SimpleMaintenanceConfigLoader:
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
            "maintenance": {
                "ltm_max_entries_soft": 1_800,
                "ltm_max_entries_hard": 2_000,
                "ltm_min_importance_to_keep": 0.3,
                "ltm_min_seen_count_to_keep": 1,
                "ltm_prune_batch_size": 100,
                "ltm_old_age_days": 365,
            },
            "telemetry": {
                "enabled": True,
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
        operation: Operation का नाम (e.g., "run_compaction")।
        duration_ms: Execution time milliseconds में।
        timestamp: Event का time (epoch seconds)।
        meta: Extra metadata (e.g., counts, thresholds)।
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

    enabled: bool = True
    records: List[TelemetryRecord] = field(default_factory=list)

    def record(self, operation: str, duration_ms: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        नया telemetry record add करता है।

        Args:
            operation: Operation का नाम।
            duration_ms: Duration in milliseconds।
            meta: Optional metadata।
        """
        if not self.enabled:
            return
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


def timed_operation(operation_name: str) -> Callable[..., Any]:
    """
    Decorator जो sync operation को measure करता है और telemetry में record करता है।

    Args:
        operation_name: Operation का नाम (string)।

    Returns:
        Wrapped function।
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
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


# ============================================================
# ✅ Chunk 2: Sanitization + Small Helpers
# ============================================================

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


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Any को safe float में convert करता है।

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


def _clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Float को range में clamp करता है।

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


# ============================================================
# ✅ Chunk 2: Target Store Protocol (to work with LongTermMemoryStore)
# ============================================================

class LongTermStoreProtocol(Protocol):
    """
    Long-term memory store की minimal interface (Phase‑1)।

    यह protocol ऐसे design किया गया है कि `LongTermMemoryStore`
    जैसा implementation आसानी से satisfy कर सके।
    """

    def list_entries(self) -> List[Any]:
        """
        सारे entries की list देता है।
        """

    def delete_entry(self, entry_id: str) -> bool:
        """
        Entry remove करता है और success flag return करता है।
        """

    def statistics(self) -> Dict[str, Any]:
        """
        Basic stats (total_entries, by_category, avg_importance, max_entries) देता है।
        """

    def is_entry_old(self, entry: Any) -> bool:
        """
        Default TTL से पुरानी entries के लिए boolean देता है।
        """


# ============================================================
# ✅ Chunk 2: Data Structures for Maintenance Config
# ============================================================

@dataclass
class MaintenanceConfig:
    """
    Memory maintenance के लिए Phase‑1 configuration।

    Fields:
        ltm_max_entries_soft: Soft limit, इससे ऊपर warn + prune।
        ltm_max_entries_hard: Hard cap, force prune अगर इससे ज्यादा हो।
        ltm_min_importance_to_keep: इससे नीचे वाले low-importance माने जाएंगे।
        ltm_min_seen_count_to_keep: बहुत कम seen_count वाले entries prune हो सकते हैं।
        ltm_prune_batch_size: एक pass में prune करने वाली max entries।
        ltm_old_age_days: इससे पुरानी entries "old" मानी जाएंगी reports के लिए।
    """

    ltm_max_entries_soft: int = 1_800
    ltm_max_entries_hard: int = 2_000
    ltm_min_importance_to_keep: float = 0.3
    ltm_min_seen_count_to_keep: int = 1
    ltm_prune_batch_size: int = 100
    ltm_old_age_days: int = 365

    def __post_init__(self) -> None:
        """
        Config validation।
        """
        if self.ltm_max_entries_soft <= 0 or self.ltm_max_entries_hard <= 0:
            raise MemoryMaintenanceConfigError("max_entries soft/hard positive होने चाहिए।")
        if self.ltm_max_entries_hard < self.ltm_max_entries_soft:
            raise MemoryMaintenanceConfigError("hard limit soft limit से बड़ा होना चाहिए।")
        if self.ltm_min_importance_to_keep < 0.0:
            raise MemoryMaintenanceConfigError("ltm_min_importance_to_keep negative नहीं हो सकता।")
        if self.ltm_min_seen_count_to_keep < 0:
            raise MemoryMaintenanceConfigError("ltm_min_seen_count_to_keep negative नहीं हो सकता।")
        if self.ltm_prune_batch_size <= 0:
            raise MemoryMaintenanceConfigError("ltm_prune_batch_size positive होना चाहिए।")
        if self.ltm_old_age_days <= 0:
            raise MemoryMaintenanceConfigError("ltm_old_age_days positive होना चाहिए।")


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
        meta: Metadata (e.g., timings, counters)।
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
# ✅ Core Class: Memory Maintenance Engine (Phase‑1)
# ============================================================

class MemoryMaintenanceEngine:
    """
    Phase‑1 compatible memory maintenance engine।

    Responsibilities:
        - Long-term memory store के लिए:
            - size monitoring
            - low-importance prune
            - old entries report
        - Deterministic, config-driven, in-memory operations
    """

    def __init__(
        self,
        store: LongTermStoreProtocol,
        config_loader: Optional[ConfigLoader] = None,
        module_name: str = MODULE_NAME,
    ) -> None:
        """
        Engine को target store और config loader के साथ initialize करता है।

        Args:
            store: LongTermStoreProtocol compatible store instance।
            config_loader: Config loader (optional, default internal stub)।
            module_name: Module name for config।
        """
        if store is None:
            raise MemoryMaintenanceValidationError("store None नहीं होना चाहिए।")

        self.store: LongTermStoreProtocol = store
        self.module_name: str = module_name
        self._config_loader: ConfigLoader = config_loader or SimpleMaintenanceConfigLoader()
        raw_config: Dict[str, Any] = self._config_loader.load(module_name)

        maint_cfg = raw_config.get("maintenance", {})
        telemetry_cfg = raw_config.get("telemetry", {})

        self.config: MaintenanceConfig = MaintenanceConfig(
            ltm_max_entries_soft=_safe_int(maint_cfg.get("ltm_max_entries_soft", 1_800), 1_800),
            ltm_max_entries_hard=_safe_int(maint_cfg.get("ltm_max_entries_hard", 2_000), 2_000),
            ltm_min_importance_to_keep=_safe_float(
                maint_cfg.get("ltm_min_importance_to_keep", 0.3),
                0.3,
            ),
            ltm_min_seen_count_to_keep=_safe_int(
                maint_cfg.get("ltm_min_seen_count_to_keep", 1),
                1,
            ),
            ltm_prune_batch_size=_safe_int(maint_cfg.get("ltm_prune_batch_size", 100), 100),
            ltm_old_age_days=_safe_int(maint_cfg.get("ltm_old_age_days", 365), 365),
        )

        telemetry.enabled = bool(telemetry_cfg.get("enabled", True))

        logger.debug("MemoryMaintenanceEngine initialized with config: %r", self.config)

    # --------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------

    def _compute_entry_age_days(self, entry: Any, now: Optional[datetime] = None) -> float:
        """
        Entry के creation time से age days में compute करता है।

        Args:
            entry: Store-specific entry object।
            now: Optional current time।

        Returns:
            float: Age in days।
        """
        created_at: datetime = getattr(entry, "created_at", datetime.now())
        now_dt = now or datetime.now()
        delta: timedelta = now_dt - created_at
        return max(delta.total_seconds() / 86400.0, 0.0)

    # --------------------------------------------------------
    # Public API: Health Snapshot (Sync)
    # --------------------------------------------------------

    @timed_operation("maintenance_health_snapshot")
    def health_snapshot(self) -> Dict[str, Any]:
        """
        Long-term memory store के बारे में basic health snapshot देता है।

        Returns:
            dict: Standard response with stats + threshold comparison।
        """
        try:
            stats = self.store.statistics()
            total = _safe_int(stats.get("total_entries", 0), 0)

            soft = self.config.ltm_max_entries_soft
            hard = self.config.ltm_max_entries_hard

            status_level = "ok"
            if total > soft:
                status_level = "warning"
            if total > hard:
                status_level = "critical"

            logger.debug(
                "Health snapshot computed: total=%d soft=%d hard=%d level=%s",
                total,
                soft,
                hard,
                status_level,
            )

            return make_response(
                status="success",
                data={
                    "total_entries": total,
                    "limits": {
                        "soft": soft,
                        "hard": hard,
                    },
                    "status_level": status_level,
                    "raw_stats": stats,
                },
                meta={
                    "module": self.module_name,
                    "version": MODULE_VERSION,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in health_snapshot")
            return make_response(
                status="error",
                data={},
                meta={"module": self.module_name},
                errors=[{"type": "UnexpectedError", "message": str(exc)}],
            )

    # --------------------------------------------------------
    # Public API: Find Candidates for Prune (Sync)
    # --------------------------------------------------------

    @timed_operation("maintenance_list_prune_candidates")
    def list_prune_candidates(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Low-importance prune candidates की list देता है (delete नहीं करता)।

        Criteria (Phase‑1, deterministic):
            - importance < ltm_min_importance_to_keep
            - OR seen_count < ltm_min_seen_count_to_keep
        Sorted by:
            - importance asc
            - seen_count asc
            - created_at asc
        """
        try:
            entries = self.store.list_entries()
            candidates: List[Dict[str, Any]] = []

            importance_thr = self.config.ltm_min_importance_to_keep
            seen_thr = self.config.ltm_min_seen_count_to_keep

            for e in entries:
                imp = float(getattr(e, "importance", 0.0))
                seen = int(getattr(e, "seen_count", 0))
                created_at: datetime = getattr(e, "created_at", datetime.now())
                if imp >= importance_thr and seen >= seen_thr:
                    continue

                candidates.append(
                    {
                        "id": getattr(e, "id", ""),
                        "importance": imp,
                        "seen_count": seen,
                        "created_at": created_at,
                    }
                )

            candidates.sort(
                key=lambda c: (c["importance"], c["seen_count"], c["created_at"]),
            )

            if limit is not None and limit > 0:
                candidates = candidates[:limit]

            serialized: List[Dict[str, Any]] = [
                {
                    "id": c["id"],
                    "importance": c["importance"],
                    "seen_count": c["seen_count"],
                    "created_at": c["created_at"].isoformat(),
                }
                for c in candidates
            ]

            return make_response(
                status="success",
                data={"candidates": serialized},
                meta={
                    "module": self.module_name,
                    "version": MODULE_VERSION,
                    "count": len(serialized),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in list_prune_candidates")
            return make_response(
                status="error",
                data={},
                meta={"module": self.module_name},
                errors=[{"type": "UnexpectedError", "message": str(exc)}],
            )

    # --------------------------------------------------------
    # Public API: Prune In-place (Sync)
    # --------------------------------------------------------

    @timed_operation("maintenance_prune_in_place")
    def prune_in_place(self) -> Dict[str, Any]:
        """
        Config rules के आधार पर long-term store को in-place prune करता है।

        Behavior:
            - पहले health snapshot देखता है।
            - अगर total_entries <= soft limit => कोई prune नहीं।
            - वरना low-importance prune candidates select करता है।
            - Configured batch size तक delete करता है।
        """
        try:
            health = self.health_snapshot()
            if health.get("status") != "success":
                return health

            data = health.get("data", {})
            total = _safe_int(data.get("total_entries", 0), 0)
            soft = self.config.ltm_max_entries_soft

            if total <= soft:
                logger.debug("Prune skipped: total_entries <= soft limit (%d <= %d)", total, soft)
                return make_response(
                    status="success",
                    data={"pruned": [], "skipped": True},
                    meta={
                        "module": self.module_name,
                        "reason": "below_soft_limit",
                    },
                )

            cand_resp = self.list_prune_candidates(limit=self.config.ltm_prune_batch_size)
            if cand_resp.get("status") != "success":
                return cand_resp

            candidates = cand_resp.get("data", {}).get("candidates", [])
            pruned_ids: List[str] = []

            for c in candidates:
                entry_id = str(c.get("id", "")).strip()
                if not entry_id:
                    continue
                ok = self.store.delete_entry(entry_id)
                if ok:
                    pruned_ids.append(entry_id)

            logger.info(
                "Prune completed: requested=%d deleted=%d",
                len(candidates),
                len(pruned_ids),
            )

            return make_response(
                status="success",
                data={"pruned": pruned_ids, "skipped": False},
                meta={
                    "module": self.module_name,
                    "version": MODULE_VERSION,
                    "requested": len(candidates),
                    "deleted": len(pruned_ids),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in prune_in_place")
            return make_response(
                status="error",
                data={},
                meta={"module": self.module_name},
                errors=[{"type": "UnexpectedError", "message": str(exc)}],
            )

    # --------------------------------------------------------
    # Public API: List Old Entries (Informational)
    # --------------------------------------------------------

    @timed_operation("maintenance_list_old_entries")
    def list_old_entries(self) -> Dict[str, Any]:
        """
        Configured old_age_days से पुरानी entries की list देता है (delete नहीं करता)।

        Returns:
            dict: Standard response with old entries summary।
        """
        try:
            entries = self.store.list_entries()
            now = datetime.now()
            threshold_days = self.config.ltm_old_age_days

            old_entries: List[Dict[str, Any]] = []

            for e in entries:
                age_days = self._compute_entry_age_days(e, now=now)
                if age_days < threshold_days:
                    continue
                old_entries.append(
                    {
                        "id": getattr(e, "id", ""),
                        "age_days": age_days,
                        "created_at": getattr(e, "created_at", now).isoformat(),
                        "importance": float(getattr(e, "importance", 0.0)),
                        "seen_count": int(getattr(e, "seen_count", 0)),
                    }
                )

            old_entries.sort(key=lambda x: x["age_days"], reverse=True)

            return make_response(
                status="success",
                data={"old_entries": old_entries},
                meta={
                    "module": self.module_name,
                    "version": MODULE_VERSION,
                    "count": len(old_entries),
                    "threshold_days": threshold_days,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in list_old_entries")
            return make_response(
                status="error",
                data={},
                meta={"module": self.module_name},
                errors=[{"type": "UnexpectedError", "message": str(exc)}],
            )


# ============================================================
# Public Factory Helper
# ============================================================

def create_default_memory_maintenance_engine(
    store: LongTermStoreProtocol,
) -> MemoryMaintenanceEngine:
    """
    Default config के साथ MemoryMaintenanceEngine instance create करता है।

    Args:
        store: LongTermStoreProtocol compatible store।

    Returns:
        MemoryMaintenanceEngine: Initialized engine।
    """
    return MemoryMaintenanceEngine(store=store)


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "MemoryMaintenanceError",
    "MemoryMaintenanceValidationError",
    "MemoryMaintenanceConfigError",
    "ConfigLoader",
    "SimpleMaintenanceConfigLoader",
    "TelemetryRecord",
    "TelemetryCollector",
    "MaintenanceConfig",
    "MemoryMaintenanceEngine",
    "create_default_memory_maintenance_engine",
    "make_response",
]

# End of File: memory/memory_maintenance.py (Phase‑1, Chunk‑1 + Chunk‑2)
