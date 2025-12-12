"""
AumCore_AI Memory Subsystem - FAISS Adapter (Phase‑1 Stub)
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/faiss_adapter.py

Description:
    यह module Phase‑1 में FAISS-like interface provide करता है,
    लेकिन असली FAISS index use नहीं करता।

    क्यों?
        Phase‑1 में:
            - कोई embeddings नहीं
            - कोई vector store नहीं
            - कोई ANN search नहीं

    इसलिए यह module सिर्फ:
        - deterministic, rule-based, in-memory index रखता है
        - API को FAISS जैसा बनाता है (add, search, delete)
        - future Phase‑3 integration के लिए placeholder नहीं,
          बल्कि fully functional Phase‑1 logic देता है।

    Future (Phase‑3+):
        - Real FAISS index
        - GPU acceleration
        - HNSW / IVF / PQ indexing
"""

# ============================================================
# ✅ Chunk 1: Imports (PEP8 + Strict Typing)
# ============================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple


# ============================================================
# ✅ Chunk 1: Module Metadata
# ============================================================

MODULE_NAME: str = "AumCoreAI.Memory.FAISSAdapter"
MODULE_VERSION: str = "1.0.0"
MODULE_AUTHOR: str = "AumCore_AI"
MODULE_CREATED_ON: str = "2025-12-12"
MODULE_UPDATED_ON: str = "2025-12-12"


# ============================================================
# ✅ Chunk 1: Local Logger Setup
# ============================================================

def get_faiss_logger(name: str = MODULE_NAME) -> logging.Logger:
    """
    FAISS adapter module के लिए local logger बनाता है।
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [FAISS_ADAPTER] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger: logging.Logger = get_faiss_logger()


# ============================================================
# ✅ Chunk 2: Custom Error Classes
# ============================================================

class FAISSAdapterError(Exception):
    """Base error for FAISS adapter."""


class FAISSAdapterValidationError(FAISSAdapterError):
    """Invalid input / state error."""


class FAISSAdapterConfigError(FAISSAdapterError):
    """Config missing or invalid."""


# ============================================================
# ✅ Chunk 2: Config Loader Protocol
# ============================================================

class ConfigLoader(Protocol):
    """
    Config loader protocol (Phase‑1 stub).
    """

    def load(self, module_name: str) -> Dict[str, Any]:
        """
        Module-specific config load करता है।
        """


@dataclass
class SimpleFAISSConfigLoader:
    """
    Simple in-memory config loader (Phase‑1 stub).
    """

    global_config: Dict[str, Any] = field(default_factory=dict)

    def load(self, module_name: str) -> Dict[str, Any]:
        """
        Safe fallback config return करता है।
        """
        config = self.global_config.get(module_name, {})
        if config:
            return config

        return {
            "logging": {"level": "DEBUG"},
            "faiss": {
                "dim": 128,
                "max_entries": 5000,
                "distance_metric": "l2",  # Phase‑1: deterministic fake metric
            },
        }


# ============================================================
# ✅ Chunk 2: Telemetry + Timers
# ============================================================

@dataclass
class TelemetryRecord:
    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryCollector:
    enabled: bool = True
    records: List[TelemetryRecord] = field(default_factory=list)

    def record(self, op: str, dur: float, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self.records.append(
            TelemetryRecord(operation=op, duration_ms=dur, meta=dict(meta or {}))
        )

    def snapshot(self, limit: int = 50) -> List[Dict[str, Any]]:
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


telemetry = TelemetryCollector()


def timed(op: str):
    """
    Decorator for timing sync operations.
    """

    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dur = (time.perf_counter() - start) * 1000
                telemetry.record(op, dur)
                logger.debug("Timed '%s' = %.3f ms", op, dur)

        return wrapper

    return decorator


# ============================================================
# ✅ Chunk 2: Sanitization Helpers
# ============================================================

def _safe_list(v: Any) -> List[float]:
    if not isinstance(v, (list, tuple)):
        return []
    return [float(x) for x in v]


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


# ============================================================
# ✅ Data Structures (Phase‑1 Fake Vector Index)
# ============================================================

@dataclass
class VectorEntry:
    """
    Phase‑1 FAISS-like entry.
    """

    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str):
            raise FAISSAdapterValidationError("VectorEntry.id must be non-empty string.")
        if not isinstance(self.vector, list):
            raise FAISSAdapterValidationError("VectorEntry.vector must be list[float].")


# ============================================================
# ✅ Core Class: FAISSAdapter (Phase‑1)
# ============================================================

class FAISSAdapter:
    """
    Phase‑1 FAISS adapter (no real FAISS).

    Features:
        - In-memory vector index
        - Deterministic fake distance metric
        - add / delete / search API
        - Config-driven dimension + max entries
    """

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        module_name: str = MODULE_NAME,
    ) -> None:
        self.module_name = module_name
        self._config_loader = config_loader or SimpleFAISSConfigLoader()
        raw = self._config_loader.load(module_name)

        faiss_cfg = raw.get("faiss", {})

        self.dim: int = _safe_int(faiss_cfg.get("dim", 128), 128)
        self.max_entries: int = _safe_int(faiss_cfg.get("max_entries", 5000), 5000)
        self.metric: str = str(faiss_cfg.get("distance_metric", "l2")).lower()

        self._index: Dict[str, VectorEntry] = {}

        logger.debug(
            "FAISSAdapter initialized: dim=%d max_entries=%d metric=%s",
            self.dim,
            self.max_entries,
            self.metric,
        )

    # --------------------------------------------------------
    # Fake deterministic distance metric
    # --------------------------------------------------------

    def _distance(self, a: List[float], b: List[float]) -> float:
        """
        Phase‑1 deterministic fake metric:
            - If metric == "l2": sum(|a[i] - b[i]|)
            - If metric == "dot": sum(a[i] * b[i])
        """
        if len(a) != len(b):
            return 999999.0

        if self.metric == "dot":
            return -sum(x * y for x, y in zip(a, b))

        # default: L1 distance (deterministic, cheap)
        return sum(abs(x - y) for x, y in zip(a, b))

    # --------------------------------------------------------
    # Public API: Add
    # --------------------------------------------------------

    @timed("faiss_add")
    def add(self, entry_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Vector add करता है (Phase‑1 in-memory).
        """
        try:
            if len(vector) != self.dim:
                raise FAISSAdapterValidationError(f"Vector dim mismatch: expected {self.dim}")

            if len(self._index) >= self.max_entries:
                raise FAISSAdapterError("FAISS index full (Phase‑1 limit).")

            entry = VectorEntry(
                id=str(entry_id).strip(),
                vector=_safe_list(vector),
                metadata=dict(metadata or {}),
            )
            self._index[entry.id] = entry

            return {
                "status": "success",
                "data": {"id": entry.id},
                "meta": {"count": len(self._index)},
                "errors": [],
            }
        except Exception as exc:
            logger.exception("Error in FAISSAdapter.add")
            return {
                "status": "error",
                "data": {},
                "meta": {},
                "errors": [{"type": "AddError", "message": str(exc)}],
            }

    # --------------------------------------------------------
    # Public API: Delete
    # --------------------------------------------------------

    @timed("faiss_delete")
    def delete(self, entry_id: str) -> Dict[str, Any]:
        """
        Vector delete करता है।
        """
        try:
            if entry_id in self._index:
                del self._index[entry_id]
                return {
                    "status": "success",
                    "data": {"deleted": entry_id},
                    "meta": {},
                    "errors": [],
                }
            return {
                "status": "success",
                "data": {"deleted": None},
                "meta": {},
                "errors": [],
            }
        except Exception as exc:
            logger.exception("Error in FAISSAdapter.delete")
            return {
                "status": "error",
                "data": {},
                "meta": {},
                "errors": [{"type": "DeleteError", "message": str(exc)}],
            }

    # --------------------------------------------------------
    # Public API: Search
    # --------------------------------------------------------

    @timed("faiss_search")
    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        """
        Deterministic fake nearest-neighbor search.
        """
        try:
            if len(query_vector) != self.dim:
                raise FAISSAdapterValidationError(f"Query dim mismatch: expected {self.dim}")

            q = _safe_list(query_vector)

            scored: List[Tuple[str, float]] = []
            for eid, entry in self._index.items():
                d = self._distance(q, entry.vector)
                scored.append((eid, d))

            scored.sort(key=lambda x: x[1])
            scored = scored[:max(1, top_k)]

            results = [
                {"id": eid, "distance": dist}
                for eid, dist in scored
            ]

            return {
                "status": "success",
                "data": {"results": results},
                "meta": {"count": len(results)},
                "errors": [],
            }
        except Exception as exc:
            logger.exception("Error in FAISSAdapter.search")
            return {
                "status": "error",
                "data": {},
                "meta": {},
                "errors": [{"type": "SearchError", "message": str(exc)}],
            }


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    "FAISSAdapterError",
    "FAISSAdapterValidationError",
    "FAISSAdapterConfigError",
    "ConfigLoader",
    "SimpleFAISSConfigLoader",
    "TelemetryRecord",
    "TelemetryCollector",
    "VectorEntry",
    "FAISSAdapter",
]

# End of File: memory/faiss_adapter.py (Phase‑1, Chunk‑1 + Chunk‑2)
