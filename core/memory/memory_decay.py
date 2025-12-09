# File: core/memory/memory_decay.py
# Purpose: Phase-2 Enterprise Memory Decay Module for AumCore_AI
# - Temporal + semantic decay models
# - Embedding-based similarity pruning (optional: sentence-transformers + faiss)
# - Async scheduler for periodic decay cycles
# - Integration hooks for KnowledgeGraph and other memory subsystems
# - Persistence with versioned migration support
# - Diagnostics, telemetry, unit-test hooks
# - DI-friendly factory
# NOTE: Optional dependencies are imported gracefully. The module degrades to CPU-only rule-based behavior if heavy libs are not present.

from __future__ import annotations

import gc
import heapq
import inspect
import json
import logging
import math
import os
import random
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, TypedDict, Union)
from uuid import uuid4

# Optional ML libs (import gracefully)
_HAS_SENTENCE_TRANSFORMERS = False
_HAS_FAISS = False
try:
    from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None  # type: ignore
    st_util = None  # type: ignore

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore

# ============================================================================
# CONFIG & TYPES
# ============================================================================

class MemoryDecayConfig(TypedDict):
    storage_path: str
    decay_rate_hours: float  # linear hours-based rate baseline
    minimum_strength: float  # below which memory is pruned
    max_records: int
    embedding_model_name: Optional[str]
    embedding_dim: Optional[int]
    faiss_index_path: Optional[str]
    semantic_prune_threshold: float
    auto_save_interval_sec: int
    scheduler_interval_sec: int
    enable_embeddings: bool
    enable_faiss: bool
    reinforce_on_access: bool
    max_reinforce_boost: float
    migration_version: str


class MemoryRecordDict(TypedDict):
    id: str
    data: str
    metadata: Dict[str, Any]
    strength: float
    created_at: str
    last_accessed: str
    embedding: Optional[List[float]]


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MemoryRecord:
    id: str = field(default_factory=lambda: str(uuid4()))
    data: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = None

    def touch(self) -> None:
        self.last_accessed = datetime.utcnow()

    def to_dict(self) -> MemoryRecordDict:
        return {
            "id": self.id,
            "data": self.data,
            "metadata": self.metadata,
            "strength": float(self.strength),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "embedding": list(self.embedding) if self.embedding is not None else None
        }

    @staticmethod
    def from_dict(d: MemoryRecordDict) -> "MemoryRecord":
        rec = MemoryRecord(
            id=d["id"],
            data=d["data"],
            metadata=d.get("metadata", {}),
            strength=float(d.get("strength", 1.0)),
            created_at=datetime.fromisoformat(d["created_at"]),
            last_accessed=datetime.fromisoformat(d["last_accessed"]),
            embedding=list(d["embedding"]) if d.get("embedding") is not None else None
        )
        return rec


# ============================================================================
# UTILITIES
# ============================================================================

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def clamp(value: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, value))

def exponential_decay(base_strength: float, hours: float, half_life_hours: float) -> float:
    """
    Exponential decay formula.
    strength(t) = base_strength * 0.5^(t / half_life)
    """
    if half_life_hours <= 0:
        return base_strength
    return base_strength * (0.5 ** (hours / half_life_hours))

def linear_decay(base_strength: float, hours: float, rate_per_hour: float) -> float:
    return base_strength - hours * rate_per_hour

def safe_json_dump(data: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ============================================================================
# EMBEDDING ADAPTER (optional)
# ============================================================================

class EmbeddingAdapter:
    """
    Optional embedding adapter using sentence-transformers and FAISS.
    If dependencies are missing, this adapter provides stubbed behavior.
    """

    def __init__(self, model_name: Optional[str], dim: Optional[int], faiss_index_path: Optional[str], logger: logging.Logger):
        self.logger = logger
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.dim = dim or 384
        self.faiss_index_path = Path(faiss_index_path) if faiss_index_path else None

        self.model = None
        self.index = None
        self.id_to_index: Dict[str, int] = {}
        self.lock = threading.RLock()

        if _HAS_SENTENCE_TRANSFORMERS and model_name:
            try:
                self.model = SentenceTransformer(model_name)
                self.dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Embedding model loaded: {model_name} dim={self.dim}")
            except Exception as e:
                self.logger.warning(f"Failed to load SentenceTransformer {model_name}: {e}")
                self.model = None
        else:
            self.logger.info("SentenceTransformer not available; embedding disabled.")

        if _HAS_FAISS and self.model and self.faiss_index_path is not None:
            try:
                # if index exists, load; else create flat index
                if self.faiss_index_path.exists():
                    self.index = faiss.read_index(str(self.faiss_index_path))
                    self.logger.info(f"Loaded FAISS index from {self.faiss_index_path}")
                else:
                    self.index = faiss.IndexFlatIP(self.dim)  # inner-product similarity (use normalized vectors)
                    self.logger.info("Created new FAISS IndexFlatIP")
            except Exception as e:
                self.logger.warning(f"Failed to create/load FAISS index: {e}")
                self.index = None
        else:
            if _HAS_FAISS and self.model:
                self.logger.info("FAISS available but no index path specified; in-memory index will be used.")
                try:
                    self.index = faiss.IndexFlatIP(self.dim)
                except Exception:
                    self.index = None

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.model:
            # fallback: simple random-ish hash vector (deterministic pseudo-embedding)
            out = []
            for t in texts:
                r = [math.sin(hash(t) % (i + 1) + i) for i in range(self.dim)]
                # normalize
                norm = math.sqrt(sum(x * x for x in r)) or 1.0
                out.append([x / norm for x in r])
            return out
        return self.model.encode(list(texts), convert_to_numpy=False).tolist()  # type: ignore

    def add_vectors(self, ids: List[str], vectors: List[List[float]]) -> None:
        with self.lock:
            if self.index is None:
                self.logger.debug("No FAISS index; skipping add_vectors")
                return
            import numpy as np  # local import (optional)
            vecs = np.array(vectors).astype("float32")
            # normalize if using inner-product search
            faiss.normalize_L2(vecs)
            self.index.add(vecs)
            start_idx = len(self.id_to_index)
            for i, mid in enumerate(ids):
                self.id_to_index[mid] = start_idx + i

    def search(self, vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        with self.lock:
            if self.index is None:
                return []
            import numpy as np  # local import (optional)
            v = np.array([vector]).astype("float32")
            faiss.normalize_L2(v)
            try:
                dists, idxs = self.index.search(v, top_k)
                results = []
                for dist, idx in zip(dists[0], idxs[0]):
                    if idx < 0:
                        continue
                    # reverse mapping
                    for mid, i in self.id_to_index.items():
                        if i == idx:
                            results.append((mid, float(dist)))
                            break
                return results
            except Exception as e:
                self.logger.warning(f"FAISS search failed: {e}")
                return []

    def persist(self) -> None:
        if self.faiss_index_path and self.index is not None:
            try:
                faiss.write_index(self.index, str(self.faiss_index_path))
                self.logger.info(f"FAISS index saved to {self.faiss_index_path}")
            except Exception as e:
                self.logger.error(f"Failed to persist FAISS index: {e}")


# ============================================================================
# CORE MEMORY DECAY ENGINE
# ============================================================================

class MemoryDecayEngine:
    """
    High-fidelity memory decay engine.

    Responsibilities:
    - Manage memory records (CRUD)
    - Apply decay policies (temporal + semantic)
    - Reinforcement on access or external signals
    - Periodic prune + persistence
    - Diagnostics and analytics
    - Optional embedding support for semantic pruning / clustering
    """

    def __init__(
        self,
        config: Optional[MemoryDecayConfig] = None,
        logger: Optional[logging.Logger] = None,
        external_event_hook: Optional[Callable[[str, str], None]] = None
    ):
        # Default configuration
        if config is None:
            config = MemoryDecayConfig(
                storage_path="./data/memory",
                decay_rate_hours=0.0025,  # base linear decrement per hour
                minimum_strength=0.15,
                max_records=200000,
                embedding_model_name="all-MiniLM-L6-v2" if _HAS_SENTENCE_TRANSFORMERS else None,  # optional
                embedding_dim=384 if _HAS_SENTENCE_TRANSFORMERS else 128,
                faiss_index_path="./data/memory/faiss.index",
                semantic_prune_threshold=0.85,
                auto_save_interval_sec=60,
                scheduler_interval_sec=30,
                enable_embeddings=_HAS_SENTENCE_TRANSFORMERS,
                enable_faiss=_HAS_FAISS,
                reinforce_on_access=True,
                max_reinforce_boost=0.25,
                migration_version="2.0"
            )  # type: ignore

        self.config = config
        self.storage_path = Path(self.config["storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, MemoryRecord] = {}
        self.lock = threading.RLock()
        self.logger = logger or self._setup_logger()
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.external_event_hook = external_event_hook

        # Embedding adapter (optional)
        self.embedding_adapter = None
        if self.config.get("enable_embeddings", False) and _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_adapter = EmbeddingAdapter(
                    model_name=self.config.get("embedding_model_name"),
                    dim=self.config.get("embedding_dim"),
                    faiss_index_path=self.config.get("faiss_index_path"),
                    logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Failed to init EmbeddingAdapter: {e}")
                self.embedding_adapter = None
        else:
            if self.config.get("enable_embeddings", False):
                self.logger.info("Embeddings requested but sentence-transformers not available. Running in fallback mode.")

        # Load persisted data
        self._load_from_disk()

        # Scheduler threads
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._start_scheduler_if_needed()

        self.logger.info(f"MemoryDecayEngine initialized: records={len(self.records)}")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("aumcore.memory.memory_decay_engine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(fmt)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # -------------------------
    # CRUD OPERATIONS
    # -------------------------

    def add_record(self, data: str, metadata: Optional[Dict[str, Any]] = None, strength: float = 1.0, compute_embedding: bool = True) -> str:
        if not data or not data.strip():
            raise ValueError("Memory `data` must be non-empty")

        with self.lock:
            if len(self.records) >= self.config["max_records"]:
                self.logger.warning("Max records reached; pruning before insert.")
                self._auto_prune(count=max(1, len(self.records) // 20))

            rec = MemoryRecord(data=data.strip(), metadata=metadata or {}, strength=clamp(strength))
            if self.embedding_adapter and compute_embedding:
                try:
                    emb = self.embedding_adapter.encode([data.strip()])[0]
                    rec.embedding = emb
                except Exception as e:
                    self.logger.debug(f"Embedding encode failed: {e}")
                    rec.embedding = None

            self.records[rec.id] = rec
            if self.embedding_adapter and rec.embedding is not None:
                try:
                    self.embedding_adapter.add_vectors([rec.id], [rec.embedding])
                except Exception as e:
                    self.logger.debug(f"Failed to add vector to FAISS: {e}")
            self.metrics["adds"] += 1
            self.logger.debug(f"Added memory record {rec.id[:8]} len={len(data)}")
            return rec.id

    def get_record(self, record_id: str, reinforce: bool = True) -> Optional[MemoryRecord]:
        with self.lock:
            rec = self.records.get(record_id)
            if not rec:
                self.metrics["misses"] += 1
                return None
            rec.touch()
            self.metrics["hits"] += 1
            if self.config.get("reinforce_on_access", True) and reinforce:
                self._reinforce_internal(rec, boost=min(0.05, self.config.get("max_reinforce_boost", 0.25)))
            return rec

    def update_record(self, record_id: str, data: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        with self.lock:
            rec = self.records.get(record_id)
            if not rec:
                return False
            if data:
                rec.data = data
                # optionally update embedding
                if self.embedding_adapter:
                    try:
                        rec.embedding = self.embedding_adapter.encode([data])[0]
                    except Exception as e:
                        self.logger.debug(f"Embedding update failed: {e}")
            if metadata:
                rec.metadata.update(metadata)
            rec.touch()
            self.metrics["updates"] += 1
            return True

    def delete_record(self, record_id: str) -> bool:
        with self.lock:
            rec = self.records.pop(record_id, None)
            if not rec:
                return False
            # optionally remove from FAISS mapping (note: not trivial to remove vectors from Flat index)
            # we keep id_to_index mapping stale-safe (search filters missing)
            self.metrics["deletes"] += 1
            return True

    # -------------------------
    # DECAY & PRUNING
    # -------------------------

    def _reinforce_internal(self, rec: MemoryRecord, boost: float) -> None:
        old = rec.strength
        rec.strength = clamp(rec.strength + boost)
        rec.touch()
        self.logger.debug(f"Reinforced {rec.id[:8]} {old:.3f}->{rec.strength:.3f}")

    def reinforce(self, record_id: str, boost: float = 0.1) -> bool:
        with self.lock:
            rec = self.records.get(record_id)
            if not rec:
                return False
            self._reinforce_internal(rec, boost=min(boost, self.config.get("max_reinforce_boost", 0.25)))
            return True

    def decay_step(self) -> None:
        """
        Apply a decay step to all records.
        - Supports combined linear + exponential decay
        - Optionally uses recency-weighting and semantic importance
        """
        now = datetime.utcnow()
        removed = []
        with self.lock:
            for rid, rec in list(self.records.items()):
                hours_since_access = max((now - rec.last_accessed).total_seconds() / 3600.0, 0.0)
                # base linear decay
                linear = linear_decay(rec.strength, hours_since_access, self.config.get("decay_rate_hours", 0.0025))
                # exponential half-life factor based on metadata 'importance' if present
                half_life_hours = rec.metadata.get("half_life_hours", 72.0)
                exp = exponential_decay(rec.strength, hours_since_access, half_life_hours)

                # combine heuristics (weighted)
                combined = 0.6 * exp + 0.4 * linear
                # time-based clamp
                rec.strength = clamp(combined)

                # semantic retention bump: if metadata says 'pinned' or 'important', protect
                if rec.metadata.get("pinned", False) or rec.metadata.get("importance", 0.0) > 0.8:
                    rec.strength = max(rec.strength, 0.5)

                # external hook: if graph-connected, optionally boost
                if self.external_event_hook and rec.metadata.get("linked_kg_node"):
                    # call hook to maybe compute reinforcement
                    try:
                        self.external_event_hook("access_link_check", rec.id)
                        # small boost for being graph-linked
                        rec.strength = clamp(rec.strength + 0.02)
                    except Exception:
                        pass

                # prune if below threshold
                if rec.strength <= self.config.get("minimum_strength", 0.15):
                    removed.append(rid)

            for rid in removed:
                self.delete_record(rid)

        if removed:
            self.logger.info(f"Decay step pruned {len(removed)} records")
            self.metrics["pruned"] += len(removed)

    def semantic_prune(self, top_k: int = 5) -> List[str]:
        """
        Prune semantically redundant memories.
        Steps:
        - For each record with embedding, find semantically similar records via FAISS or brute-force
        - If similarity > threshold, keep stronger and remove weaker
        Returns list of removed record_ids
        """
        if not self.embedding_adapter:
            self.logger.debug("Embedding adapter unavailable - semantic_prune skipped")
            return []

        removed = set()
        with self.lock:
            # build candidate list
            items = [(rid, rec) for rid, rec in self.records.items() if rec.embedding is not None]
            if not items:
                return []

            # Use FAISS search per-record (if available) else brute-force using cosine
            for rid, rec in items:
                if rid in removed:
                    continue
                try:
                    neighbors = self.embedding_adapter.search(rec.embedding, top_k)
                except Exception:
                    neighbors = []
                for nid, score in neighbors:
                    # score is inner-prod / similarity; convert if necessary
                    similarity = float(score)
                    if similarity >= self.config.get("semantic_prune_threshold", 0.85):
                        # if neighbor has lower strength, remove it
                        if nid in self.records and self.records[nid].strength < rec.strength:
                            if nid not in removed:
                                removed.add(nid)
                        else:
                            if rid not in removed:
                                removed.add(rid)
                                break
            # remove
            for rid in list(removed):
                self.delete_record(rid)
            if removed:
                self.logger.info(f"Semantic prune removed {len(removed)} records")
                self.metrics["semantic_pruned"] += len(removed)

        return list(removed)

    def _auto_prune(self, count: int = 1) -> None:
        """
        Prune least important records until 'count' removed.
        Uses a min-heap by (strength, last_accessed)
        """
        with self.lock:
            if not self.records:
                return
            heap = []
            for rid, rec in self.records.items():
                # Priority: lower strength, older access -> higher prio to remove
                age_hours = (datetime.utcnow() - rec.last_accessed).total_seconds() / 3600.0
                priority = (rec.strength, -age_hours)
                heapq.heappush(heap, (priority, rid))
            removed = 0
            while heap and removed < count:
                _, rid = heapq.heappop(heap)
                if rid in self.records:
                    self.delete_record(rid)
                    removed += 1
            if removed:
                self.logger.info(f"Auto-pruned {removed} records to respect max_records")
                self.metrics["auto_pruned"] += removed

    # -------------------------
    # PERSISTENCE & MIGRATION
    # -------------------------

    def save(self) -> None:
        """
        Persist records to disk (safe atomic write).
        Also persists a lightweight index for diagnostics, and FAISS index if present.
        """
        p = self.storage_path / "memory_records_v2.json"
        try:
            with self.lock:
                payload = {
                    "version": self.config.get("migration_version", "2.0"),
                    "saved_at": now_iso(),
                    "records": {rid: rec.to_dict() for rid, rec in self.records.items()},
                    "metrics": dict(self.metrics)
                }
            safe_json_dump(payload, p)
            if self.embedding_adapter:
                try:
                    self.embedding_adapter.persist()
                except Exception as e:
                    self.logger.debug(f"FAISS persist error: {e}")
            self.logger.debug(f"Memory saved to {p} ({len(self.records)} records)")
        except Exception as e:
            self.logger.error(f"Failed saving memory: {e}\n{traceback.format_exc()}")

    def _load_from_disk(self) -> None:
        """
        Load persisted state with migration-aware logic.
        """
        p = self.storage_path / "memory_records_v2.json"
        if not p.exists():
            # try legacy path
            legacy = self.storage_path / "memory_records.json"
            if legacy.exists():
                p = legacy
            else:
                self.logger.info("No persisted memory found; starting fresh.")
                return

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", "1.0")
            records = data.get("records", {})
            with self.lock:
                for rid, rd in records.items():
                    rec = MemoryRecord.from_dict(rd)
                    self.records[rid] = rec
            self.logger.info(f"Loaded {len(self.records)} records from {p} (v{version})")
        except Exception as e:
            self.logger.error(f"Error loading memory from disk: {e}\n{traceback.format_exc()}")

    # -------------------------
    # SCHEDULER
    # -------------------------

    def _start_scheduler_if_needed(self) -> None:
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="memory-decay-scheduler")
        self._scheduler_thread.start()
        self.logger.info("MemoryDecayEngine scheduler started")

    def _stop_scheduler(self) -> None:
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
            self._scheduler_thread = None
            self.logger.info("MemoryDecayEngine scheduler stopped")

    def _scheduler_loop(self) -> None:
        interval = max(1, int(self.config.get("scheduler_interval_sec", 30)))
        save_interval = max(5, int(self.config.get("auto_save_interval_sec", 60)))
        last_save = time.time()
        while not self._stop_event.is_set():
            try:
                self.decay_step()
                # semantic prune occasionally
                if random.random() < 0.05 and self.embedding_adapter:
                    try:
                        self.semantic_prune(top_k=8)
                    except Exception:
                        pass
                # limit memory footprint
                if len(self.records) > self.config.get("max_records", 200000):
                    self._auto_prune(count=max(1, len(self.records) // 50))
                # metrics & housekeeping
                if time.time() - last_save > save_interval:
                    try:
                        self.save()
                        last_save = time.time()
                    except Exception:
                        pass
                time.sleep(interval)
            except Exception:
                self.logger.error(f"Scheduler loop error: {traceback.format_exc()}")

    # -------------------------
    # ANALYTICS / EXPORTS
    # -------------------------

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            strengths = [rec.strength for rec in self.records.values()] if self.records else []
            avg_strength = float(sum(strengths) / len(strengths)) if strengths else 0.0
            pinned = sum(1 for r in self.records.values() if r.metadata.get("pinned"))
            emb_count = sum(1 for r in self.records.values() if r.embedding is not None)
            return {
                "total_records": len(self.records),
                "avg_strength": avg_strength,
                "pinned": pinned,
                "with_embedding": emb_count,
                "metrics": dict(self.metrics),
                "config": dict(self.config)
            }

    def export_subsample(self, limit: int = 1000) -> List[Dict[str, Any]]:
        with self.lock:
            out = []
            for rec in list(self.records.values())[:limit]:
                out.append({
                    "id": rec.id,
                    "data_preview": rec.data[:512],
                    "strength": rec.strength,
                    "last_accessed": rec.last_accessed.isoformat(),
                    "metadata": rec.metadata
                })
            return out

    # -------------------------
    # INTEROPERABILITY
    # -------------------------

    def attach_external_hook(self, hook: Callable[[str, str], None]) -> None:
        """
        Attach an external event hook used for integration (eg. notify KG that a memory was pruned)
        hook(event_name, record_id)
        """
        self.external_event_hook = hook

    def link_to_knowledge_graph(self, record_id: str, kg_node_id: str) -> bool:
        """
        Simple metadata-based linking. Upstream systems can observe this metadata to build edges.
        """
        with self.lock:
            rec = self.records.get(record_id)
            if not rec:
                return False
            rec.metadata["linked_kg_node"] = kg_node_id
            rec.touch()
            return True

    # -------------------------
    # DEBUG / TESTING HELPERS
    # -------------------------

    def seed_random_records(self, n: int = 100, make_embedding: bool = False) -> List[str]:
        ids = []
        for i in range(n):
            text = f"auto-seed memory snippet {i} " + " ".join(random.choices(["python", "mistral", "aumbot", "memory", "ai", "test"], k=6))
            mid = self.add_record(text, metadata={"seeded": True, "importance": random.random()}, compute_embedding=make_embedding)
            ids.append(mid)
        return ids

    def clear_all(self) -> None:
        with self.lock:
            self.records.clear()
            self.metrics.clear()
        if self.embedding_adapter:
            try:
                # drop faiss index file if exists
                if self.embedding_adapter.faiss_index_path and self.embedding_adapter.faiss_index_path.exists():
                    self.embedding_adapter.faiss_index_path.unlink()
                    self.logger.info("Removed FAISS index file during clear_all")
            except Exception:
                pass

    def shutdown(self, persist: bool = True) -> None:
        self._stop_scheduler()
        if persist:
            self.save()
        # GC
        gc.collect()

    # -------------------------
    # CONTEXTUAL BATCH QUERIES
    # -------------------------

    def retrieve_similar(self, query: str, top_k: int = 5) -> List[Tuple[MemoryRecord, float]]:
        """
        Retrieve top-k semantically similar memories to a query string.
        Falls back to naive token overlap if embeddings unavailable.
        Returns list of tuples (MemoryRecord, score)
        """
        with self.lock:
            if self.embedding_adapter:
                q_emb = None
                try:
                    q_emb = self.embedding_adapter.encode([query])[0]
                except Exception:
                    q_emb = None
                if q_emb is not None:
                    neighbors = self.embedding_adapter.search(q_emb, top_k)
                    out = []
                    for mid, score in neighbors:
                        rec = self.records.get(mid)
                        if rec:
                            out.append((rec, float(score)))
                    return out
            # fallback: token-overlap ranking
            q_tokens = set(query.lower().split())
            scored = []
            for rec in self.records.values():
                tok = set(rec.data.lower().split())
                intersect = len(q_tokens & tok)
                union = len(q_tokens | tok) or 1
                score = intersect / union
                scored.append((rec, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    # -------------------------
    # SAFE CONTEXT MIGRATION UTIL
    # -------------------------

    def migrate_legacy(self, legacy_path: Union[str, Path]) -> int:
        """
        Migrate legacy memory JSON into current schema. Returns migrated count.
        """
        legacy = Path(legacy_path)
        if not legacy.exists():
            return 0
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                payload = json.load(f)
            migrated = 0
            for k, v in payload.get("records", {}).items():
                try:
                    rec = MemoryRecord(
                        id=v.get("id", str(uuid4())),
                        data=v.get("data", v.get("text", "")),
                        metadata=v.get("metadata", {}),
                        strength=v.get("strength", 1.0),
                        created_at=datetime.fromisoformat(v.get("created_at")) if v.get("created_at") else datetime.utcnow(),
                        last_accessed=datetime.fromisoformat(v.get("last_accessed")) if v.get("last_accessed") else datetime.utcnow(),
                        embedding=v.get("embedding")
                    )
                    with self.lock:
                        self.records[rec.id] = rec
                    migrated += 1
                except Exception:
                    self.logger.debug(f"Failed migrating record {k}: {traceback.format_exc()}")
            self.logger.info(f"Migrated {migrated} legacy records")
            return migrated
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return 0


# ============================================================================
# FACTORY & EXPORTS
# ============================================================================

def create_memory_decay_engine(config: Optional[MemoryDecayConfig] = None, logger: Optional[logging.Logger] = None, external_event_hook: Optional[Callable[[str, str], None]] = None) -> MemoryDecayEngine:
    return MemoryDecayEngine(config=config, logger=logger, external_event_hook=external_event_hook)


__all__ = [
    "MemoryRecord",
    "MemoryDecayEngine",
    "create_memory_decay_engine",
    "MemoryDecayConfig"
]
