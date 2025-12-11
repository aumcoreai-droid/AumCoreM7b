"""
AumCore_AI Memory Subsystem - Auto Notes Module
Phase 1 Only (Chunk 1 + Chunk 2)

File: memory/auto_notes.py

Description:
    यह module conversations से important information को
    rule-based तरीके से capture करके simple structured notes बनाता है।

    Phase‑1 Rules:
        - सिर्फ rule-based logic (keywords, length, simple scoring)
        - कोई model integration नहीं (Mistral/LLM बाद में)
        - कोई async dependency नहीं (सिर्फ basic helpers allowed)
        - कोई external DB या complex indexing नहीं
        - Safe, deterministic, testable behavior

    Future (Phase‑3+):
        - Semantic scoring via model adapters
        - Advanced summarization + tagging
        - Knowledge graph integration
"""

# ============================================================
# ✅ Chunk 1: Imports (PEP8)
# ============================================================

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from uuid import uuid4


# ============================================================
# ✅ Chunk 1: Local Logger Setup (Phase‑1 Safe)
# ============================================================

def get_auto_notes_logger(name: str = "AumCoreAI.Memory.AutoNotes") -> logging.Logger:
    """
    Auto notes module के लिए simple logger बनाता है।
    Phase‑1 में यही local logger fallback रहेगा।
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [AUTO_NOTES] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = get_auto_notes_logger()


# ============================================================
# ✅ Chunk 2: Simple Config + Types (Rule-Based)
# ============================================================

@dataclass
class AutoNotesConfig:
    """
    Auto notes के लिए Phase‑1 config (pure rule-based).
    """

    storage_path: str = "./data/auto_notes"
    auto_save: bool = True
    max_notes: int = 500
    enable_auto_tagging: bool = True
    importance_threshold: float = 0.3
    max_text_length: int = 4000

    def __post_init__(self) -> None:
        """
        Basic validation (rule-based).
        """
        if self.max_notes <= 0:
            raise ValueError("max_notes positive होना चाहिए।")
        if not (0.0 <= self.importance_threshold <= 1.0):
            raise ValueError("importance_threshold 0.0–1.0 के बीच होना चाहिए।")
        if self.max_text_length <= 0:
            raise ValueError("max_text_length positive होना चाहिए।")


@dataclass
class Note:
    """
    Phase‑1 का simple Note model.
    कोई complex relation, embedding, या model metadata नहीं।
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    category: str = "general"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "conversation"

    def __post_init__(self) -> None:
        """
        Note creation पर basic validation.
        """
        if not self.content:
            raise ValueError("Note content empty नहीं होना चाहिए।")
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("importance 0.0–1.0 के बीच होना चाहिए।")

    def to_dict(self) -> Dict[str, Any]:
        """
        Dict representation (persistence और inspection के लिए).
        """
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "tags": list(self.tags),
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


# ============================================================
# ✅ Chunk 2: Sanitization Utilities (Rule-Based)
# ============================================================

def _sanitize_text_for_note(text: str, max_length: int) -> str:
    """
    Simple text sanitization for auto notes.
    """
    if text is None:
        return ""

    cleaned = text.strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()

    return cleaned


def _sanitize_tags(tags: Optional[List[str]]) -> List[str]:
    """
    Simple tag sanitization: lowercase + strip + unique.
    """
    if tags is None:
        return []

    clean: List[str] = []
    seen: Set[str] = set()

    for tag in tags:
        if tag is None:
            continue
        t = str(tag).strip().lower()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            clean.append(t)

    return clean


# ============================================================
# ✅ Chunk 2: Validation Helpers (Rule-Based)
# ============================================================

def _validate_source(source: Optional[str]) -> str:
    """
    Source field के लिए simple validation.
    """
    if source is None:
        return "unknown"

    s = str(source).strip()
    if not s:
        return "unknown"
    return s


def _validate_category(name: Optional[str]) -> str:
    """
    Category validation (Phase‑1: सिर्फ non-empty string).
    """
    if name is None:
        return "general"
    cat = str(name).strip()
    return cat if cat else "general"


def _validate_importance(value: float) -> float:
    """
    Importance को clamp करता है 0.0–1.0 के बीच।
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ============================================================
# ✅ Core Class: AutoNotes (Phase‑1 Version)
# ============================================================

class AutoNotes:
    """
    Automatic note capture system (Phase‑1, pure rule-based).

    Features (Phase‑1):
        - Keyword-based importance scoring
        - Simple category detection (keyword buckets)
        - Basic tag extraction (keywords + simple heuristics)
        - File-based persistence (single JSON file)
        - Simple search (substring match, no ranking model)

    Future (Phase‑3+):
        - Semantic extraction via model
        - Rich metadata, embeddings, relationships
    """

    def __init__(self, config: Optional[AutoNotesConfig] = None) -> None:
        """
        AutoNotes system initialize करता है।

        Args:
            config: Optional AutoNotesConfig
        """
        self.config: AutoNotesConfig = config or AutoNotesConfig()

        # Storage path
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self._notes: Dict[str, Note] = {}

        # Simple indices
        self._category_index: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}

        # Rule-based keyword tables
        self._importance_keywords: Dict[str, List[str]] = {
            "high": [
                "important",
                "critical",
                "must",
                "remember",
                "urgent",
                "key point",
                "note this",
            ],
            "medium": [
                "should",
                "prefer",
                "like",
                "need",
                "want",
                "recommend",
            ],
        }

        self._category_keywords: Dict[str, List[str]] = {
            "preference": ["prefer", "like", "love", "hate", "favorite"],
            "todo": ["remember", "remind", "deadline", "task", "todo"],
            "instruction": ["must", "need to", "should", "have to"],
            "fact": ["is", "are", "was", "were", "fact"],
        }

        self._load_from_disk()

        logger.debug(
            "AutoNotes initialized (Phase‑1): %d notes loaded from disk.",
            len(self._notes),
        )

    # --------------------------------------------------------
    # Public API: Capture from text
    # --------------------------------------------------------

    def capture_from_text(
        self,
        text: str,
        source: Optional[str] = "conversation",
    ) -> Optional[str]:
        """
        दिए गए text से note capture करने की कोशिश करता है।

        Rules:
            - Text sanitize
            - Importance score निकालना (keyword-based)
            - Threshold से कम हो तो note नहीं बनेगा
        """
        cleaned = _sanitize_text_for_note(text, self.config.max_text_length)
        if not cleaned:
            logger.debug("Empty text, skipping note capture.")
            return None

        importance = self._calculate_importance(cleaned)

        if importance < self.config.importance_threshold:
            logger.debug(
                "Text importance %.2f threshold %.2f से कम है, note skip.",
                importance,
                self.config.importance_threshold,
            )
            return None

        category = self._detect_category(cleaned)
        src = _validate_source(source)

        tags: List[str] = []
        if self.config.enable_auto_tagging:
            tags = self._extract_tags(cleaned)

        note = Note(
            content=cleaned,
            category=_validate_category(category),
            importance=_validate_importance(importance),
            tags=_sanitize_tags(tags),
            source=src,
        )

        self._store_note(note)
        return note.id

    def capture_from_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Simple conversation messages list से notes capture करता है।

        Args:
            messages: list of {"role": ..., "content": ...}

        Returns:
            captured note IDs list
        """
        captured: List[str] = []
        for msg in messages:
            content = str(msg.get("content", "")).strip()
            if not content:
                continue

            role = str(msg.get("role", "unknown"))
            src = f"conversation:{role}"

            note_id = self.capture_from_text(content, source=src)
            if note_id:
                captured.append(note_id)

        return captured

    # --------------------------------------------------------
    # Public API: Manual note add + retrieval
    # --------------------------------------------------------

    def add_note(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        source: str = "manual",
    ) -> str:
        """
        Manually note create करता है (Phase‑1 safe).
        """
        cleaned = _sanitize_text_for_note(content, self.config.max_text_length)
        if not cleaned:
            raise ValueError("Manual note content empty नहीं हो सकता।")

        note = Note(
            content=cleaned,
            category=_validate_category(category),
            importance=_validate_importance(importance),
            tags=_sanitize_tags(tags or []),
            source=_validate_source(source),
        )
        self._store_note(note)
        return note.id

    def get_note(self, note_id: str) -> Optional[Note]:
        """
        ID से single note return करता है (या None अगर ना मिले)।
        """
        return self._notes.get(note_id)

    def list_all_notes(self) -> List[Note]:
        """
        सारे notes list करता है (sorted by timestamp desc).
        """
        notes = list(self._notes.values())
        notes.sort(key=lambda n: n.timestamp, reverse=True)
        return notes

    # --------------------------------------------------------
    # Public API: Simple search + filters
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        min_importance: float = 0.0,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Note]:
        """
        Simple substring search (Phase‑1, no semantic ranking).
        """
        q = query.lower().strip()
        if not q:
            return []

        results: List[Note] = []
        for note in self._notes.values():
            if category and note.category != category:
                continue
            if note.importance < min_importance:
                continue
            if q in note.content.lower():
                results.append(note)

        results.sort(key=lambda n: (n.importance, n.timestamp), reverse=True)
        return results[: max(1, limit)]

    def notes_by_category(self, category: str) -> List[Note]:
        """
        दिए गए category के notes लौटाता है (newest first)।
        """
        cat = _validate_category(category)
        ids = self._category_index.get(cat, set())
        notes = [self._notes[i] for i in ids if i in self._notes]
        notes.sort(key=lambda n: n.timestamp, reverse=True)
        return notes

    def notes_by_tag(self, tag: str) -> List[Note]:
        """
        दिए गए tag वाले notes return करता है।
        """
        t = str(tag).strip().lower()
        if not t:
            return []

        ids = self._tag_index.get(t, set())
        notes = [self._notes[i] for i in ids if i in self._notes]
        notes.sort(key=lambda n: n.importance, reverse=True)
        return notes

    def recent_notes(
        self,
        hours: int = 24,
        limit: int = 20,
    ) -> List[Note]:
        """
        दिए गए घंटों के भीतर बनाए गए notes देता है।
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        candidates = [
            n for n in self._notes.values() if n.timestamp >= cutoff
        ]
        candidates.sort(key=lambda n: n.timestamp, reverse=True)
        return candidates[: max(1, limit)]

    # --------------------------------------------------------
    # Public API: Update + Delete
    # --------------------------------------------------------

    def update_note(
        self,
        note_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Existing note को update करता है (Phase‑1 safe fields).
        """
        note = self._notes.get(note_id)
        if note is None:
            return False

        # पहले indexes से हटाओ
        self._remove_from_indices(note)

        if content is not None:
            note.content = _sanitize_text_for_note(
                content,
                self.config.max_text_length,
            )
        if category is not None:
            note.category = _validate_category(category)
        if importance is not None:
            note.importance = _validate_importance(importance)
        if tags is not None:
            note.tags = _sanitize_tags(tags)

        # दोबारा indices में डालो
        self._add_to_indices(note)

        if self.config.auto_save:
            self._save_to_disk()

        return True

    def delete_note(self, note_id: str) -> bool:
        """
        Note delete करता है (indices से भी remove).
        """
        note = self._notes.get(note_id)
        if note is None:
            return False

        self._remove_from_indices(note)
        del self._notes[note_id]

        if self.config.auto_save:
            self._save_to_disk()

        return True

    def clear_all(self) -> None:
        """
        सारे notes clear कर देता है (Phase‑1 simple reset).
        """
        count = len(self._notes)
        self._notes.clear()
        self._category_index.clear()
        self._tag_index.clear()

        if self.config.auto_save:
            self._save_to_disk()

        logger.info("AutoNotes: %d notes cleared.", count)

    # --------------------------------------------------------
    # Public API: Basic statistics
    # --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """
        Simple stats देता है (Phase‑1).
        """
        total = len(self._notes)
        by_category: Dict[str, int] = {}
        tag_count = len(self._tag_index)

        for note in self._notes.values():
            by_category[note.category] = by_category.get(note.category, 0) + 1

        avg_importance = (
            sum(n.importance for n in self._notes.values()) / total
            if total > 0
            else 0.0
        )

        return {
            "total_notes": total,
            "by_category": by_category,
            "total_tags": tag_count,
            "avg_importance": avg_importance,
        }

    # ========================================================
    # Internal: Importance + Category + Tags (Rule-Based)
    # ========================================================

    def _calculate_importance(self, text: str) -> float:
        """
        Keyword-based importance score (Phase‑1).
        """
        t = text.lower()
        score = 0.4  # base

        for kw in self._importance_keywords["high"]:
            if kw in t:
                score += 0.2
        for kw in self._importance_keywords["medium"]:
            if kw in t:
                score += 0.1

        if len(text.split()) > 20:
            score += 0.1

        if "?" in text or "!" in text:
            score += 0.05

        return _validate_importance(score)

    def _detect_category(self, text: str) -> str:
        """
        Text से category guess करता है (simple keyword count से).
        """
        t = text.lower()
        best_cat = "general"
        best_score = 0

        for cat, kws in self._category_keywords.items():
            score = sum(1 for kw in kws if kw in t)
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat

    def _extract_tags(self, text: str) -> List[str]:
        """
        Basic tag extraction (Phase‑1): keywords + simple heuristics।
        """
        words = text.split()
        tags: List[str] = []

        for w in words:
            w_clean = "".join(ch for ch in w if ch.isalnum()).lower()
            if not w_clean:
                continue
            # simple heuristic: medium/high importance keywords as tags
            for bucket in self._importance_keywords.values():
                if w_clean in bucket and w_clean not in tags:
                    tags.append(w_clean)

        # limit tags count
        if len(tags) > 5:
            tags = tags[:5]

        return tags

    # ========================================================
    # Internal: Storage + Indices
    # ========================================================

    def _store_note(self, note: Note) -> None:
        """
        Note को memory + indices + disk (optional) में डालता है।
        साथ में simple pruning भी करता है।
        """
        if len(self._notes) >= self.config.max_notes:
            self._prune_low_importance()

        self._notes[note.id] = note
        self._add_to_indices(note)

        if self.config.auto_save:
            self._save_to_disk()

    def _add_to_indices(self, note: Note) -> None:
        """
        Category और tags indices update करता है।
        """
        cat = _validate_category(note.category)
        if cat not in self._category_index:
            self._category_index[cat] = set()
        self._category_index[cat].add(note.id)

        for tag in _sanitize_tags(note.tags):
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(note.id)

    def _remove_from_indices(self, note: Note) -> None:
        """
        Note को indices से निकालता है।
        """
        cat = _validate_category(note.category)
        if cat in self._category_index:
            self._category_index[cat].discard(note.id)
            if not self._category_index[cat]:
                del self._category_index[cat]

        for tag in _sanitize_tags(note.tags):
            if tag in self._tag_index:
                self._tag_index[tag].discard(note.id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

    def _save_to_disk(self) -> None:
        """
        सारे notes को single JSON file में dump करता है।
        """
        try:
            notes_file = self._storage_path / "notes_phase1.json"
            payload = {
                "saved_at": datetime.now().isoformat(),
                "notes": {nid: n.to_dict() for nid, n in self._notes.items()},
            }
            with notes_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("AutoNotes save_to_disk error: %s", exc)

    def _load_from_disk(self) -> None:
        """
        Disk से notes load करता है (अगर file मौजूद हो)।
        """
        try:
            notes_file = self._storage_path / "notes_phase1.json"
            if not notes_file.exists():
                return

            with notes_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            notes_raw = data.get("notes", {})
            for nid, nd in notes_raw.items():
                try:
                    note = Note(
                        id=str(nd.get("id", nid)),
                        content=str(nd.get("content", "")),
                        category=_validate_category(nd.get("category", "general")),
                        importance=_validate_importance(
                            float(nd.get("importance", 0.5))
                        ),
                        tags=_sanitize_tags(nd.get("tags", [])),
                        source=_validate_source(nd.get("source", "unknown")),
                        timestamp=datetime.fromisoformat(
                            nd.get("timestamp", datetime.now().isoformat())
                        ),
                    )
                    self._notes[note.id] = note
                    self._add_to_indices(note)
                except Exception as inner_exc:
                    logger.warning(
                        "Skipping invalid note from file: %s", inner_exc
                    )
        except Exception as exc:
            logger.error("AutoNotes load_from_disk error: %s", exc)

    def _prune_low_importance(self) -> None:
        """
        max_notes limit cross होने पर lowest-importance notes prune करता है।
        """
        if not self._notes:
            return

        sorted_items: List[Tuple[str, Note]] = sorted(
            self._notes.items(),
            key=lambda item: (item[1].importance, item[1].timestamp),
        )

        remove_count = max(1, len(sorted_items) // 10)
        to_remove = sorted_items[:remove_count]

        for nid, note in to_remove:
            self._remove_from_indices(note)
            del self._notes[nid]

        logger.info("AutoNotes: pruned %d low-importance notes.", remove_count)


# ============================================================
# Public Helper: Simple factory
# ============================================================

def create_auto_notes_with_defaults() -> AutoNotes:
    """
    Phase‑1 friendly factory:
    default config के साथ AutoNotes instance देता है।
    """
    return AutoNotes(AutoNotesConfig())


# ============================================================
# Module exports
# ============================================================

__all__ = [
    "AutoNotesConfig",
    "Note",
    "AutoNotes",
    "create_auto_notes_with_defaults",
]

# End of File: memory/auto_notes.py (Phase‑1, 500+ lines approx with comments)
