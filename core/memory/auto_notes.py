"""
Auto Notes Module for AumCore_AI

This module automatically captures important information, insights, and key points
from conversations and stores them as structured notes.

Phase-1: Rule-based note extraction with keyword detection.
Future: Integration with Mistral-7B for semantic importance detection.

File: core/memory/auto_notes.py
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

# Type annotations
from typing import TypedDict


# ============================================================================
# CONFIGURATION & TYPES
# ============================================================================

class NotesConfig(TypedDict):
    """Configuration for auto notes."""
    storage_path: str
    auto_save: bool
    max_notes: int
    enable_auto_tagging: bool
    importance_threshold: float


class NoteDict(TypedDict):
    """Type definition for a note."""
    id: str
    content: str
    category: str
    importance: float
    tags: List[str]
    timestamp: datetime
    source: str


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Note:
    """Represents a single auto-generated note."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    category: str = "general"
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "conversation"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate note after initialization."""
        if not self.content:
            raise ValueError("Note content cannot be empty")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")
    
    def to_dict(self) -> NoteDict:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "source": self.source
        }
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the note."""
        tag_lower = tag.lower().strip()
        if tag_lower and tag_lower not in self.tags:
            self.tags.append(tag_lower)


# ============================================================================
# CORE AUTO NOTES CLASS
# ============================================================================

class AutoNotes:
    """
    Automatic note-taking system for conversations.
    
    Features:
    - Automatic key point extraction
    - Importance scoring
    - Category classification
    - Tag generation
    - Search and filtering
    - Note organization
    
    Attributes:
        config: Notes configuration
        notes: Dictionary of notes by ID
        category_index: Index of notes by category
        tag_index: Index of notes by tag
        logger: Centralized logger instance
    
    Example:
        >>> auto_notes = AutoNotes()
        >>> note_id = auto_notes.capture_from_text("User prefers Python for AI")
        >>> notes = auto_notes.search("Python")
    """
    
    def __init__(
        self,
        storage_path: str = "./data/auto_notes",
        auto_save: bool = True,
        max_notes: int = 5000,
        enable_auto_tagging: bool = True,
        importance_threshold: float = 0.3,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize auto notes system.
        
        Args:
            storage_path: Directory path for notes storage
            auto_save: Automatically save after modifications
            max_notes: Maximum number of notes to retain
            enable_auto_tagging: Enable automatic tag generation
            importance_threshold: Minimum importance for capturing
            logger: Optional logger instance
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validation
        if max_notes <= 0:
            raise ValueError("max_notes must be positive")
        if not 0.0 <= importance_threshold <= 1.0:
            raise ValueError("importance_threshold must be between 0.0 and 1.0")
        
        # Configuration
        self.config: NotesConfig = {
            "storage_path": storage_path,
            "auto_save": auto_save,
            "max_notes": max_notes,
            "enable_auto_tagging": enable_auto_tagging,
            "importance_threshold": importance_threshold
        }
        
        # Storage
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Notes storage
        self.notes: Dict[str, Note] = {}
        
        # Indexes for fast lookup
        self.category_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        
        # Keywords for importance detection (Phase-1)
        self._importance_keywords = {
            "high": ["important", "critical", "essential", "must", "required", 
                    "remember", "note", "key", "crucial", "vital"],
            "medium": ["should", "prefer", "like", "want", "need", "consider"],
            "context": ["because", "since", "due to", "reason", "explanation"]
        }
        
        # Category keywords (Phase-1)
        self._category_keywords = {
            "preference": ["prefer", "like", "favorite", "choice", "want"],
            "fact": ["is", "are", "was", "were", "fact", "true"],
            "instruction": ["do", "don't", "should", "must", "need to"],
            "question": ["what", "why", "how", "when", "where", "who"],
            "insight": ["realize", "understand", "learn", "discover", "found"]
        }
        
        # Logging
        self.logger = logger or self._setup_logger()
        
        # Load existing notes
        self._load_from_disk()
        
        self.logger.info(
            f"AutoNotes initialized: {len(self.notes)} notes loaded"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup centralized logger (Phase-1 fallback)."""
        logger = logging.getLogger("aumcore.memory.auto_notes")
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
    # PUBLIC API - NOTE CAPTURE
    # ========================================================================
    
    def capture_from_text(
        self,
        text: str,
        source: str = "conversation",
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture notes from text automatically.
        
        Args:
            text: Text to extract notes from
            source: Source of the text
            context: Optional context information
        
        Returns:
            Note ID if captured, None if below threshold
        
        Example:
            >>> note_id = auto_notes.capture_from_text("Remember to use Python 3.10")
        """
        try:
            # Calculate importance
            importance = self._calculate_importance(text)
            
            # Skip if below threshold
            if importance < self.config["importance_threshold"]:
                self.logger.debug(f"Text below importance threshold: {importance:.2f}")
                return None
            
            # Detect category
            category = self._detect_category(text)
            
            # Create note
            note = Note(
                content=text.strip(),
                category=category,
                importance=importance,
                source=source,
                context=context or {}
            )
            
            # Auto-tagging
            if self.config["enable_auto_tagging"]:
                tags = self._extract_tags(text)
                for tag in tags:
                    note.add_tag(tag)
            
            # Check limit
            if len(self.notes) >= self.config["max_notes"]:
                self._prune_notes()
            
            # Store note
            self.notes[note.id] = note
            
            # Update indexes
            self._update_indexes(note)
            
            self.logger.debug(
                f"Captured note: {category}, importance={importance:.2f}"
            )
            
            if self.config["auto_save"]:
                self._save_to_disk()
            
            return note.id
        
        except Exception as e:
            self.logger.error(f"Error capturing note: {e}")
            return None
    
    def capture_from_conversation(
        self,
        messages: List[Dict[str, str]],
        extract_summary: bool = True
    ) -> List[str]:
        """
        Capture notes from conversation history.
        
        Args:
            messages: List of conversation messages
            extract_summary: Extract summary note
        
        Returns:
            List of captured note IDs
        
        Example:
            >>> messages = [{"role": "user", "content": "I like Python"}]
            >>> note_ids = auto_notes.capture_from_conversation(messages)
        """
        captured_ids = []
        
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "unknown")
            
            note_id = self.capture_from_text(
                content,
                source=f"conversation:{role}",
                context={"role": role}
            )
            
            if note_id:
                captured_ids.append(note_id)
        
        # Extract summary if enabled
        if extract_summary and len(messages) > 2:
            summary = self._generate_conversation_summary(messages)
            if summary:
                summary_id = self.capture_from_text(
                    summary,
                    source="conversation:summary",
                    context={"message_count": len(messages)}
                )
                if summary_id:
                    captured_ids.append(summary_id)
        
        return captured_ids
    
    def add_note(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        source: str = "manual"
    ) -> str:
        """
        Manually add a note.
        
        Args:
            content: Note content
            category: Note category
            importance: Importance score
            tags: Optional tags
            source: Note source
        
        Returns:
            Note ID
        
        Example:
            >>> note_id = auto_notes.add_note("Important deadline", "reminder", 0.9)
        """
        note = Note(
            content=content,
            category=category,
            importance=importance,
            tags=tags or [],
            source=source
        )
        
        self.notes[note.id] = note
        self._update_indexes(note)
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        self.logger.debug(f"Added manual note: {note.id}")
        
        return note.id
    
    # ========================================================================
    # PUBLIC API - NOTE RETRIEVAL
    # ========================================================================
    
    def get_note(self, note_id: str) -> Optional[Note]:
        """
        Retrieve a specific note.
        
        Args:
            note_id: Note ID
        
        Returns:
            Note if found, None otherwise
        
        Example:
            >>> note = auto_notes.get_note(note_id)
        """
        return self.notes.get(note_id)
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[Note]:
        """
        Search notes by content.
        
        Args:
            query: Search query
            category: Filter by category
            min_importance: Minimum importance threshold
            limit: Maximum results
        
        Returns:
            List of matching notes
        
        Example:
            >>> results = auto_notes.search("Python", category="preference")
        """
        query_lower = query.lower()
        matches = []
        
        for note in self.notes.values():
            # Category filter
            if category and note.category != category:
                continue
            
            # Importance filter
            if note.importance < min_importance:
                continue
            
            # Content match
            if query_lower in note.content.lower():
                matches.append(note)
        
        # Sort by importance (descending)
        matches.sort(key=lambda n: n.importance, reverse=True)
        
        return matches[:limit]
    
    def get_by_category(self, category: str) -> List[Note]:
        """
        Get all notes in a category.
        
        Args:
            category: Category to filter
        
        Returns:
            List of notes
        
        Example:
            >>> prefs = auto_notes.get_by_category("preference")
        """
        note_ids = self.category_index.get(category, set())
        notes = [self.notes[nid] for nid in note_ids if nid in self.notes]
        
        # Sort by timestamp (newest first)
        notes.sort(key=lambda n: n.timestamp, reverse=True)
        
        return notes
    
    def get_by_tag(self, tag: str) -> List[Note]:
        """
        Get all notes with a specific tag.
        
        Args:
            tag: Tag to filter
        
        Returns:
            List of notes
        
        Example:
            >>> python_notes = auto_notes.get_by_tag("python")
        """
        note_ids = self.tag_index.get(tag.lower(), set())
        notes = [self.notes[nid] for nid in note_ids if nid in self.notes]
        
        # Sort by importance
        notes.sort(key=lambda n: n.importance, reverse=True)
        
        return notes
    
    def get_most_important(self, limit: int = 10) -> List[Note]:
        """
        Get most important notes.
        
        Args:
            limit: Number of notes to retrieve
        
        Returns:
            List of notes sorted by importance
        
        Example:
            >>> important = auto_notes.get_most_important(5)
        """
        sorted_notes = sorted(
            self.notes.values(),
            key=lambda n: n.importance,
            reverse=True
        )
        return sorted_notes[:limit]
    
    def get_recent(self, hours: int = 24, limit: int = 10) -> List[Note]:
        """
        Get recent notes within time window.
        
        Args:
            hours: Time window in hours
            limit: Maximum results
        
        Returns:
            List of recent notes
        
        Example:
            >>> recent = auto_notes.get_recent(hours=48)
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_notes = [
            note for note in self.notes.values()
            if note.timestamp >= cutoff_time
        ]
        
        # Sort by timestamp (newest first)
        recent_notes.sort(key=lambda n: n.timestamp, reverse=True)
        
        return recent_notes[:limit]
    
    # ========================================================================
    # PUBLIC API - NOTE MANAGEMENT
    # ========================================================================
    
    def update_note(
        self,
        note_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update a note.
        
        Args:
            note_id: Note ID to update
            content: New content
            category: New category
            importance: New importance
            tags: New tags
        
        Returns:
            True if updated, False if not found
        
        Example:
            >>> auto_notes.update_note(note_id, importance=0.9)
        """
        note = self.notes.get(note_id)
        if not note:
            return False
        
        # Remove from old indexes
        self._remove_from_indexes(note)
        
        # Update fields
        if content is not None:
            note.content = content
        if category is not None:
            note.category = category
        if importance is not None:
            note.importance = max(0.0, min(1.0, importance))
        if tags is not None:
            note.tags = [t.lower() for t in tags]
        
        # Re-add to indexes
        self._update_indexes(note)
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        self.logger.debug(f"Updated note: {note_id}")
        return True
    
    def delete_note(self, note_id: str) -> bool:
        """
        Delete a note.
        
        Args:
            note_id: Note ID to delete
        
        Returns:
            True if deleted, False if not found
        
        Example:
            >>> auto_notes.delete_note(note_id)
        """
        note = self.notes.get(note_id)
        if not note:
            return False
        
        # Remove from indexes
        self._remove_from_indexes(note)
        
        # Delete note
        del self.notes[note_id]
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        self.logger.debug(f"Deleted note: {note_id}")
        return True
    
    def clear_all(self) -> None:
        """
        Clear all notes.
        
        Example:
            >>> auto_notes.clear_all()
        """
        count = len(self.notes)
        
        self.notes.clear()
        self.category_index.clear()
        self.tag_index.clear()
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        self.logger.info(f"Cleared {count} notes")
    
    # ========================================================================
    # STATISTICS & ANALYSIS
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get notes statistics.
        
        Returns:
            Dictionary with statistics
        
        Example:
            >>> stats = auto_notes.get_statistics()
        """
        if not self.notes:
            return {
                "total_notes": 0,
                "by_category": {},
                "by_source": {},
                "avg_importance": 0.0
            }
        
        categories = {}
        sources = {}
        
        for note in self.notes.values():
            categories[note.category] = categories.get(note.category, 0) + 1
            sources[note.source] = sources.get(note.source, 0) + 1
        
        return {
            "total_notes": len(self.notes),
            "by_category": categories,
            "by_source": sources,
            "avg_importance": sum(n.importance for n in self.notes.values()) / len(self.notes),
            "total_tags": len(self.tag_index),
            "config": self.config.copy()
        }
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self) -> None:
        """
        Manually save notes to disk.
        
        Example:
            >>> auto_notes.save()
        """
        self._save_to_disk()
    
    def _save_to_disk(self) -> None:
        """Save notes to disk storage."""
        try:
            notes_file = self.storage_path / "notes.json"
            
            data = {
                "notes": {},
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            # Serialize notes
            for note_id, note in self.notes.items():
                data["notes"][note_id] = {
                    "id": note.id,
                    "content": note.content,
                    "category": note.category,
                    "importance": note.importance,
                    "tags": note.tags,
                    "timestamp": note.timestamp.isoformat(),
                    "source": note.source,
                    "context": note.context
                }
            
            with open(notes_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved {len(self.notes)} notes to disk")
        
        except Exception as e:
            self.logger.error(f"Error saving notes: {e}")
    
    def _load_from_disk(self) -> None:
        """Load notes from disk storage."""
        try:
            notes_file = self.storage_path / "notes.json"
            
            if not notes_file.exists():
                self.logger.info("No existing notes file found")
                return
            
            with open(notes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load notes
            for note_id, note_data in data.get("notes", {}).items():
                note = Note(
                    id=note_data["id"],
                    content=note_data["content"],
                    category=note_data["category"],
                    importance=note_data["importance"],
                    tags=note_data.get("tags", []),
                    timestamp=datetime.fromisoformat(note_data["timestamp"]),
                    source=note_data.get("source", "unknown"),
                    context=note_data.get("context", {})
                )
                self.notes[note_id] = note
                self._update_indexes(note)
            
            self.logger.info(f"Loaded {len(self.notes)} notes from disk")
        
        except Exception as e:
            self.logger.error(f"Error loading notes: {e}")
    
    # ========================================================================
    # INTERNAL HELPERS - IMPORTANCE & CATEGORIZATION
    # ========================================================================
    
    def _calculate_importance(self, text: str) -> float:
        """
        Calculate importance score for text (Phase-1: keyword-based).
        
        Args:
            text: Text to score
        
        Returns:
            Importance score (0.0 to 1.0)
        """
        text_lower = text.lower()
        score = 0.5  # Base score
        
        # Check high importance keywords
        for keyword in self._importance_keywords["high"]:
            if keyword in text_lower:
                score += 0.15
        
        # Check medium importance keywords
        for keyword in self._importance_keywords["medium"]:
            if keyword in text_lower:
                score += 0.08
        
        # Check context keywords
        for keyword in self._importance_keywords["context"]:
            if keyword in text_lower:
                score += 0.05
        
        # Length bonus (longer text might be more detailed)
        words = len(text.split())
        if words > 20:
            score += 0.1
        
        # Question mark or exclamation (emphasis)
        if '?' in text or '!' in text:
            score += 0.05
        
        return min(1.0, score)
    
    def _detect_category(self, text: str) -> str:
        """
        Detect category from text (Phase-1: keyword-based).
        
        Args:
            text: Text to categorize
        
        Returns:
            Category name
        """
        text_lower = text.lower()
        
        category_scores = {}
        for category, keywords in self._category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _extract_tags(self, text: str) -> List[str]:
        """
        Extract tags from text (Phase-1: simple extraction).
        
        Args:
            text: Text to extract tags from
        
        Returns:
            List of tags
        """
        tags = []
        
        # Extract capitalized words (potential proper nouns)
        words = re.findall(r'[A-Z][a-z]+', text)
        tags.extend([w.lower() for w in words[:3]])  # Limit to 3
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', text)
        tags.extend([q.lower() for q in quoted[:2]])
        
        # Extract technology/programming terms (common patterns)
        tech_patterns = [
            r'(python|java|javascript|c\\+\\+|rust|go)',
            r'(ai|ml|nlp|api|database|server)',
            r'(react|vue|angular|django|flask)'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text.lower())
            tags.extend(matches)
        
        # Remove duplicates and limit
        return list(dict.fromkeys(tags))[:5]
    
    def _generate_conversation_summary(
        self,
        messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Generate summary from conversation (Phase-1: simple extraction).
        
        Args:
            messages: Conversation messages
        
        Returns:
            Summary text or None
        """
        if len(messages) < 3:
            return None
        
        # Extract important sentences
        important_parts = []
        
        for msg in messages:
            content = msg.get("content", "")
            importance = self._calculate_importance(content)
            
            if importance > 0.6:
                # Take first sentence
                sentences = content.split('.')
                if sentences:
                    important_parts.append(sentences[0].strip())
        
        if important_parts:
            summary = "Conversation summary: " + "; ".join(important_parts[:3])
            return summary
        
        return None
    
    # ========================================================================
    # INTERNAL HELPERS - INDEX MANAGEMENT
    # ========================================================================
    
    def _update_indexes(self, note: Note) -> None:
        """Update indexes with note information."""
        # Category index
        if note.category not in self.category_index:
            self.category_index[note.category] = set()
        self.category_index[note.category].add(note.id)
        
        # Tag index
        for tag in note.tags:
            tag_lower = tag.lower()
            if tag_lower not in self.tag_index:
                self.tag_index[tag_lower] = set()
            self.tag_index[tag_lower].add(note.id)
    
    def _remove_from_indexes(self, note: Note) -> None:
        """Remove note from indexes."""
        # Category index
        if note.category in self.category_index:
            self.category_index[note.category].discard(note.id)
        
        # Tag index
        for tag in note.tags:
            tag_lower = tag.lower()
            if tag_lower in self.tag_index:
                self.tag_index[tag_lower].discard(note.id)
    
    def _prune_notes(self) -> None:
        """Remove low importance notes when limit reached."""
        if len(self.notes) < self.config["max_notes"]:
            return
        
        # Sort by importance (ascending)
        sorted_notes = sorted(
            self.notes.items(),
            key=lambda x: x[1].importance
        )
        
        # Remove bottom 10%
        remove_count = max(1, len(self.notes) // 10)
        
        for note_id, note in sorted_notes[:remove_count]:
            self.delete_note(note_id)
        
        self.logger.info(f"Pruned {remove_count} low-importance notes")
    
    # ========================================================================
    # ASYNC API (Future-ready)
    # ========================================================================
    
    async def capture_from_text_async(
        self,
        text: str,
        source: str = "conversation"
    ) -> Optional[str]:
        """Async version of capture_from_text."""
        return await asyncio.to_thread(
            self.capture_from_text, text, source
        )
    
    async def search_async(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Note]:
        """Async version of search."""
        return await asyncio.to_thread(
            self.search, query, category, 0.0, limit
        )


# ============================================================================
# FACTORY PATTERN (Dependency Injection)
# ============================================================================

def create_auto_notes(
    config: Optional[NotesConfig] = None,
    logger: Optional[logging.Logger] = None
) -> AutoNotes:
    """
    Factory function to create AutoNotes instance.
    
    Args:
        config: Optional notes configuration
        logger: Optional logger instance
    
    Returns:
        Configured AutoNotes instance
    
    Example:
        >>> notes = create_auto_notes({"max_notes": 10000})
    """
    if config is None:
        config = {
            "storage_path": "./data/auto_notes",
            "auto_save": True,
            "max_notes": 5000,
            "enable_auto_tagging": True,
            "importance_threshold": 0.3
        }
    
    return AutoNotes(
        storage_path=config["storage_path"],
        auto_save=config["auto_save"],
        max_notes=config["max_notes"],
        enable_auto_tagging=config["enable_auto_tagging"],
        importance_threshold=config["importance_threshold"],
        logger=logger
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "AutoNotes",
    "Note",
    "NoteDict",
    "NotesConfig",
    "create_auto_notes"
]


# ============================================================================
# TODO: FUTURE MODEL INTEGRATION (Phase-2+)
# ============================================================================

# TODO: Integrate Mistral-7B for:
# - Semantic importance detection
# - Intelligent summarization
# - Entity and concept extraction
# - Automatic categorization
# - Context-aware tagging

# TODO: Add note linking and relationship detection

# TODO: Implement smart note merging (duplicate detection)

# TODO: Add export to markdown/PDF formats

# TODO: Implement collaborative note-taking features
