"""
Long-Term Memory Module for AumCore_AI

This module manages persistent storage and retrieval of conversation history
and learned user preferences across sessions.

Phase-1: Rule-based implementation with file-based storage.
Future: Integration with Mistral-7B for semantic indexing and retrieval.

File: core/memory/long_term_memory.py
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

# Type annotations
from typing import TypedDict


# ============================================================================
# CONFIGURATION & TYPES
# ============================================================================

class MemoryEntryDict(TypedDict):
    """Type definition for a memory entry."""
    id: str
    content: str
    category: str
    timestamp: datetime
    importance: float
    metadata: Dict[str, Any]


class LongTermConfig(TypedDict):
    """Configuration for long-term memory."""
    storage_path: str
    max_entries: int
    auto_save: bool
    compression_enabled: bool


# ============================================================================
# ABSTRACT INTERFACES (SOLID: Dependency Inversion)
# ============================================================================

class PersistentStorageProtocol(Protocol):
    """Protocol for persistent storage backends."""
    
    def save(self, data: Dict[str, Any]) -> None:
        """Save data to storage."""
        ...
    
    def load(self) -> Dict[str, Any]:
        """Load data from storage."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete specific entry."""
        ...


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MemoryEntry:
    """Represents a single long-term memory entry."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    category: str = "general"
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate entry after initialization."""
        if not self.content:
            raise ValueError("Content cannot be empty")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")
        if not self.category:
            raise ValueError("Category cannot be empty")
    
    def to_dict(self) -> MemoryEntryDict:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata
        }
    
    def increment_access(self) -> None:
        """Increment access counter and update last accessed time."""
        self.access_count += 1
        self.last_accessed = datetime.now()


# ============================================================================
# CORE LONG-TERM MEMORY CLASS
# ============================================================================

class LongTermMemory:
    """
    Manages persistent long-term memory across sessions.
    
    Features:
    - Persistent file-based storage
    - Category-based organization
    - Importance-weighted retrieval
    - Auto-save capabilities
    - Search and filtering
    - Access pattern tracking
    
    Attributes:
        config: Memory configuration
        entries: Dictionary of memory entries by ID
        storage_path: Path to storage directory
        logger: Centralized logger instance
    
    Example:
        >>> memory = LongTermMemory(storage_path="./data/memory")
        >>> memory.add_entry("User prefers Python", category="preference")
        >>> results = memory.search("Python")
    """
    
    def __init__(
        self,
        storage_path: str = "./data/long_term_memory",
        max_entries: int = 10000,
        auto_save: bool = True,
        compression_enabled: bool = False,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize long-term memory.
        
        Args:
            storage_path: Directory path for persistent storage
            max_entries: Maximum number of entries to retain
            auto_save: Automatically save after modifications
            compression_enabled: Enable storage compression
            logger: Optional logger instance
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validation
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        
        # Configuration
        self.config: LongTermConfig = {
            "storage_path": storage_path,
            "max_entries": max_entries,
            "auto_save": auto_save,
            "compression_enabled": compression_enabled
        }
        
        # Storage
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory entries: {id: MemoryEntry}
        self.entries: Dict[str, MemoryEntry] = {}
        
        # Logging
        self.logger = logger or self._setup_logger()
        
        # Load existing memories
        self._load_from_disk()
        
        self.logger.info(
            f"LongTermMemory initialized: path={storage_path}, "
            f"entries={len(self.entries)}"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup centralized logger (Phase-1 fallback)."""
        logger = logging.getLogger("aumcore.memory.long_term")
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
    # PUBLIC API - WRITE OPERATIONS
    # ========================================================================
    
    def add_entry(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new memory entry.
        
        Args:
            content: Memory content
            category: Category classification
            importance: Importance score (0.0-1.0)
            metadata: Optional metadata dictionary
        
        Returns:
            Entry ID
        
        Raises:
            ValueError: If content is invalid
        
        Example:
            >>> entry_id = memory.add_entry("User likes Python", "preference", 0.8)
        """
        try:
            # Input validation
            sanitized_content = self._sanitize_input(content)
            
            entry = MemoryEntry(
                content=sanitized_content,
                category=category,
                importance=importance,
                metadata=metadata or {}
            )
            
            # Check max entries limit
            if len(self.entries) >= self.config["max_entries"]:
                self._prune_low_importance()
            
            self.entries[entry.id] = entry
            
            self.logger.debug(f"Added entry: id={entry.id}, category={category}")
            
            # Auto-save
            if self.config["auto_save"]:
                self._save_to_disk()
            
            return entry.id
        
        except Exception as e:
            self.logger.error(f"Error adding entry: {e}")
            raise
    
    def update_entry(
        self,
        entry_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: Entry ID to update
            content: New content (optional)
            category: New category (optional)
            importance: New importance (optional)
            metadata: New metadata (optional)
        
        Returns:
            True if updated, False if not found
        
        Example:
            >>> memory.update_entry(entry_id, importance=0.9)
        """
        if entry_id not in self.entries:
            self.logger.warning(f"Entry not found: {entry_id}")
            return False
        
        entry = self.entries[entry_id]
        
        if content is not None:
            entry.content = self._sanitize_input(content)
        if category is not None:
            entry.category = category
        if importance is not None:
            entry.importance = max(0.0, min(1.0, importance))
        if metadata is not None:
            entry.metadata.update(metadata)
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        self.logger.debug(f"Updated entry: {entry_id}")
        return True
    
    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: Entry ID to delete
        
        Returns:
            True if deleted, False if not found
        
        Example:
            >>> memory.delete_entry(entry_id)
        """
        if entry_id in self.entries:
            del self.entries[entry_id]
            
            if self.config["auto_save"]:
                self._save_to_disk()
            
            self.logger.debug(f"Deleted entry: {entry_id}")
            return True
        
        return False
    
    # ========================================================================
    # PUBLIC API - READ OPERATIONS
    # ========================================================================
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory entry.
        
        Args:
            entry_id: Entry ID to retrieve
        
        Returns:
            MemoryEntry if found, None otherwise
        
        Example:
            >>> entry = memory.get_entry(entry_id)
        """
        entry = self.entries.get(entry_id)
        
        if entry:
            entry.increment_access()
            if self.config["auto_save"]:
                self._save_to_disk()
        
        return entry
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search memory entries by content.
        
        Args:
            query: Search query string
            category: Filter by category (optional)
            min_importance: Minimum importance threshold
            limit: Maximum number of results
        
        Returns:
            List of matching memory entries
        
        Example:
            >>> results = memory.search("Python", category="preference")
        """
        query_lower = query.lower()
        
        matches = []
        for entry in self.entries.values():
            # Category filter
            if category and entry.category != category:
                continue
            
            # Importance filter
            if entry.importance < min_importance:
                continue
            
            # Content match (Phase-1: simple substring)
            if query_lower in entry.content.lower():
                matches.append(entry)
        
        # Sort by importance (descending)
        matches.sort(key=lambda e: e.importance, reverse=True)
        
        # Update access counts
        for entry in matches[:limit]:
            entry.increment_access()
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        return matches[:limit]
    
    def get_by_category(
        self,
        category: str,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve all entries in a category.
        
        Args:
            category: Category to filter
            limit: Maximum number of results
        
        Returns:
            List of memory entries
        
        Example:
            >>> prefs = memory.get_by_category("preference")
        """
        matches = [
            entry for entry in self.entries.values()
            if entry.category == category
        ]
        
        # Sort by timestamp (newest first)
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            return matches[:limit]
        return matches
    
    def get_most_important(self, limit: int = 10) -> List[MemoryEntry]:
        """
        Retrieve most important entries.
        
        Args:
            limit: Number of entries to retrieve
        
        Returns:
            List of memory entries sorted by importance
        
        Example:
            >>> important = memory.get_most_important(5)
        """
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.importance,
            reverse=True
        )
        return sorted_entries[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        
        Example:
            >>> stats = memory.get_statistics()
        """
        if not self.entries:
            return {
                "total_entries": 0,
                "categories": {},
                "avg_importance": 0.0
            }
        
        categories: Dict[str, int] = {}
        for entry in self.entries.values():
            categories[entry.category] = categories.get(entry.category, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "categories": categories,
            "avg_importance": sum(e.importance for e in self.entries.values()) / len(self.entries),
            "storage_path": str(self.storage_path),
            "config": self.config.copy()
        }
    
    # ========================================================================
    # PERSISTENCE OPERATIONS
    # ========================================================================
    
    def save(self) -> None:
        """
        Manually save memories to disk.
        
        Example:
            >>> memory.save()
        """
        self._save_to_disk()
    
    def clear(self) -> None:
        """
        Clear all memory entries.
        
        Example:
            >>> memory.clear()
        """
        count = len(self.entries)
        self.entries.clear()
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        self.logger.info(f"Cleared {count} memory entries")
    
    # ========================================================================
    # ASYNC API (Future-ready)
    # ========================================================================
    
    async def add_entry_async(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Async version of add_entry."""
        return await asyncio.to_thread(
            self.add_entry, content, category, importance, metadata
        )
    
    async def search_async(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Async version of search."""
        return await asyncio.to_thread(
            self.search, query, category, 0.0, limit
        )
    
    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================
    
    def _sanitize_input(self, content: str) -> str:
        """Sanitize input content (Security)."""
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        
        sanitized = content.strip()
        sanitized = sanitized.replace(' ', '')
        
        max_length = 50000
        if len(sanitized) > max_length:
            self.logger.warning(f"Content truncated from {len(sanitized)} to {max_length}")
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def _save_to_disk(self) -> None:
        """Save memories to disk storage."""
        try:
            storage_file = self.storage_path / "memories.json"
            
            # Convert entries to serializable format
            data = {
                "entries": {},
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            for entry_id, entry in self.entries.items():
                data["entries"][entry_id] = {
                    "id": entry.id,
                    "content": entry.content,
                    "category": entry.category,
                    "timestamp": entry.timestamp.isoformat(),
                    "importance": entry.importance,
                    "metadata": entry.metadata,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None
                }
            
            with open(storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved {len(self.entries)} entries to disk")
        
        except Exception as e:
            self.logger.error(f"Error saving to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load memories from disk storage."""
        try:
            storage_file = self.storage_path / "memories.json"
            
            if not storage_file.exists():
                self.logger.info("No existing memory file found")
                return
            
            with open(storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry_id, entry_data in data.get("entries", {}).items():
                entry = MemoryEntry(
                    id=entry_data["id"],
                    content=entry_data["content"],
                    category=entry_data["category"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    importance=entry_data["importance"],
                    metadata=entry_data.get("metadata", {}),
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]) 
                        if entry_data.get("last_accessed") else None
                )
                self.entries[entry_id] = entry
            
            self.logger.info(f"Loaded {len(self.entries)} entries from disk")
        
        except Exception as e:
            self.logger.error(f"Error loading from disk: {e}")
    
    def _prune_low_importance(self) -> None:
        """Remove low-importance entries when limit reached."""
        if len(self.entries) < self.config["max_entries"]:
            return
        
        # Sort by importance (ascending)
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].importance
        )
        
        # Remove bottom 10%
        remove_count = max(1, len(self.entries) // 10)
        
        for entry_id, _ in sorted_entries[:remove_count]:
            del self.entries[entry_id]
        
        self.logger.info(f"Pruned {remove_count} low-importance entries")


# ============================================================================
# FACTORY PATTERN (Dependency Injection)
# ============================================================================

def create_long_term_memory(
    config: Optional[LongTermConfig] = None,
    logger: Optional[logging.Logger] = None
) -> LongTermMemory:
    """
    Factory function to create LongTermMemory instance.
    
    Args:
        config: Optional memory configuration
        logger: Optional logger instance
    
    Returns:
        Configured LongTermMemory instance
    
    Example:
        >>> memory = create_long_term_memory({"storage_path": "./data"})
    """
    if config is None:
        config = {
            "storage_path": "./data/long_term_memory",
            "max_entries": 10000,
            "auto_save": True,
            "compression_enabled": False
        }
    
    return LongTermMemory(
        storage_path=config["storage_path"],
        max_entries=config["max_entries"],
        auto_save=config["auto_save"],
        compression_enabled=config["compression_enabled"],
        logger=logger
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "LongTermMemory",
    "MemoryEntry",
    "MemoryEntryDict",
    "LongTermConfig",
    "create_long_term_memory"
]


# ============================================================================
# TODO: FUTURE MODEL INTEGRATION (Phase-2+)
# ============================================================================

# TODO: Integrate Mistral-7B for:
# - Semantic search and retrieval
# - Automatic importance scoring
# - Content summarization for compression
# - Entity extraction and linking
# - Memory clustering by topic

# TODO: Add vector database integration (FAISS/Chroma)

# TODO: Implement memory consolidation (merging similar entries)

# TODO: Add encryption for sensitive memories