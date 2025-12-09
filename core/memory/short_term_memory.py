"""
Short-Term Memory Module for AumCore_AI

This module manages temporary conversation context and recent interactions.
Implements a sliding window approach for maintaining recent dialogue history.

Phase-1: Rule-based implementation with zero model loading.
Future: Integration with Mistral-7B for semantic context extraction.

File: core/memory/short_term_memory.py
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod

# Type annotations
from typing import TypedDict


# ============================================================================
# CONFIGURATION & TYPES
# ============================================================================

class MessageDict(TypedDict):
    """Type definition for a message entry."""
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]


class MemoryConfig(TypedDict):
    """Configuration for short-term memory."""
    max_messages: int
    max_age_minutes: int
    enable_compression: bool
    compression_threshold: int


# ============================================================================
# ABSTRACT INTERFACES (SOLID: Dependency Inversion)
# ============================================================================

class MemoryStorageProtocol(Protocol):
    """Protocol for memory storage backends."""
    
    def add(self, message: MessageDict) -> None:
        """Add a message to storage."""
        ...
    
    def get_recent(self, count: int) -> List[MessageDict]:
        """Retrieve recent messages."""
        ...
    
    def clear(self) -> None:
        """Clear all stored messages."""
        ...


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ConversationMessage:
    """Represents a single conversation message with metadata."""
    
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    
    def to_dict(self) -> MessageDict:
        """Convert to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def __post_init__(self) -> None:
        """Validate message after initialization."""
        if not self.role or not isinstance(self.role, str):
            raise ValueError("Role must be a non-empty string")
        if not isinstance(self.content, str):
            raise ValueError("Content must be a string")


# ============================================================================
# CORE SHORT-TERM MEMORY CLASS
# ============================================================================

class ShortTermMemory:
    """
    Manages short-term conversation memory with configurable retention.
    
    Features:
    - Sliding window of recent messages
    - Automatic expiry based on age
    - Token-aware compression (Phase-1: character count)
    - Async-ready operations
    - Type-safe interface
    
    Attributes:
        config: Memory configuration parameters
        messages: Deque of conversation messages
        logger: Centralized logger instance
    
    Example:
        >>> memory = ShortTermMemory(max_messages=10)
        >>> memory.add_message("user", "Hello!")
        >>> recent = memory.get_recent_messages(5)
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        max_age_minutes: int = 30,
        enable_compression: bool = True,
        compression_threshold: int = 50,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize short-term memory.
        
        Args:
            max_messages: Maximum number of messages to retain
            max_age_minutes: Maximum age of messages in minutes
            enable_compression: Enable automatic message compression
            compression_threshold: Number of messages before compression
            logger: Optional logger instance
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validation
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if max_age_minutes <= 0:
            raise ValueError("max_age_minutes must be positive")
        
        # Configuration
        self.config: MemoryConfig = {
            "max_messages": max_messages,
            "max_age_minutes": max_age_minutes,
            "enable_compression": enable_compression,
            "compression_threshold": compression_threshold
        }
        
        # Storage: Deque for O(1) append/pop operations
        self.messages: Deque[ConversationMessage] = deque(maxlen=max_messages)
        
        # Logging (centralized)
        self.logger = logger or self._setup_logger()
        
        self.logger.info(
            f"ShortTermMemory initialized: max_messages={max_messages}, "
            f"max_age={max_age_minutes}min"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup centralized logger (Phase-1 fallback)."""
        logger = logging.getLogger("aumcore.memory.short_term")
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
    # PUBLIC API
    # ========================================================================
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to short-term memory.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata dictionary
        
        Raises:
            ValueError: If role or content is invalid
        
        Example:
            >>> memory.add_message("user", "What is AI?")
        """
        try:
            # Input validation (Security)
            sanitized_content = self._sanitize_input(content)
            
            message = ConversationMessage(
                role=role,
                content=sanitized_content,
                metadata=metadata or {},
                token_count=len(sanitized_content)  # Phase-1: char count
            )
            
            self.messages.append(message)
            self.logger.debug(f"Added message: role={role}, length={len(content)}")
            
            # Auto-cleanup expired messages
            self._cleanup_expired()
            
            # Auto-compression if threshold reached
            if (self.config["enable_compression"] and 
                len(self.messages) >= self.config["compression_threshold"]):
                self._compress_old_messages()
        
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")
            raise
    
    def get_recent_messages(
        self,
        count: Optional[int] = None,
        include_system: bool = True
    ) -> List[MessageDict]:
        """
        Retrieve recent messages from memory.
        
        Args:
            count: Number of messages to retrieve (None = all)
            include_system: Include system messages
        
        Returns:
            List of message dictionaries
        
        Example:
            >>> recent = memory.get_recent_messages(5)
        """
        self._cleanup_expired()
        
        messages = list(self.messages)
        
        if not include_system:
            messages = [m for m in messages if m.role != "system"]
        
        if count is not None:
            messages = messages[-count:]
        
        return [msg.to_dict() for msg in messages]
    
    def get_context_string(
        self,
        max_chars: int = 2000,
        separator: str = "
"
    ) -> str:
        """
        Get recent conversation as formatted string.
        
        Args:
            max_chars: Maximum character length
            separator: Message separator
        
        Returns:
            Formatted conversation string
        
        Example:
            >>> context = memory.get_context_string(1000)
        """
        messages = self.get_recent_messages()
        
        context_parts = []
        total_chars = 0
        
        for msg in reversed(messages):
            msg_str = f"{msg['role']}: {msg['content']}"
            msg_len = len(msg_str) + len(separator)
            
            if total_chars + msg_len > max_chars:
                break
            
            context_parts.insert(0, msg_str)
            total_chars += msg_len
        
        return separator.join(context_parts)
    
    def clear(self) -> None:
        """
        Clear all messages from memory.
        
        Example:
            >>> memory.clear()
        """
        count = len(self.messages)
        self.messages.clear()
        self.logger.info(f"Cleared {count} messages from memory")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        
        Example:
            >>> stats = memory.get_statistics()
            >>> print(stats["total_messages"])
        """
        messages = list(self.messages)
        
        return {
            "total_messages": len(messages),
            "total_chars": sum(m.token_count for m in messages),
            "oldest_message": messages[0].timestamp if messages else None,
            "newest_message": messages[-1].timestamp if messages else None,
            "config": self.config.copy()
        }
    
    # ========================================================================
    # ASYNC API (Future-ready)
    # ========================================================================
    
    async def add_message_async(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Async version of add_message.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        await asyncio.to_thread(self.add_message, role, content, metadata)
    
    async def get_recent_messages_async(
        self,
        count: Optional[int] = None
    ) -> List[MessageDict]:
        """
        Async version of get_recent_messages.
        
        Args:
            count: Number of messages to retrieve
        
        Returns:
            List of message dictionaries
        """
        return await asyncio.to_thread(self.get_recent_messages, count)
    
    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================
    
    def _sanitize_input(self, content: str) -> str:
        """
        Sanitize input content (Security).
        
        Args:
            content: Raw input content
        
        Returns:
            Sanitized content
        """
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        
        # Phase-1: Basic sanitization
        sanitized = content.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace(' ', '')
        
        # Limit length (DoS prevention)
        max_length = 10000
        if len(sanitized) > max_length:
            self.logger.warning(f"Content truncated from {len(sanitized)} to {max_length}")
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def _cleanup_expired(self) -> None:
        """Remove messages older than max_age."""
        cutoff_time = datetime.now() - timedelta(
            minutes=self.config["max_age_minutes"]
        )
        
        original_count = len(self.messages)
        
        # Remove expired messages from left
        while self.messages and self.messages[0].timestamp < cutoff_time:
            self.messages.popleft()
        
        removed = original_count - len(self.messages)
        if removed > 0:
            self.logger.debug(f"Removed {removed} expired messages")
    
    def _compress_old_messages(self) -> None:
        """
        Compress older messages (Phase-1: summarization placeholder).
        
        TODO: Phase-2 integration with Mistral-7B for semantic compression
        """
        if len(self.messages) < self.config["compression_threshold"]:
            return
        
        # Phase-1: Simple truncation of older messages
        messages_to_compress = len(self.messages) // 4
        
        for i in range(messages_to_compress):
            if len(self.messages) > 0:
                old_msg = self.messages.popleft()
                
                # Create compressed placeholder
                compressed = ConversationMessage(
                    role="system",
                    content=f"[Compressed: {old_msg.role} message from {old_msg.timestamp}]",
                    metadata={"compressed": True, "original_length": old_msg.token_count}
                )
                self.messages.appendleft(compressed)
        
        self.logger.info(f"Compressed {messages_to_compress} old messages")


# ============================================================================
# FACTORY PATTERN (Dependency Injection)
# ============================================================================

def create_short_term_memory(
    config: Optional[MemoryConfig] = None,
    logger: Optional[logging.Logger] = None
) -> ShortTermMemory:
    """
    Factory function to create ShortTermMemory instance.
    
    Args:
        config: Optional memory configuration
        logger: Optional logger instance
    
    Returns:
        Configured ShortTermMemory instance
    
    Example:
        >>> memory = create_short_term_memory({"max_messages": 15})
    """
    if config is None:
        config = {
            "max_messages": 20,
            "max_age_minutes": 30,
            "enable_compression": True,
            "compression_threshold": 50
        }
    
    return ShortTermMemory(
        max_messages=config["max_messages"],
        max_age_minutes=config["max_age_minutes"],
        enable_compression=config["enable_compression"],
        compression_threshold=config["compression_threshold"],
        logger=logger
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "ShortTermMemory",
    "ConversationMessage",
    "MessageDict",
    "MemoryConfig",
    "create_short_term_memory"
]


# ============================================================================
# TODO: FUTURE MODEL INTEGRATION (Phase-2+)
# ============================================================================

# TODO: Integrate Mistral-7B for:
# - Semantic message compression
# - Context importance scoring
# - Automatic summarization
# - Entity extraction from messages
# - Sentiment tracking

# TODO: Add FAISS integration for semantic search in conversation history

# TODO: Implement conversation threading (multi-topic tracking)