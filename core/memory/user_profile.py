"""
User Profile Module for AumCore_AI

This module manages user-specific data, preferences, behavior patterns,
and interaction history for personalized AI experiences.

Phase-1: Rule-based profile management with file storage.
Future: Integration with Mistral-7B for behavior prediction and preference learning.

File: core/memory/user_profile.py
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

# Type annotations
from typing import TypedDict


# ============================================================================
# CONFIGURATION & TYPES
# ============================================================================

class ProfileConfig(TypedDict):
    """Configuration for user profile."""
    storage_path: str
    auto_save: bool
    track_interactions: bool
    max_history_size: int


class InteractionRecord(TypedDict):
    """Type definition for interaction record."""
    timestamp: datetime
    interaction_type: str
    context: Dict[str, Any]
    outcome: str


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UserPreference:
    """Represents a single user preference."""
    
    key: str
    value: Any
    confidence: float = 1.0  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "explicit"  # explicit, inferred, learned
    
    def __post_init__(self) -> None:
        """Validate preference after initialization."""
        if not self.key:
            raise ValueError("Preference key cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class InteractionHistory:
    """Represents user interaction history."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    interaction_type: str = "query"
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> InteractionRecord:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "interaction_type": self.interaction_type,
            "context": self.context,
            "outcome": self.outcome
        }


# ============================================================================
# CORE USER PROFILE CLASS
# ============================================================================

class UserProfile:
    """
    Comprehensive user profile management system.
    
    Features:
    - Preference storage and retrieval
    - Interaction history tracking
    - Behavior pattern analysis
    - Interest and topic tracking
    - Session management
    - Personalization signals
    
    Attributes:
        config: Profile configuration
        user_id: Unique user identifier
        preferences: User preferences dictionary
        interaction_history: List of interactions
        interests: Set of user interests
        logger: Centralized logger instance
    
    Example:
        >>> profile = UserProfile(user_id="user123")
        >>> profile.set_preference("language", "English")
        >>> profile.add_interest("machine_learning")
        >>> history = profile.get_recent_interactions(10)
    """
    
    def __init__(
        self,
        user_id: str,
        storage_path: str = "./data/user_profiles",
        auto_save: bool = True,
        track_interactions: bool = True,
        max_history_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize user profile.
        
        Args:
            user_id: Unique user identifier
            storage_path: Directory path for profile storage
            auto_save: Automatically save after modifications
            track_interactions: Enable interaction tracking
            max_history_size: Maximum interaction history size
            logger: Optional logger instance
        
        Raises:
            ValueError: If user_id is empty
        """
        # Validation
        if not user_id:
            raise ValueError("User ID cannot be empty")
        
        # Configuration
        self.user_id = user_id
        self.config: ProfileConfig = {
            "storage_path": storage_path,
            "auto_save": auto_save,
            "track_interactions": track_interactions,
            "max_history_size": max_history_size
        }
        
        # Storage
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Profile data
        self.preferences: Dict[str, UserPreference] = {}
        self.interaction_history: List[InteractionHistory] = []
        self.interests: Set[str] = set()
        self.active_topics: Set[str] = set()
        
        # Metadata
        self.created_at: datetime = datetime.now()
        self.last_active: datetime = datetime.now()
        self.session_count: int = 0
        self.total_interactions: int = 0
        
        # Statistics
        self.interaction_stats: Dict[str, int] = {}
        
        # Logging
        self.logger = logger or self._setup_logger()
        
        # Load existing profile
        self._load_from_disk()
        
        self.logger.info(f"UserProfile initialized: user_id={user_id}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup centralized logger (Phase-1 fallback)."""
        logger = logging.getLogger("aumcore.memory.user_profile")
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
    # PUBLIC API - PREFERENCE MANAGEMENT
    # ========================================================================
    
    def set_preference(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit"
    ) -> None:
        """
        Set a user preference.
        
        Args:
            key: Preference key
            value: Preference value
            confidence: Confidence score (0.0-1.0)
            source: Source of preference (explicit, inferred, learned)
        
        Example:
            >>> profile.set_preference("theme", "dark", confidence=1.0)
        """
        try:
            preference = UserPreference(
                key=key,
                value=value,
                confidence=confidence,
                source=source
            )
            
            self.preferences[key] = preference
            
            self.logger.debug(f"Set preference: {key}={value}")
            
            if self.config["auto_save"]:
                self._save_to_disk()
        
        except Exception as e:
            self.logger.error(f"Error setting preference: {e}")
            raise
    
    def get_preference(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get a user preference value.
        
        Args:
            key: Preference key
            default: Default value if not found
        
        Returns:
            Preference value or default
        
        Example:
            >>> theme = profile.get_preference("theme", "light")
        """
        preference = self.preferences.get(key)
        return preference.value if preference else default
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """
        Get all preferences as dictionary.
        
        Returns:
            Dictionary of all preferences
        
        Example:
            >>> prefs = profile.get_all_preferences()
        """
        return {
            key: pref.value
            for key, pref in self.preferences.items()
        }
    
    def remove_preference(self, key: str) -> bool:
        """
        Remove a preference.
        
        Args:
            key: Preference key to remove
        
        Returns:
            True if removed, False if not found
        
        Example:
            >>> profile.remove_preference("old_setting")
        """
        if key in self.preferences:
            del self.preferences[key]
            
            if self.config["auto_save"]:
                self._save_to_disk()
            
            self.logger.debug(f"Removed preference: {key}")
            return True
        
        return False
    
    def update_preference_confidence(
        self,
        key: str,
        confidence: float
    ) -> bool:
        """
        Update confidence score for a preference.
        
        Args:
            key: Preference key
            confidence: New confidence score
        
        Returns:
            True if updated, False if not found
        
        Example:
            >>> profile.update_preference_confidence("language", 0.9)
        """
        preference = self.preferences.get(key)
        if not preference:
            return False
        
        preference.confidence = max(0.0, min(1.0, confidence))
        preference.last_updated = datetime.now()
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        return True
    
    # ========================================================================
    # PUBLIC API - INTERACTION TRACKING
    # ========================================================================
    
    def record_interaction(
        self,
        interaction_type: str,
        context: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a user interaction.
        
        Args:
            interaction_type: Type of interaction (query, command, etc.)
            context: Interaction context
            outcome: Interaction outcome (success, error, etc.)
            metadata: Additional metadata
        
        Example:
            >>> profile.record_interaction("query", {"topic": "AI"}, "success")
        """
        if not self.config["track_interactions"]:
            return
        
        try:
            interaction = InteractionHistory(
                interaction_type=interaction_type,
                context=context or {},
                outcome=outcome,
                metadata=metadata or {}
            )
            
            self.interaction_history.append(interaction)
            
            # Update statistics
            self.interaction_stats[interaction_type] =                 self.interaction_stats.get(interaction_type, 0) + 1
            
            self.total_interactions += 1
            self.last_active = datetime.now()
            
            # Maintain history size limit
            if len(self.interaction_history) > self.config["max_history_size"]:
                self.interaction_history.pop(0)
            
            self.logger.debug(f"Recorded interaction: {interaction_type}")
            
            if self.config["auto_save"]:
                self._save_to_disk()
        
        except Exception as e:
            self.logger.error(f"Error recording interaction: {e}")
    
    def get_recent_interactions(
        self,
        count: int = 10,
        interaction_type: Optional[str] = None
    ) -> List[InteractionRecord]:
        """
        Get recent interactions.
        
        Args:
            count: Number of interactions to retrieve
            interaction_type: Filter by interaction type
        
        Returns:
            List of interaction records
        
        Example:
            >>> recent = profile.get_recent_interactions(5, "query")
        """
        interactions = self.interaction_history
        
        if interaction_type:
            interactions = [
                i for i in interactions
                if i.interaction_type == interaction_type
            ]
        
        return [i.to_dict() for i in interactions[-count:]]
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """
        Get interaction statistics.
        
        Returns:
            Dictionary with interaction statistics
        
        Example:
            >>> stats = profile.get_interaction_statistics()
        """
        if not self.interaction_history:
            return {
                "total_interactions": 0,
                "by_type": {},
                "success_rate": 0.0
            }
        
        # Calculate success rate
        successful = sum(
            1 for i in self.interaction_history
            if i.outcome == "success"
        )
        success_rate = successful / len(self.interaction_history)
        
        return {
            "total_interactions": self.total_interactions,
            "tracked_interactions": len(self.interaction_history),
            "by_type": self.interaction_stats.copy(),
            "success_rate": success_rate,
            "last_active": self.last_active
        }
    
    # ========================================================================
    # PUBLIC API - INTEREST & TOPIC MANAGEMENT
    # ========================================================================
    
    def add_interest(self, interest: str) -> None:
        """
        Add a user interest.
        
        Args:
            interest: Interest to add
        
        Example:
            >>> profile.add_interest("machine_learning")
        """
        interest_lower = interest.lower().strip()
        if interest_lower:
            self.interests.add(interest_lower)
            
            if self.config["auto_save"]:
                self._save_to_disk()
            
            self.logger.debug(f"Added interest: {interest_lower}")
    
    def remove_interest(self, interest: str) -> bool:
        """
        Remove a user interest.
        
        Args:
            interest: Interest to remove
        
        Returns:
            True if removed, False if not found
        
        Example:
            >>> profile.remove_interest("old_topic")
        """
        interest_lower = interest.lower().strip()
        if interest_lower in self.interests:
            self.interests.remove(interest_lower)
            
            if self.config["auto_save"]:
                self._save_to_disk()
            
            return True
        
        return False
    
    def get_interests(self) -> List[str]:
        """
        Get all user interests.
        
        Returns:
            List of interests
        
        Example:
            >>> interests = profile.get_interests()
        """
        return list(self.interests)
    
    def has_interest(self, interest: str) -> bool:
        """
        Check if user has specific interest.
        
        Args:
            interest: Interest to check
        
        Returns:
            True if interest exists
        
        Example:
            >>> if profile.has_interest("python"):
        """
        return interest.lower().strip() in self.interests
    
    def set_active_topics(self, topics: List[str]) -> None:
        """
        Set currently active conversation topics.
        
        Args:
            topics: List of active topics
        
        Example:
            >>> profile.set_active_topics(["coding", "python"])
        """
        self.active_topics = {t.lower().strip() for t in topics if t}
        
        self.logger.debug(f"Set active topics: {self.active_topics}")
    
    def get_active_topics(self) -> List[str]:
        """
        Get currently active topics.
        
        Returns:
            List of active topics
        
        Example:
            >>> topics = profile.get_active_topics()
        """
        return list(self.active_topics)
    
    # ========================================================================
    # PUBLIC API - SESSION MANAGEMENT
    # ========================================================================
    
    def start_session(self) -> None:
        """
        Start a new user session.
        
        Example:
            >>> profile.start_session()
        """
        self.session_count += 1
        self.last_active = datetime.now()
        
        self.logger.info(f"Started session #{self.session_count}")
        
        if self.config["auto_save"]:
            self._save_to_disk()
    
    def end_session(self) -> None:
        """
        End current user session.
        
        Example:
            >>> profile.end_session()
        """
        self.last_active = datetime.now()
        
        self.logger.info("Ended session")
        
        if self.config["auto_save"]:
            self._save_to_disk()
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get session information.
        
        Returns:
            Dictionary with session info
        
        Example:
            >>> info = profile.get_session_info()
        """
        return {
            "user_id": self.user_id,
            "session_count": self.session_count,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "total_interactions": self.total_interactions
        }
    
    # ========================================================================
    # PUBLIC API - PERSONALIZATION
    # ========================================================================
    
    def get_personalization_context(self) -> Dict[str, Any]:
        """
        Get context for personalized responses.
        
        Returns:
            Dictionary with personalization context
        
        Example:
            >>> context = profile.get_personalization_context()
        """
        return {
            "preferences": self.get_all_preferences(),
            "interests": list(self.interests),
            "active_topics": list(self.active_topics),
            "recent_interactions": self.get_recent_interactions(5),
            "session_info": self.get_session_info()
        }
    
    def infer_preferences_from_interactions(self) -> None:
        """
        Infer preferences from interaction patterns (Phase-1: basic).
        
        Example:
            >>> profile.infer_preferences_from_interactions()
        """
        if len(self.interaction_history) < 10:
            return  # Not enough data
        
        # Analyze interaction types
        most_common_type = max(
            self.interaction_stats.items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]
        
        if most_common_type:
            self.set_preference(
                "preferred_interaction_type",
                most_common_type,
                confidence=0.7,
                source="inferred"
            )
        
        # Extract topics from context
        topic_counts: Dict[str, int] = {}
        for interaction in self.interaction_history[-50:]:
            context_topics = interaction.context.get("topics", [])
            for topic in context_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Add frequently mentioned topics as interests
        for topic, count in topic_counts.items():
            if count >= 5 and topic not in self.interests:
                self.add_interest(topic)
        
        self.logger.info("Inferred preferences from interactions")
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self) -> None:
        """
        Manually save profile to disk.
        
        Example:
            >>> profile.save()
        """
        self._save_to_disk()
    
    def clear_history(self) -> None:
        """
        Clear interaction history.
        
        Example:
            >>> profile.clear_history()
        """
        count = len(self.interaction_history)
        self.interaction_history.clear()
        self.interaction_stats.clear()
        
        if self.config["auto_save"]:
            self._save_to_disk()
        
        self.logger.info(f"Cleared {count} interaction records")
    
    def _save_to_disk(self) -> None:
        """Save profile to disk storage."""
        try:
            profile_file = self.storage_path / f"{self.user_id}.json"
            
            data = {
                "user_id": self.user_id,
                "created_at": self.created_at.isoformat(),
                "last_active": self.last_active.isoformat(),
                "session_count": self.session_count,
                "total_interactions": self.total_interactions,
                "preferences": {},
                "interests": list(self.interests),
                "active_topics": list(self.active_topics),
                "interaction_history": [],
                "interaction_stats": self.interaction_stats,
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            
            # Serialize preferences
            for key, pref in self.preferences.items():
                data["preferences"][key] = {
                    "value": pref.value,
                    "confidence": pref.confidence,
                    "last_updated": pref.last_updated.isoformat(),
                    "source": pref.source
                }
            
            # Serialize interaction history
            for interaction in self.interaction_history:
                data["interaction_history"].append({
                    "timestamp": interaction.timestamp.isoformat(),
                    "interaction_type": interaction.interaction_type,
                    "context": interaction.context,
                    "outcome": interaction.outcome,
                    "metadata": interaction.metadata
                })
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved profile: {self.user_id}")
        
        except Exception as e:
            self.logger.error(f"Error saving profile: {e}")
    
    def _load_from_disk(self) -> None:
        """Load profile from disk storage."""
        try:
            profile_file = self.storage_path / f"{self.user_id}.json"
            
            if not profile_file.exists():
                self.logger.info(f"No existing profile found for {self.user_id}")
                return
            
            with open(profile_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load metadata
            self.created_at = datetime.fromisoformat(data["created_at"])
            self.last_active = datetime.fromisoformat(data["last_active"])
            self.session_count = data.get("session_count", 0)
            self.total_interactions = data.get("total_interactions", 0)
            
            # Load preferences
            for key, pref_data in data.get("preferences", {}).items():
                self.preferences[key] = UserPreference(
                    key=key,
                    value=pref_data["value"],
                    confidence=pref_data["confidence"],
                    last_updated=datetime.fromisoformat(pref_data["last_updated"]),
                    source=pref_data["source"]
                )
            
            # Load interests
            self.interests = set(data.get("interests", []))
            self.active_topics = set(data.get("active_topics", []))
            
            # Load interaction history
            for interaction_data in data.get("interaction_history", []):
                interaction = InteractionHistory(
                    timestamp=datetime.fromisoformat(interaction_data["timestamp"]),
                    interaction_type=interaction_data["interaction_type"],
                    context=interaction_data.get("context", {}),
                    outcome=interaction_data.get("outcome", "success"),
                    metadata=interaction_data.get("metadata", {})
                )
                self.interaction_history.append(interaction)
            
            # Load statistics
            self.interaction_stats = data.get("interaction_stats", {})
            
            self.logger.info(f"Loaded profile: {self.user_id}")
        
        except Exception as e:
            self.logger.error(f"Error loading profile: {e}")
    
    # ========================================================================
    # ASYNC API (Future-ready)
    # ========================================================================
    
    async def set_preference_async(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0
    ) -> None:
        """Async version of set_preference."""
        await asyncio.to_thread(
            self.set_preference, key, value, confidence
        )
    
    async def record_interaction_async(
        self,
        interaction_type: str,
        context: Optional[Dict[str, Any]] = None,
        outcome: str = "success"
    ) -> None:
        """Async version of record_interaction."""
        await asyncio.to_thread(
            self.record_interaction, interaction_type, context, outcome
        )


# ============================================================================
# FACTORY PATTERN (Dependency Injection)
# ============================================================================

def create_user_profile(
    user_id: str,
    config: Optional[ProfileConfig] = None,
    logger: Optional[logging.Logger] = None
) -> UserProfile:
    """
    Factory function to create UserProfile instance.
    
    Args:
        user_id: Unique user identifier
        config: Optional profile configuration
        logger: Optional logger instance
    
    Returns:
        Configured UserProfile instance
    
    Example:
        >>> profile = create_user_profile("user123", {"auto_save": True})
    """
    if config is None:
        config = {
            "storage_path": "./data/user_profiles",
            "auto_save": True,
            "track_interactions": True,
            "max_history_size": 1000
        }
    
    return UserProfile(
        user_id=user_id,
        storage_path=config["storage_path"],
        auto_save=config["auto_save"],
        track_interactions=config["track_interactions"],
        max_history_size=config["max_history_size"],
        logger=logger
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "UserProfile",
    "UserPreference",
    "InteractionHistory",
    "ProfileConfig",
    "InteractionRecord",
    "create_user_profile"
]


# ============================================================================
# TODO: FUTURE MODEL INTEGRATION (Phase-2+)
# ============================================================================

# TODO: Integrate Mistral-7B for:
# - Automatic preference learning from conversations
# - Behavior pattern prediction
# - Interest extraction from text
# - Personalized response generation

# TODO: Add collaborative filtering for recommendations

# TODO: Implement privacy controls and data anonymization

# TODO: Add multi-user profile support with shared contexts

# TODO: Implement profile merging and migration tools