"""
================================================================================
AumCore_AI - Topic Tracker
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: topic_tracker.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    Topic Tracker. This module keeps track of the "current topic" of the
    conversation over time, using simple heuristics and signals from NLU
    and the AutoTopicSwitchDetector.

    Phase-1 constraints:
        - No ML models, no embeddings, no vector search
        - No long-term semantic memory
        - Pure rule-based topic updates + small ring buffer history
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Maintain the current topic label and a short history of past topics
        - Update topic deterministically based on:
            * NLU topic info (if provided)
            * AutoTopicSwitchDetector outputs (optional)
            * Explicit override commands (from orchestrator)
        - Expose a clean TopicState snapshot for other modules:
            * current_topic
            * previous_topic
            * topic_history
            * last_updated_at (logical step counter)

    Future phases may add:
        - Time-based decay and topic fading
        - Multi-topic stacking (foreground/background topics)
        - User-profile-aware topic persistence

================================================================================
"""

# ==============================================================================
# Chunk-1: Imports, Metadata, Logging Setup
# ==============================================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import yaml

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "topic_tracker"
__description__ = (
    "Phase-1 rule-based Topic Tracker with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TopicTracker")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class TopicTrackerError(Exception):
    """
    Base class for all Topic Tracker errors.
    """

    def __init__(self, message: str, *, code: str = "TOPIC_TRACKER_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidTopicError(TopicTrackerError):
    """
    Raised when an invalid topic label is used.
    """

    def __init__(self, message: str = "Invalid topic label."):
        super().__init__(message, code="INVALID_TOPIC")


class InvalidConfigError(TopicTrackerError):
    """
    Raised when YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid Topic Tracker configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class TopicTrackerConfig:
    """
    Configuration for the Topic Tracker.

    Attributes
    ----------
    default_topic : str
        Topic used when no topic is known or when resetting.
    max_history_size : int
        Maximum number of historical topics to keep.
    allow_unknown_topics : bool
        Whether to allow arbitrary topic strings (if False, use known_topics).
    known_topics : List[str]
        Optional whitelist of known topics (lowercase).
    """

    default_topic: str = "general"
    max_history_size: int = 20
    allow_unknown_topics: bool = True
    known_topics: List[str] = field(default_factory=list)


@dataclass
class TopicState:
    """
    Represents the current topic tracking state.

    Attributes
    ----------
    current_topic : Optional[str]
        Currently active topic label (or None if unknown).
    previous_topic : Optional[str]
        Most recent topic before current_topic changed.
    history : List[str]
        Recent topic labels, most recent last (excluding None).
    last_updated_step : int
        Logical "step" counter incremented each time topic changes.
    """

    current_topic: Optional[str] = None
    previous_topic: Optional[str] = None
    history: List[str] = field(default_factory=list)
    last_updated_step: int = 0


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for Topic Tracker, if present.

    Expected structure (Phase-1, optional):

        default_topic: "general"
        max_history_size: 20
        allow_unknown_topics: true
        known_topics:
          - "general"
          - "shopping"
          - "travel"
          - "support"
          - "coding"

    Parameters
    ----------
    path : Optional[str]
        Path to YAML config file.

    Returns
    -------
    Dict[str, Any]
        Parsed config dict or empty dict on error.
    """
    if not path:
        logger.info("No config path provided for TopicTracker. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Topic tracker config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded Topic Tracker config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load Topic Tracker YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_topic_name(topic: Any) -> str:
    """
    Normalize topic names to lowercase string without surrounding whitespace.
    """
    if topic is None:
        return ""
    if not isinstance(topic, str):
        topic = str(topic)
    cleaned = topic.strip().lower()
    return cleaned


def sanitize_topic_list(values: Any) -> List[str]:
    """
    Normalize a list of topic names to lowercase list.
    """
    if values is None:
        return []
    if not isinstance(values, list):
        logger.warning(
            "Invalid known_topics config (not a list). Ignoring and using empty list."
        )
        return []
    result: List[str] = []
    for v in values:
        cleaned = sanitize_topic_name(v)
        if cleaned:
            result.append(cleaned)
    return result


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def validate_topic_label(topic: Any) -> str:
    """
    Validate a topic label as a non-empty string and return normalized value.
    """
    cleaned = sanitize_topic_name(topic)
    if not cleaned:
        raise InvalidTopicError("Topic label must be a non-empty string.")
    return cleaned


def _validate_positive_int(value: Any, default: int, name: str) -> int:
    try:
        ivalue = int(value)
    except Exception:
        logger.warning(
            "Invalid %s value %r. Falling back to default %d.", name, value, default
        )
        return default

    if ivalue <= 0:
        logger.warning("%s must be > 0. Using default %d.", name, default)
        return default

    return ivalue


def _validate_bool(value: Any, default: bool, name: str) -> bool:
    if isinstance(value, bool):
        return value
    logger.warning(
        "Invalid %s value %r. Falling back to default %s.", name, value, default
    )
    return default


def _sanitize_default_topic(value: Any, known_topics: List[str]) -> str:
    candidate = sanitize_topic_name(value)
    if not candidate:
        candidate = "general"

    if known_topics and candidate not in known_topics:
        logger.warning(
            "default_topic %r not in known_topics list. Keeping it but note mismatch.",
            candidate,
        )
    return candidate


# ==============================================================================
# Chunk-1 + Chunk-2: Topic Tracker (Phase-1 Only)
# ==============================================================================


class TopicTracker:
    """
    Phase-1 rule-based Topic Tracker.

    High-level behavior:
        - Maintains a TopicState in memory.
        - Can be updated by:
            * explicit topic override
            * NLU topic info (current_topic, is_topic_switch)
            * direct set/reset commands

    The tracker does not infer topics by itself (no NLP); it relies on
    external modules to propose topics and on simple rules to accept them.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the TopicTracker.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing TopicTracker (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: TopicTrackerConfig = self._build_config_from_raw(raw_config)

        # Start with default_topic as current_topic
        self._state: TopicState = TopicState(
            current_topic=self._config.default_topic,
            previous_topic=None,
            history=[self._config.default_topic] if self._config.default_topic else [],
            last_updated_step=0,
        )

        logger.info(
            "TopicTracker configured: default_topic=%r, max_history_size=%d, allow_unknown_topics=%s, known_topics=%d",
            self._config.default_topic,
            self._config.max_history_size,
            self._config.allow_unknown_topics,
            len(self._config.known_topics),
        )

        logger.info("TopicTracker initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> TopicTrackerConfig:
        known_topics = sanitize_topic_list(raw.get("known_topics", []))
        allow_unknown = _validate_bool(
            raw.get("allow_unknown_topics", True),
            default=True,
            name="allow_unknown_topics",
        )
        max_history_size = _validate_positive_int(
            raw.get("max_history_size", 20),
            default=20,
            name="max_history_size",
        )

        default_topic_raw = raw.get("default_topic", "general")
        default_topic = _sanitize_default_topic(default_topic_raw, known_topics)

        return TopicTrackerConfig(
            default_topic=default_topic,
            max_history_size=max_history_size,
            allow_unknown_topics=allow_unknown,
            known_topics=known_topics,
        )

    # ------------------------------ Internal Utils ---------------------------

    def _is_known_topic(self, topic: str) -> bool:
        """
        Check if topic is in known_topics or unknowns are allowed.
        """
        if not topic:
            return False

        if self._config.allow_unknown_topics:
            return True

        return topic in self._config.known_topics

    def _push_history(self, topic: str) -> None:
        """
        Append topic to history, enforcing max_history_size.
        """
        if not topic:
            return

        self._state.history.append(topic)
        if len(self._state.history) > self._config.max_history_size:
            overflow = len(self._state.history) - self._config.max_history_size
            if overflow > 0:
                self._state.history = self._state.history[overflow:]

    def _update_state_with_topic(self, new_topic: str) -> None:
        """
        Update TopicState with new topic value.
        """
        new_topic_clean = sanitize_topic_name(new_topic)
        if not new_topic_clean:
            return

        current = self._state.current_topic or ""
        if current == new_topic_clean:
            # No change
            return

        prev = self._state.current_topic
        self._state.previous_topic = prev
        self._state.current_topic = new_topic_clean
        self._state.last_updated_step += 1
        self._push_history(new_topic_clean)

        logger.info(
            "TopicTracker: topic changed from %r to %r (step=%d)",
            prev,
            new_topic_clean,
            self._state.last_updated_step,
        )

    # --------------------------- Public Interface ----------------------------

    def get_state(self) -> TopicState:
        """
        Return a copy-like snapshot of the current TopicState.

        NOTE: The returned TopicState object is the live state, but Phase-1
        assumes single-threaded orchestrator usage, so no deep copy is needed.
        """
        return self._state

    def get_current_topic(self) -> Optional[str]:
        """
        Convenience method: return current_topic.
        """
        return self._state.current_topic

    def get_previous_topic(self) -> Optional[str]:
        """
        Convenience method: return previous_topic.
        """
        return self._state.previous_topic

    def get_topic_history(self) -> List[str]:
        """
        Return a shallow copy of topic history.
        """
        return list(self._state.history)

    # ------------------------- Explicit Control Methods ----------------------

    def reset(self) -> None:
        """
        Reset the topic tracker to default_topic and clear history.
        """
        default_topic = self._config.default_topic or None
        history = [default_topic] if default_topic else []

        self._state = TopicState(
            current_topic=default_topic,
            previous_topic=None,
            history=history,
            last_updated_step=0,
        )

        logger.info("TopicTracker: reset to default_topic=%r", default_topic)

    def force_set_topic(self, topic: Any) -> Optional[str]:
        """
        Forcefully set the current topic, bypassing NLU logic.

        Parameters
        ----------
        topic : Any
            New topic label.

        Returns
        -------
        Optional[str]
            Final applied topic label, or None if rejected.
        """
        try:
            cleaned = validate_topic_label(topic)
        except InvalidTopicError as exc:
            logger.error("TopicTracker.force_set_topic: %s", exc)
            return None

        if not self._is_known_topic(cleaned):
            logger.warning(
                "TopicTracker.force_set_topic: topic %r not in known_topics and allow_unknown_topics=False. Ignoring.",
                cleaned,
            )
            return None

        self._update_state_with_topic(cleaned)
        return self._state.current_topic

    # -------------------------- Update via NLU Signals -----------------------

    def update_from_nlu_topic(self, nlu_topic_info: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Update topic based on NLU topic info payload.

        Expected shape:

            nlu_topic_info = {
                "current_topic": "<topic_name or None>",
                "is_topic_switch": <bool or None>
            }

        Rules (Phase-1):
            - If current_topic is missing or empty => no update.
            - Normalize topic label.
            - If topic is not allowed (known_topics & !allow_unknown): ignore.
            - If is_topic_switch is True:
                * Always update to new topic.
              Else:
                * If tracker has no current_topic (None): update.
                * If new_topic != current_topic: update (soft switch).
        """
        if not isinstance(nlu_topic_info, dict):
            return self._state.current_topic

        raw_topic = nlu_topic_info.get("current_topic")
        new_topic = sanitize_topic_name(raw_topic)
        if not new_topic:
            return self._state.current_topic

        if not self._is_known_topic(new_topic):
            logger.info(
                "TopicTracker.update_from_nlu_topic: ignoring unknown topic %r.",
                new_topic,
            )
            return self._state.current_topic

        is_switch = nlu_topic_info.get("is_topic_switch", None)
        if is_switch not in (True, False):
            is_switch = None

        if is_switch is True:
            self._update_state_with_topic(new_topic)
            return self._state.current_topic

        # If no explicit switch flag:
        # - If no current topic, set it.
        # - If different from current, also set (soft switch).
        current = self._state.current_topic or ""
        if not current or current != new_topic:
            self._update_state_with_topic(new_topic)

        return self._state.current_topic

    def update_from_detector(
        self,
        detector_output: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Update topic based on AutoTopicSwitchDetector-style output.

        Expected shape:

            detector_output = {
                "new_topic": "<topic_name>",
                "is_topic_switch": <bool>
            }

        Rules (Phase-1):
            - If new_topic is missing/empty => no update.
            - If is_topic_switch is False => set only if no current topic.
            - If is_topic_switch is True => always update to new_topic.
        """
        if not isinstance(detector_output, dict):
            return self._state.current_topic

        raw_topic = detector_output.get("new_topic")
        new_topic = sanitize_topic_name(raw_topic)
        if not new_topic:
            return self._state.current_topic

        if not self._is_known_topic(new_topic):
            logger.info(
                "TopicTracker.update_from_detector: ignoring unknown topic %r.",
                new_topic,
            )
            return self._state.current_topic

        is_switch = detector_output.get("is_topic_switch", None)
        if is_switch is True:
            self._update_state_with_topic(new_topic)
            return self._state.current_topic

        if is_switch is False:
            if not self._state.current_topic:
                self._update_state_with_topic(new_topic)
            return self._state.current_topic

        # If is_switch is None (not specified): apply soft logic similar to NLU
        current = self._state.current_topic or ""
        if not current or current != new_topic:
            self._update_state_with_topic(new_topic)
        return self._state.current_topic

    # ------------------------------ Introspection -----------------------------

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of configuration and current state.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "config": {
                "default_topic": self._config.default_topic,
                "max_history_size": self._config.max_history_size,
                "allow_unknown_topics": self._config.allow_unknown_topics,
                "known_topics": list(self._config.known_topics),
            },
            "state": {
                "current_topic": self._state.current_topic,
                "previous_topic": self._state.previous_topic,
                "history": list(self._state.history),
                "last_updated_step": self._state.last_updated_step,
            },
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for TopicTracker...")

    # Example in-memory configuration (when no YAML is provided)
    raw_config = {
        "default_topic": "general",
        "max_history_size": 5,
        "allow_unknown_topics": False,
        "known_topics": ["general", "shopping", "travel", "support", "coding"],
    }

    tracker = TopicTracker(config_path=None)
    tracker._config = tracker._build_config_from_raw(raw_config)
    tracker.reset()

    print("Initial state:", tracker.debug_state()["state"])

    # Simulate NLU topic updates
    nlu_updates = [
        {"current_topic": "shopping", "is_topic_switch": True},
        {"current_topic": "shopping", "is_topic_switch": False},
        {"current_topic": "travel", "is_topic_switch": True},
        {"current_topic": "support", "is_topic_switch": None},
        {"current_topic": "coding", "is_topic_switch": True},
    ]

    for idx, info in enumerate(nlu_updates, start=1):
        new_topic = tracker.update_from_nlu_topic(info)
        print(f"\nAfter NLU update #{idx} ({info}):")
        print("  current_topic :", new_topic)
        print("  state         :", tracker.debug_state()["state"])

    # Explicit force set
    print("\nForcing topic to 'shopping'...")
    tracker.force_set_topic("shopping")
    print("State after force_set_topic:", tracker.debug_state()["state"])

    # Try unknown topic with allow_unknown_topics=False
    print("\nTrying to force unknown topic 'sports' (should be ignored)...")
    tracker.force_set_topic("sports")
    print("State after unknown topic:", tracker.debug_state()["state"])

    logger.info("Phase-1 manual test for TopicTracker completed.")
