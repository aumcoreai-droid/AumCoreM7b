"""
================================================================================
AumCore_AI - Auto Topic Switch Detector
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: auto_topic_switch_detector.py

Description:
    Phase-1 enterprise-grade implementation of a rule-based topic switch detector.
    This module analyzes the current user message and previous topic context to
    determine whether the user has switched topics.

    Phase-1 constraints:
        - Pure rule-based logic
        - No embeddings, no ML models, no vector search
        - Deterministic keyword + heuristic matching
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Maintain a configurable set of topics and their associated keywords.
        - Classify the topic of a given user message.
        - Decide whether a topic switch has occurred compared to a previous topic.
        - Provide a simple, inspectable debug state for higher layers.

    This file is intentionally verbose and structured to act as a stable
    foundation for later phases (Chunk-3 to Chunk-8), where more sophisticated
    NLP, embeddings, and user modeling may be added.

================================================================================
"""

# ==============================================================================
# Chunk-1: Imports, Metadata, Logging Setup
# ==============================================================================

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import yaml

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "auto_topic_switch_detector"
__description__ = (
    "Phase-1 rule-based Auto Topic Switch Detector with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("AutoTopicSwitchDetector")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class TopicSwitchError(Exception):
    """
    Base class for all auto topic switch detector errors.

    Using a dedicated base exception keeps the module's error handling
    encapsulated and easier to integrate with higher-level orchestration.
    """

    def __init__(self, message: str, *, code: str = "TOPIC_SWITCH_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidTextError(TopicSwitchError):
    """
    Raised when provided text is missing or invalid.
    """

    def __init__(self, message: str = "Text must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_TEXT")


class InvalidTopicError(TopicSwitchError):
    """
    Raised when provided topic label is missing or invalid.
    """

    def __init__(self, message: str = "Topic must be a valid non-empty string."):
        super().__init__(message, code="INVALID_TOPIC")


class InvalidConfigError(TopicSwitchError):
    """
    Raised when YAML config structure is invalid or unusable.
    """

    def __init__(self, message: str = "Invalid topic switch configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class TopicDefinition:
    """
    Represents a single topic configuration.

    Attributes
    ----------
    name : str
        Canonical topic name (normalized to lowercase).
    keywords : List[str]
        List of keywords associated with this topic.
    description : Optional[str]
        Optional human-readable description.
    """

    name: str
    keywords: List[str]
    description: Optional[str] = None


@dataclass
class TopicSwitchRules:
    """
    Represents the rules and thresholds for topic switch detection.

    Attributes
    ----------
    topics : Dict[str, TopicDefinition]
        Mapping of topic name -> TopicDefinition.
    min_keyword_overlap : int
        Minimum number of overlapping keywords for a topic to be considered.
    switch_threshold : float
        Maximum allowed score difference before treating it as a topic switch.
        In Phase-1, we use a simple overlap ratio as score per topic.
    """

    topics: Dict[str, TopicDefinition]
    min_keyword_overlap: int = 1
    switch_threshold: float = 0.3


@dataclass
class TopicSwitchConfig:
    """
    Represents the overall configuration for the topic switch detector.

    Attributes
    ----------
    rules : TopicSwitchRules
        Configured topic detection rules.
    default_topic : str
        Fallback topic name when no match is found.
    """

    rules: TopicSwitchRules
    default_topic: str = "general"


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a YAML configuration file if provided and exists.

    Expected YAML structure (Phase-1, optional):

        default_topic: "general"
        min_keyword_overlap: 1
        switch_threshold: 0.3
        topics:
          shopping:
            keywords: ["buy", "order", "cart", "price"]
            description: "Shopping related queries"
          travel:
            keywords: ["flight", "hotel", "ticket", "train"]
            description: "Travel planning and booking"
          support:
            keywords: ["error", "issue", "bug", "help"]
            description: "Technical support or troubleshooting"

    Parameters
    ----------
    path : Optional[str]
        Path to YAML configuration, or None.

    Returns
    -------
    Dict[str, Any]
        Parsed dictionary or empty dict on failure.
    """
    if not path:
        logger.info("No config path provided for topic switch detector. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning("Topic switch config file not found at '%s'. Using defaults.", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded topic switch config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load topic switch YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_text(text: str) -> str:
    """
    Sanitize user text for topic analysis.

    - Ensures lowercase.
    - Normalizes whitespace.
    - Keeps alphanumeric content for keyword matching.

    Parameters
    ----------
    text : str
        Raw user text.

    Returns
    -------
    str
        Sanitized text.
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")

    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def sanitize_topic_name(topic: str) -> str:
    """
    Normalize topic names to a canonical form for dictionary keys.

    Parameters
    ----------
    topic : str
        Raw topic name.

    Returns
    -------
    str
        Normalized topic name.
    """
    if not isinstance(topic, str):
        raise InvalidTopicError("Topic must be a string.")

    cleaned = topic.strip().lower()
    if not cleaned:
        raise InvalidTopicError("Topic must not be empty.")
    return cleaned


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def validate_text(text: str) -> None:
    """
    Validate that text is a non-empty string.
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")
    if not text.strip():
        raise InvalidTextError("Text must not be empty.")


def validate_topic(topic: str) -> None:
    """
    Validate that topic is a non-empty string.
    """
    if not isinstance(topic, str):
        raise InvalidTopicError("Topic must be a string.")
    if not topic.strip():
        raise InvalidTopicError("Topic must not be empty.")


def validate_switch_threshold(value: Any) -> float:
    """
    Validate and normalize switch_threshold.

    For Phase-1 we expect a float between 0.0 and 1.0. Values outside this
    range are clamped and logged.
    """
    try:
        threshold = float(value)
    except Exception:
        logger.warning(
            "Invalid switch_threshold value %r. Falling back to default 0.3.", value
        )
        return 0.3

    if threshold < 0.0:
        logger.warning("switch_threshold < 0.0. Clamping to 0.0.")
        threshold = 0.0
    elif threshold > 1.0:
        logger.warning("switch_threshold > 1.0. Clamping to 1.0.")
        threshold = 1.0

    return threshold


def validate_min_overlap(value: Any) -> int:
    """
    Validate and normalize min_keyword_overlap.

    Must be a positive integer. Values <= 0 are replaced with 1.
    """
    try:
        overlap = int(value)
    except Exception:
        logger.warning(
            "Invalid min_keyword_overlap value %r. Falling back to default 1.", value
        )
        return 1

    if overlap <= 0:
        logger.warning("min_keyword_overlap <= 0. Using 1 instead.")
        overlap = 1

    return overlap


# ==============================================================================
# Chunk-1 + Chunk-2: Auto Topic Switch Detector (Phase-1 Only)
# ==============================================================================


class AutoTopicSwitchDetector:
    """
    Phase-1 rule-based topic switch detector.

    Design goals:
        - Deterministic and fully inspectable logic.
        - Configurable topics and keywords via YAML or in-memory setup.
        - Pure string-based keyword overlap heuristic.

    Core algorithm (Phase-1):
        1. Sanitize and tokenize the new message into keywords.
        2. For each configured topic, compute an overlap score:
           score = (# matching keywords) / (total topic keywords)
        3. Select the topic with the highest score.
        4. If no topic has any overlap, fall back to default_topic.
        5. A topic switch occurs if new_topic != previous_topic.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AutoTopicSwitchDetector.

        Parameters
        ----------
        config_path : Optional[str], optional
            Optional path to a YAML config file.
        """
        logger.info("Initializing AutoTopicSwitchDetector (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: TopicSwitchConfig = self._build_config_from_raw(raw_config)

        logger.info(
            "AutoTopicSwitchDetector configured with %d topics. Default topic: %r",
            len(self._config.rules.topics),
            self._config.default_topic,
        )

        logger.info("AutoTopicSwitchDetector initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> TopicSwitchConfig:
        """
        Construct TopicSwitchConfig from raw YAML dictionary.

        This function performs:
            - Defaulting for missing values.
            - Validation for thresholds.
            - Normalization of topic definitions.
        """
        topics_raw = raw.get("topics", {}) or {}
        if not isinstance(topics_raw, dict):
            logger.warning("Invalid 'topics' section in config. Using empty topics.")
            topics_raw = {}

        min_overlap = validate_min_overlap(raw.get("min_keyword_overlap", 1))
        switch_threshold = validate_switch_threshold(raw.get("switch_threshold", 0.3))
        default_topic_raw = raw.get("default_topic", "general")
        default_topic = sanitize_topic_name(default_topic_raw)

        topics: Dict[str, TopicDefinition] = {}

        for topic_name, entry in topics_raw.items():
            try:
                normalized_name = sanitize_topic_name(topic_name)

                if not isinstance(entry, dict):
                    logger.warning(
                        "Topic '%s' entry must be a dict. Skipping.", topic_name
                    )
                    continue

                keywords = entry.get("keywords", [])
                description = entry.get("description")

                if not isinstance(keywords, list) or not keywords:
                    logger.warning(
                        "Topic '%s' has no valid 'keywords' list. Skipping.",
                        topic_name,
                    )
                    continue

                normalized_keywords = []
                for kw in keywords:
                    if isinstance(kw, str) and kw.strip():
                        normalized_keywords.append(kw.strip().lower())
                    else:
                        logger.warning(
                            "Skipping invalid keyword %r in topic '%s'.", kw, topic_name
                        )

                if not normalized_keywords:
                    logger.warning(
                        "Topic '%s' ended up with no usable keywords. Skipping.",
                        topic_name,
                    )
                    continue

                topics[normalized_name] = TopicDefinition(
                    name=normalized_name,
                    keywords=normalized_keywords,
                    description=description,
                )

            except TopicSwitchError as exc:
                logger.warning(
                    "Skipping invalid topic config for '%s': %s", topic_name, exc
                )

        rules = TopicSwitchRules(
            topics=topics,
            min_keyword_overlap=min_overlap,
            switch_threshold=switch_threshold,
        )

        return TopicSwitchConfig(rules=rules, default_topic=default_topic)

    # ------------------------------ Tokenization -----------------------------

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """
        Extract simple alphanumeric 'keywords' from text.

        For Phase-1 we:
            - Lowercase the text
            - Extract [a-zA-Z0-9]+ sequences
            - Do not apply stemming or stopword removal
        """
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return words

    # ---------------------------- Scoring Helpers ----------------------------

    @staticmethod
    def _compute_overlap(
        words: List[str], topic_keywords: List[str]
    ) -> Tuple[int, float]:
        """
        Compute absolute and relative overlap between message words and topic keywords.

        Returns
        -------
        Tuple[int, float]
            (match_count, overlap_score)

            where overlap_score = match_count / len(topic_keywords)
        """
        if not topic_keywords:
            return 0, 0.0

        topic_set = set(topic_keywords)
        match_count = sum(1 for w in words if w in topic_set)
        score = match_count / float(len(topic_keywords))

        return match_count, score

    # -------------------------- Topic Determination --------------------------

    def _find_best_topic(self, text: str) -> Tuple[str, float]:
        """
        Determine the best matching topic for the given text.

        Parameters
        ----------
        text : str
            Sanitized input text.

        Returns
        -------
        Tuple[str, float]
            (best_topic_name, best_score)
        """
        words = self._extract_keywords(text)
        if not words:
            logger.info(
                "No usable keywords extracted from text. Falling back to default topic."
            )
            return self._config.default_topic, 0.0

        best_topic = self._config.default_topic
        best_score = 0.0

        for topic_name, topic_def in self._config.rules.topics.items():
            match_count, score = self._compute_overlap(words, topic_def.keywords)

            logger.debug(
                "Topic '%s' overlap: match_count=%d, score=%.3f",
                topic_name,
                match_count,
                score,
            )

            if match_count < self._config.rules.min_keyword_overlap:
                # Not enough overlap to be considered
                continue

            if score > best_score:
                best_score = score
                best_topic = topic_name

        return best_topic, best_score

    # --------------------------- Public Interface ----------------------------

    def detect_topic(self, text: str) -> str:
        """
        Detect the most likely topic for a given text.

        Parameters
        ----------
        text : str
            User message.

        Returns
        -------
        str
            Detected topic name.
        """
        validate_text(text)
        cleaned = sanitize_text(text)

        topic, score = self._find_best_topic(cleaned)

        logger.info(
            "Detected topic=%r (score=%.3f) for text fragment=%r",
            topic,
            score,
            cleaned[:80],
        )
        return topic

    def is_topic_switch(self, previous_topic: str, new_text: str) -> bool:
        """
        Determine whether a topic switch has occurred.

        Parameters
        ----------
        previous_topic : str
            Previously active topic.
        new_text : str
            New user message.

        Returns
        -------
        bool
            True if topic switched, False otherwise.
        """
        validate_topic(previous_topic)
        validate_text(new_text)

        prev_topic_norm = sanitize_topic_name(previous_topic)
        cleaned = sanitize_text(new_text)

        new_topic, score = self._find_best_topic(cleaned)

        if not self._config.rules.topics:
            # If no topics configured, we only compare labels directly.
            if new_topic != prev_topic_norm:
                logger.info(
                    "Topic switch (no rules configured): '%s' -> '%s'",
                    prev_topic_norm,
                    new_topic,
                )
                return True
            logger.info(
                "No topic switch (no rules configured): still '%s'", prev_topic_norm
            )
            return False

        if new_topic != prev_topic_norm:
            logger.info(
                "Topic switch detected: '%s' -> '%s' (score=%.3f)",
                prev_topic_norm,
                new_topic,
                score,
            )
            return True

        logger.info("No topic switch: still '%s' (score=%.3f)", prev_topic_norm, score)
        return False

    # ------------------------------- Utilities -------------------------------

    def list_topics(self) -> List[str]:
        """
        Return a list of known topic names.
        """
        return sorted(self._config.rules.topics.keys())

    def describe_topic(self, topic: str) -> Optional[str]:
        """
        Return the description for a given topic, if available.
        """
        normalized = sanitize_topic_name(topic)
        topic_def = self._config.rules.topics.get(normalized)
        return topic_def.description if topic_def else None

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a debug snapshot of internal configuration.

        This is safe to log and can be extended in later phases.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "topics": self.list_topics(),
            "default_topic": self._config.default_topic,
            "min_keyword_overlap": self._config.rules.min_keyword_overlap,
            "switch_threshold": self._config.rules.switch_threshold,
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for AutoTopicSwitchDetector...")

    # Example in-memory style behavior (no external YAML)
    detector = AutoTopicSwitchDetector(config_path=None)

    # Manually simulate basic topics if no config is present
    if not detector.list_topics():
        logger.info("No topics configured via YAML. Using in-memory defaults for test.")
        raw_fallback_config = {
            "default_topic": "general",
            "min_keyword_overlap": 1,
            "switch_threshold": 0.3,
            "topics": {
                "shopping": {
                    "keywords": ["buy", "order", "cart", "price", "discount"],
                    "description": "Shopping related queries",
                },
                "travel": {
                    "keywords": ["flight", "hotel", "ticket", "train", "visa"],
                    "description": "Travel planning and booking",
                },
                "support": {
                    "keywords": ["error", "issue", "bug", "help", "support"],
                    "description": "Technical support or troubleshooting",
                },
            },
        }
        detector = AutoTopicSwitchDetector(config_path=None)
        # Rebuild config directly from raw fallback
        detector._config = detector._build_config_from_raw(raw_fallback_config)

    previous_topic = "shopping"
    message_1 = "I want to add this item to my cart and check the price."
    message_2 = "My flight ticket keeps failing to book and shows an error."
    message_3 = "I still need help with the same shopping issue."

    print("Topics configured:", detector.list_topics())

    print("\nMessage 1:", message_1)
    detected_1 = detector.detect_topic(message_1)
    print("Detected topic 1:", detected_1)
    print("Topic switch 1:", detector.is_topic_switch(previous_topic, message_1))

    print("\nMessage 2:", message_2)
    detected_2 = detector.detect_topic(message_2)
    print("Detected topic 2:", detected_2)
    print("Topic switch 2:", detector.is_topic_switch(previous_topic, message_2))

    print("\nMessage 3:", message_3)
    detected_3 = detector.detect_topic(message_3)
    print("Detected topic 3:", detected_3)
    print("Topic switch 3 (from last detected topic):", detector.is_topic_switch(detected_2, message_3))

    debug_info = detector.debug_state()
    logger.info("AutoTopicSwitchDetector debug state: %s", debug_info)

    logger.info("Phase-1 manual test for AutoTopicSwitchDetector completed.")
