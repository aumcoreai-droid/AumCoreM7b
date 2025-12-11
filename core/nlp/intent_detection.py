"""
================================================================================
AumCore_AI - Intent Detection
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: intent_detection.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    intent detection engine. This module maps user messages to high-level
    "intent" labels using ONLY handcrafted keyword rules and simple heuristics.

    Phase-1 constraints:
        - No ML models, no embeddings, no transformers
        - No external APIs, no vector search
        - Pure rule-based keyword and pattern matching
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Maintain configured set of intents and their keywords/patterns
        - Detect best-matching intent for a given user message
        - Provide a confidence-like score (heuristic only)
        - Return a safe fallback intent when ambiguous
        - Offer debug state for observability

    Future phases may add:
        - Model-based classification blended with rules
        - Per-user intent priors and personalization
        - Multilingual intent coverage with translation

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
__module_name__ = "intent_detection"
__description__ = (
    "Phase-1 rule-based Intent Detection engine with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("IntentDetection")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class IntentDetectionError(Exception):
    """
    Base class for all intent detection errors.
    """

    def __init__(self, message: str, *, code: str = "INTENT_DETECTION_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidTextError(IntentDetectionError):
    """
    Raised when provided text is missing or invalid.
    """

    def __init__(self, message: str = "Text must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_TEXT")


class InvalidConfigError(IntentDetectionError):
    """
    Raised when YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid intent detection configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class IntentRule:
    """
    Represents a single rule for an intent.

    Attributes
    ----------
    intent : str
        Canonical intent name (normalized to lowercase).
    keywords : List[str]
        Simple keyword triggers; plain lowercase tokens.
    patterns : List[str]
        Regex patterns for more flexible matching.
    description : Optional[str]
        Optional human-readable description.
    """

    intent: str
    keywords: List[str]
    patterns: List[str]
    description: Optional[str] = None


@dataclass
class IntentDetectionRules:
    """
    Represents the set of rules and thresholds for intent detection.

    Attributes
    ----------
    rules : Dict[str, IntentRule]
        Mapping of intent name -> IntentRule.
    min_keyword_matches : int
        Minimum number of keyword matches required to consider an intent.
    min_confidence_score : float
        Minimum heuristic score to avoid falling back to default.
    """

    rules: Dict[str, IntentRule]
    min_keyword_matches: int = 1
    min_confidence_score: float = 0.1


@dataclass
class IntentDetectionConfig:
    """
    Represents the overall configuration for intent detection.

    Attributes
    ----------
    rules : IntentDetectionRules
        Rule-based detection logic.
    default_intent : str
        Fallback intent when no confident match is found.
    """

    rules: IntentDetectionRules
    default_intent: str = "general_query"


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for intent detection, if present.

    Expected structure (Phase-1, optional):

        default_intent: "general_query"
        min_keyword_matches: 1
        min_confidence_score: 0.2
        intents:
          greeting:
            keywords: ["hello", "hi", "hey"]
            patterns: ["^hi there", "^hello there"]
            description: "Greeting and casual hello"
          farewell:
            keywords: ["bye", "goodbye", "see you"]
            patterns: []
            description: "Ending the conversation"

    Parameters
    ----------
    path : Optional[str]
        Path to YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed config dictionary or empty dict on error.
    """
    if not path:
        logger.info("No config path provided for IntentDetection. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Intent detection config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded intent detection config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load intent detection YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_text(text: str) -> str:
    """
    Sanitize and normalize user text.

    - Ensures string type
    - Strips leading/trailing whitespace
    - Lowercases text
    - Normalizes internal whitespace
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")

    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def sanitize_intent_name(intent: str) -> str:
    """
    Normalize intent names to canonical lowercase form.
    """
    if not isinstance(intent, str):
        raise InvalidConfigError("Intent name must be a string.")
    cleaned = intent.strip().lower()
    if not cleaned:
        raise InvalidConfigError("Intent name must not be empty.")
    return cleaned


def sanitize_keyword(keyword: str) -> str:
    """
    Normalize keywords; used for exact token matching.
    """
    if not isinstance(keyword, str):
        raise InvalidConfigError("Keyword must be a string.")
    cleaned = keyword.strip().lower()
    if not cleaned:
        raise InvalidConfigError("Keyword must not be empty.")
    return cleaned


def sanitize_pattern(pattern: str) -> str:
    """
    Normalize regex patterns; minimal validation for Phase-1.
    """
    if not isinstance(pattern, str):
        raise InvalidConfigError("Pattern must be a string.")
    cleaned = pattern.strip()
    if not cleaned:
        raise InvalidConfigError("Pattern must not be empty.")
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


def validate_min_keyword_matches(value: Any) -> int:
    """
    Validate min_keyword_matches is an integer >= 1.
    """
    try:
        ivalue = int(value)
    except Exception:
        logger.warning(
            "Invalid min_keyword_matches value %r. Falling back to default 1.", value
        )
        return 1

    if ivalue <= 0:
        logger.warning("min_keyword_matches <= 0. Using 1 instead.")
        ivalue = 1

    return ivalue


def validate_min_confidence_score(value: Any) -> float:
    """
    Validate min_confidence_score is a float between 0.0 and 1.0.
    """
    try:
        fvalue = float(value)
    except Exception:
        logger.warning(
            "Invalid min_confidence_score value %r. Falling back to default 0.1.",
            value,
        )
        return 0.1

    if fvalue < 0.0:
        logger.warning("min_confidence_score < 0.0. Clamping to 0.0.")
        fvalue = 0.0
    elif fvalue > 1.0:
        logger.warning("min_confidence_score > 1.0. Clamping to 1.0.")
        fvalue = 1.0

    return fvalue


# ==============================================================================
# Chunk-1 + Chunk-2: Intent Detection (Phase-1 Only)
# ==============================================================================


class IntentDetection:
    """
    Phase-1 rule-based intent detection engine.

    Core behavior:
        - Use keyword and regex pattern matches for each intent.
        - Compute a simple heuristic "confidence" score.
        - Return the best-scoring intent.
        - If score is too low, fall back to default intent.

    The implementation is intentionally transparent so higher layers
    can introspect and later combine it with model-based approaches.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the IntentDetection engine.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing IntentDetection (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: IntentDetectionConfig = self._build_config_from_raw(raw_config)

        logger.info(
            "IntentDetection configured with %d intents. Default intent: %r",
            len(self._config.rules.rules),
            self._config.default_intent,
        )

        logger.info("IntentDetection initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> IntentDetectionConfig:
        intents_raw = raw.get("intents", {}) or {}
        if not isinstance(intents_raw, dict):
            logger.warning("Invalid 'intents' section in config. Using empty intents.")
            intents_raw = {}

        min_kw = validate_min_keyword_matches(raw.get("min_keyword_matches", 1))
        min_conf = validate_min_confidence_score(raw.get("min_confidence_score", 0.1))
        default_intent_raw = raw.get("default_intent", "general_query")
        default_intent = sanitize_intent_name(default_intent_raw)

        rules: Dict[str, IntentRule] = {}

        for intent_name, entry in intents_raw.items():
            try:
                normalized_intent = sanitize_intent_name(intent_name)

                if not isinstance(entry, dict):
                    logger.warning(
                        "Intent '%s' entry must be a dict. Skipping.", intent_name
                    )
                    continue

                keywords_raw = entry.get("keywords", []) or []
                patterns_raw = entry.get("patterns", []) or []
                description = entry.get("description")

                if not isinstance(keywords_raw, list):
                    logger.warning(
                        "Intent '%s' keywords must be a list. Skipping keywords.",
                        intent_name,
                    )
                    keywords_raw = []
                if not isinstance(patterns_raw, list):
                    logger.warning(
                        "Intent '%s' patterns must be a list. Skipping patterns.",
                        intent_name,
                    )
                    patterns_raw = []

                keywords: List[str] = []
                for kw in keywords_raw:
                    try:
                        keywords.append(sanitize_keyword(kw))
                    except InvalidConfigError as exc:
                        logger.warning(
                            "Skipping invalid keyword %r for intent '%s': %s",
                            kw,
                            intent_name,
                            exc,
                        )

                patterns: List[str] = []
                for pat in patterns_raw:
                    try:
                        patterns.append(sanitize_pattern(pat))
                    except InvalidConfigError as exc:
                        logger.warning(
                            "Skipping invalid pattern %r for intent '%s': %s",
                            pat,
                            intent_name,
                            exc,
                        )

                if not keywords and not patterns:
                    logger.warning(
                        "Intent '%s' has no usable keywords or patterns. Skipping.",
                        intent_name,
                    )
                    continue

                rules[normalized_intent] = IntentRule(
                    intent=normalized_intent,
                    keywords=keywords,
                    patterns=patterns,
                    description=description,
                )

            except IntentDetectionError as exc:
                logger.warning(
                    "Skipping invalid intent config for '%s': %s", intent_name, exc
                )

        detection_rules = IntentDetectionRules(
            rules=rules,
            min_keyword_matches=min_kw,
            min_confidence_score=min_conf,
        )

        return IntentDetectionConfig(
            rules=detection_rules,
            default_intent=default_intent,
        )

    # ------------------------------ Tokenization -----------------------------

    @staticmethod
    def _extract_tokens(text: str) -> List[str]:
        """
        Extract simple word-like tokens from text.
        """
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return tokens

    # ---------------------------- Scoring Helpers ----------------------------

    @staticmethod
    def _score_keyword_matches(tokens: List[str], keywords: List[str]) -> Tuple[int, float]:
        """
        Compute count and ratio of keyword matches in tokens.

        Returns
        -------
        Tuple[int, float]
            (match_count, ratio) where ratio = match_count / len(keywords)
        """
        if not keywords:
            return 0, 0.0

        token_set = set(tokens)
        matches = sum(1 for kw in keywords if kw in token_set)
        ratio = matches / float(len(keywords))
        return matches, ratio

    @staticmethod
    def _score_pattern_matches(text: str, patterns: List[str]) -> Tuple[int, float]:
        """
        Compute count and ratio of regex matches in text.

        Returns
        -------
        Tuple[int, float]
            (match_count, normalized_score)
        """
        if not patterns:
            return 0, 0.0

        match_count = 0
        for pat in patterns:
            try:
                if re.search(pat, text):
                    match_count += 1
            except re.error as exc:
                logger.warning("Invalid regex pattern %r skipped: %s", pat, exc)
                continue

        ratio = match_count / float(len(patterns))
        return match_count, ratio

    def _compute_intent_score(
        self,
        text: str,
        tokens: List[str],
        rule: IntentRule,
    ) -> float:
        """
        Compute a heuristic score for a given intent rule.

        Scoring heuristic (Phase-1):
            - keyword_score: ratio of matched keywords
            - pattern_score: ratio of matched patterns
            - total_score = 0.7 * keyword_score + 0.3 * pattern_score
        """
        kw_count, kw_ratio = self._score_keyword_matches(tokens, rule.keywords)
        pt_count, pt_ratio = self._score_pattern_matches(text, rule.patterns)

        if kw_count < self._config.rules.min_keyword_matches and pt_count == 0:
            return 0.0

        total_score = 0.7 * kw_ratio + 0.3 * pt_ratio
        return total_score

    # --------------------------- Public Interface ----------------------------

    def detect_intent(self, text: str) -> Tuple[str, float]:
        """
        Detect intent for a given user message.

        Parameters
        ----------
        text : str
            User message.

        Returns
        -------
        Tuple[str, float]
            (intent_name, score)
        """
        validate_text(text)
        cleaned = sanitize_text(text)
        tokens = self._extract_tokens(cleaned)

        if not tokens:
            logger.info(
                "No usable tokens extracted from text. Falling back to default intent."
            )
            return self._config.default_intent, 0.0

        best_intent = self._config.default_intent
        best_score = 0.0

        for intent_name, rule in self._config.rules.rules.items():
            score = self._compute_intent_score(cleaned, tokens, rule)
            logger.debug(
                "Intent '%s' score=%.3f for text fragment=%r",
                intent_name,
                score,
                cleaned[:80],
            )

            if score > best_score:
                best_score = score
                best_intent = intent_name

        if best_score < self._config.rules.min_confidence_score:
            logger.info(
                "Best score %.3f below min_confidence_score %.3f. "
                "Falling back to default intent %r.",
                best_score,
                self._config.rules.min_confidence_score,
                self._config.default_intent,
            )
            return self._config.default_intent, best_score

        logger.info(
            "Detected intent=%r with score=%.3f for text fragment=%r",
            best_intent,
            best_score,
            cleaned[:80],
        )
        return best_intent, best_score

    # ------------------------------ Introspection -----------------------------

    def list_intents(self) -> List[str]:
        """
        Return list of configured intent names.
        """
        return sorted(self._config.rules.rules.keys())

    def describe_intent(self, intent: str) -> Optional[str]:
        """
        Return description for a given intent, if available.
        """
        try:
            name = sanitize_intent_name(intent)
        except InvalidConfigError:
            return None

        rule = self._config.rules.rules.get(name)
        return rule.description if rule else None

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of internal configuration.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "default_intent": self._config.default_intent,
            "min_keyword_matches": self._config.rules.min_keyword_matches,
            "min_confidence_score": self._config.rules.min_confidence_score,
            "intents": self.list_intents(),
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for IntentDetection...")

    # Example in-memory style configuration for manual testing
    raw_fallback_config = {
        "default_intent": "general_query",
        "min_keyword_matches": 1,
        "min_confidence_score": 0.2,
        "intents": {
            "greeting": {
                "keywords": ["hello", "hi", "hey"],
                "patterns": ["^hi there", "^hello there"],
                "description": "Greeting and casual hello",
            },
            "farewell": {
                "keywords": ["bye", "goodbye", "see you"],
                "patterns": [],
                "description": "Conversation ending or goodbye.",
            },
            "ask_status": {
                "keywords": ["status", "track", "update"],
                "patterns": [r"where is my", r"what is the status"],
                "description": "Asking about status of order or item.",
            },
        },
    }

    detector = IntentDetection(config_path=None)
    detector._config = detector._build_config_from_raw(raw_fallback_config)

    samples = [
        "Hi there, I just wanted to say hello.",
        "Can you tell me the status of my order?",
        "Okay bye, talk to you later.",
        "I need some help but not sure how to ask.",
    ]

    for text in samples:
        intent, score = detector.detect_intent(text)
        print(f"TEXT: {text!r}")
        print(f" -> intent={intent!r}, score={score:.3f}")
        print("-" * 60)

    debug_info = detector.debug_state()
    logger.info("IntentDetection debug state: %s", debug_info)

    logger.info("Phase-1 manual test for IntentDetection completed.")
