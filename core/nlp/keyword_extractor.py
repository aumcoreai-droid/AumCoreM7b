"""
================================================================================
AumCore_AI - Keyword Extractor
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: keyword_extractor.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    keyword extraction engine. This module extracts important keywords from
    user text using handcrafted heuristics and simple filters.

    Phase-1 constraints:
        - No ML models, no embeddings, no transformers
        - No external APIs, no TF-IDF over large corpora
        - Pure rule-based tokenization + scoring
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Tokenize text into word-like units
        - Filter out stopwords and low-signal tokens
        - Score remaining tokens using basic heuristics
        - Return top-N keywords in a deterministic way
        - Provide debug state for observability

    Future phases may add:
        - Model-based keyword/phrase extraction
        - Multi-word key-phrase identification
        - Domain-specific keyword boosting

================================================================================
"""

# ==============================================================================
# Chunk-1: Imports, Metadata, Logging Setup
# ==============================================================================

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import yaml

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "keyword_extractor"
__description__ = (
    "Phase-1 rule-based Keyword Extractor with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("KeywordExtractor")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class KeywordExtractorError(Exception):
    """
    Base class for all keyword extractor errors.
    """

    def __init__(self, message: str, *, code: str = "KEYWORD_EXTRACTOR_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidTextError(KeywordExtractorError):
    """
    Raised when provided text is missing or invalid.
    """

    def __init__(self, message: str = "Text must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_TEXT")


class InvalidConfigError(KeywordExtractorError):
    """
    Raised when YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid keyword extractor configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class KeywordScoringConfig:
    """
    Configuration for scoring and filtering keywords.

    Attributes
    ----------
    min_token_length : int
        Minimum token length to be considered as keyword candidate.
    max_token_length : int
        Maximum token length to be considered.
    max_keywords : int
        Maximum number of keywords to return.
    """

    min_token_length: int = 3
    max_token_length: int = 30
    max_keywords: int = 10


@dataclass
class KeywordExtractorConfig:
    """
    Overall configuration for the keyword extractor.

    Attributes
    ----------
    scoring : KeywordScoringConfig
        Numeric thresholds for filtering and output.
    stopwords : List[str]
        List of stopwords that should be filtered out.
    """

    scoring: KeywordScoringConfig
    stopwords: List[str]


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for keyword extractor, if present.

    Expected structure (Phase-1, optional):

        scoring:
          min_token_length: 3
          max_token_length: 30
          max_keywords: 10
        stopwords:
          - "the"
          - "is"
          - "and"
          - "a"
          - "an"

    Parameters
    ----------
    path : Optional[str]
        Path to YAML config file.

    Returns
    -------
    Dict[str, Any]
        Parsed dictionary or empty on error.
    """
    if not path:
        logger.info("No config path provided for KeywordExtractor. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Keyword extractor config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded keyword extractor config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load keyword extractor YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_text(text: str) -> str:
    """
    Sanitize and normalize input text.

    - Ensures string type
    - Strips leading/trailing whitespace
    - Lowercases
    - Normalizes internal whitespace
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")

    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def sanitize_stopword(word: str) -> str:
    """
    Normalize stopword entries.
    """
    if not isinstance(word, str):
        raise InvalidConfigError("Stopword must be a string.")
    cleaned = word.strip().lower()
    if not cleaned:
        raise InvalidConfigError("Stopword must not be empty.")
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


# ==============================================================================
# Chunk-1 + Chunk-2: Keyword Extractor (Phase-1 Only)
# ==============================================================================


class KeywordExtractor:
    """
    Phase-1 rule-based keyword extractor.

    Core behavior:
        - Tokenize text into lowercase word-like units.
        - Remove stopwords and very short/long tokens.
        - Count frequency of remaining tokens.
        - Rank by frequency, then lexicographically for stability.
        - Return top-N keywords.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the KeywordExtractor.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing KeywordExtractor (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: KeywordExtractorConfig = self._build_config_from_raw(raw_config)

        logger.info(
            "KeywordExtractor configured: min_len=%d, max_len=%d, max_keywords=%d, stopwords=%d",
            self._config.scoring.min_token_length,
            self._config.scoring.max_token_length,
            self._config.scoring.max_keywords,
            len(self._config.stopwords),
        )

        logger.info("KeywordExtractor initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> KeywordExtractorConfig:
        scoring_raw = raw.get("scoring", {}) or {}
        min_len = _validate_positive_int(
            scoring_raw.get("min_token_length", 3),
            default=3,
            name="min_token_length",
        )
        max_len = _validate_positive_int(
            scoring_raw.get("max_token_length", 30),
            default=30,
            name="max_token_length",
        )
        max_keywords = _validate_positive_int(
            scoring_raw.get("max_keywords", 10),
            default=10,
            name="max_keywords",
        )

        if min_len > max_len:
            logger.warning(
                "min_token_length > max_token_length. Swapping values: %d, %d",
                min_len,
                max_len,
            )
            min_len, max_len = max_len, min_len

        stopwords_raw = raw.get(
            "stopwords",
            [
                "the",
                "is",
                "are",
                "am",
                "a",
                "an",
                "and",
                "or",
                "to",
                "of",
                "for",
                "in",
                "on",
                "at",
                "this",
                "that",
                "it",
                "with",
                "as",
                "by",
            ],
        )

        if not isinstance(stopwords_raw, list):
            logger.warning(
                "Invalid stopwords section in config. Using default English stopwords."
            )
            stopwords_raw = [
                "the",
                "is",
                "are",
                "am",
                "a",
                "an",
                "and",
                "or",
                "to",
                "of",
                "for",
                "in",
                "on",
                "at",
                "this",
                "that",
                "it",
                "with",
                "as",
                "by",
            ]

        stopwords: List[str] = []
        for sw in stopwords_raw:
            try:
                stopwords.append(sanitize_stopword(sw))
            except InvalidConfigError as exc:
                logger.warning("Skipping invalid stopword %r: %s", sw, exc)

        scoring = KeywordScoringConfig(
            min_token_length=min_len,
            max_token_length=max_len,
            max_keywords=max_keywords,
        )

        return KeywordExtractorConfig(scoring=scoring, stopwords=stopwords)

    # ------------------------------ Tokenization -----------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text into lowercase word-like tokens.
        """
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return tokens

    # ------------------------------ Filtering --------------------------------

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on stopwords and length thresholds.
        """
        stopword_set = set(self._config.stopwords)
        min_len = self._config.scoring.min_token_length
        max_len = self._config.scoring.max_token_length

        filtered = []
        for t in tokens:
            if t in stopword_set:
                continue
            if len(t) < min_len or len(t) > max_len:
                continue
            filtered.append(t)

        return filtered

    # ------------------------------ Scoring ----------------------------------

    @staticmethod
    def _score_tokens(tokens: List[str]) -> List[Tuple[str, int]]:
        """
        Score tokens by frequency; return list of (token, count).
        """
        counter = Counter(tokens)
        items = list(counter.items())
        items.sort(key=lambda x: (-x[1], x[0]))
        return items

    # --------------------------- Public Interface ----------------------------

    def extract_keywords(self, text: str, max_keywords: Optional[int] = None) -> List[str]:
        """
        Extract top keywords from text.

        Parameters
        ----------
        text : str
            Input text.
        max_keywords : Optional[int]
            Optional override for maximum number of keywords to return.

        Returns
        -------
        List[str]
            Ordered list of keywords.
        """
        validate_text(text)
        cleaned = sanitize_text(text)

        tokens = self._tokenize(cleaned)
        if not tokens:
            logger.info("No tokens extracted from text. Returning empty keyword list.")
            return []

        filtered = self._filter_tokens(tokens)
        if not filtered:
            logger.info(
                "No tokens left after filtering. Returning empty keyword list."
            )
            return []

        scored = self._score_tokens(filtered)

        limit = (
            max_keywords
            if isinstance(max_keywords, int) and max_keywords > 0
            else self._config.scoring.max_keywords
        )

        top_tokens = [token for token, _ in scored[:limit]]
        logger.info(
            "Extracted %d keywords (limit=%d) from %d filtered tokens.",
            len(top_tokens),
            limit,
            len(filtered),
        )
        return top_tokens

    # ------------------------------ Introspection -----------------------------

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of internal configuration.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "min_token_length": self._config.scoring.min_token_length,
            "max_token_length": self._config.scoring.max_token_length,
            "max_keywords": self._config.scoring.max_keywords,
            "stopwords_count": len(self._config.stopwords),
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for KeywordExtractor...")

    extractor = KeywordExtractor(config_path=None)

    sample_text = (
        "I want to understand the current status of my order and the delivery time, "
        "because the tracking page is not updating properly and this is causing issues."
    )

    keywords_default = extractor.extract_keywords(sample_text)
    print("Keywords (default):", keywords_default)

    keywords_top5 = extractor.extract_keywords(sample_text, max_keywords=5)
    print("Keywords (top 5):", keywords_top5)

    debug_info = extractor.debug_state()
    logger.info("KeywordExtractor debug state: %s", debug_info)

    logger.info("Phase-1 manual test for KeywordExtractor completed.")
