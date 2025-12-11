"""
================================================================================
AumCore_AI - Adaptive Response Length
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: adaptive_response_length.py

Description:
    Enterprise-grade Phase-1 implementation of an adaptive response length
    controller. This module provides a deterministic, rule-based mechanism
    to normalize response length into three broad bands:

        - short  : responses that are too short are gently extended
        - medium : responses that are within acceptable bounds are preserved
        - long   : responses that are too long are safely trimmed

    This is intended as a foundational building block for later phases of
    AumCore_AI, where more advanced behavior (per-intent tuning, user profile
    awareness, telemetry-driven adaptation, etc.) will be layered on top.

    In Phase-1, the scope is intentionally constrained to:

    - Chunk-1:
        * Clean imports and metadata
        * Logging setup and basic instrumentation
        * Clear, PEP8-compliant class structure

    - Chunk-2:
        * Config loader for thresholds and modes
        * Input sanitization helpers
        * Input validation helpers
        * Custom error hierarchy
        * Deterministic, rule-based logic only

    IMPORTANT:
        - No model loading or model calls are allowed in Phase-1.
        - All logic MUST be transparent and predictable.
        - This module is intentionally verbose to support future extension.

================================================================================
"""

# ==============================================================================
# Chunk-1: Imports, Metadata, Logging Setup
# ==============================================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import yaml

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "adaptive_response_length"
__description__ = (
    "Phase-1 rule-based Adaptive Response Length controller with "
    "Chunk-1 and Chunk-2 only."
)

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("AdaptiveResponseLength")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class ResponseLengthError(Exception):
    """
    Base class for all adaptive response length errors.

    Using a dedicated base exception class makes it easier for higher-level
    components to catch any length-control-related issues without depending
    on low-level details.
    """

    def __init__(self, message: str, *, code: str = "RESPONSE_LENGTH_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidThresholdError(ResponseLengthError):
    """
    Raised when configured thresholds are invalid (e.g. negative, reversed).
    """

    def __init__(self, message: str = "Invalid response length thresholds."):
        super().__init__(message, code="INVALID_THRESHOLD")


class InvalidTextError(ResponseLengthError):
    """
    Raised when the provided text is not a valid string or is unusably empty.
    """

    def __init__(self, message: str = "Text must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_TEXT")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class ResponseLengthThresholds:
    """
    Represents normalized thresholds for response length control.

    Attributes
    ----------
    short_threshold : int
        If len(text) < short_threshold, the response is considered too short.
    long_threshold : int
        If len(text) > long_threshold, the response is considered too long.

    In Phase-1 we keep this linear and simple, but the structure is designed
    so that later phases can add more fields (per-intent thresholds, etc.).
    """

    short_threshold: int = 50
    long_threshold: int = 200


@dataclass
class ResponseLengthConfig:
    """
    Represents the overall configuration for AdaptiveResponseLength.

    Attributes
    ----------
    thresholds : ResponseLengthThresholds
        Tunable thresholds that control how responses are classified.
    extend_suffix : str
        Suffix appended to responses that are too short.
    trim_suffix : str
        Suffix appended to responses that are trimmed.
    """

    thresholds: ResponseLengthThresholds
    extend_suffix: str = "..."
    trim_suffix: str = "..."


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a YAML configuration file if a path is provided and file exists.

    Expected structure (Phase-1, optional):

        short_threshold: int
        long_threshold: int
        extend_suffix: str
        trim_suffix: str

    Parameters
    ----------
    path : Optional[str]
        Path to the YAML configuration file, or None.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary or empty dict on failure.
    """
    if not path:
        logger.info("No config path provided for response length. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Response length config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded response length config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load response length YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_text(text: str) -> str:
    """
    Perform minimal sanitization on text for length control.

    For Phase-1 we keep this deliberately conservative:
    - Ensure the value is a string.
    - Strip leading/trailing whitespace.
    - Leave the internal structure of the text unchanged so that length
      behavior is transparent and predictable.

    Parameters
    ----------
    text : str
        User-facing response text.

    Returns
    -------
    str
        Sanitized text.
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")

    cleaned = text.strip()
    return cleaned


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def validate_thresholds(short_threshold: int, long_threshold: int) -> None:
    """
    Validate that response length thresholds are numeric and logically ordered.

    Rules:
    - Both thresholds must be positive integers.
    - short_threshold < long_threshold
    """
    if not isinstance(short_threshold, int) or not isinstance(long_threshold, int):
        raise InvalidThresholdError("Thresholds must be integers.")

    if short_threshold <= 0 or long_threshold <= 0:
        raise InvalidThresholdError("Thresholds must be positive integers.")

    if short_threshold >= long_threshold:
        raise InvalidThresholdError(
            "short_threshold must be strictly less than long_threshold."
        )


def validate_text(text: str) -> None:
    """
    Validate that text is a non-empty, meaningful string.

    Rules:
    - Must be a string.
    - Must not be empty after stripping.
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")

    if not text.strip():
        raise InvalidTextError("Text must not be empty.")


# ==============================================================================
# Chunk-1 + Chunk-2: Adaptive Response Length (Phase-1 Only)
# ==============================================================================


class AdaptiveResponseLength:
    """
    Phase-1 compliant Adaptive Response Length controller.

    This class implements a deterministic strategy for adjusting text length:

        - If the text is shorter than `short_threshold`, it is considered
          too short and is gently extended by appending a suffix.
        - If the text is longer than `long_threshold`, it is considered
          too long and is trimmed to that length, followed by a suffix.
        - Otherwise, the text is returned unchanged.

    This behavior is intentionally simple in Phase-1, but the structure is
    designed to be extended in later phases, for example:

        - Different thresholds per "intent" or "channel".
        - Context-aware or user preference-aware thresholds.
        - Telemetry-driven tuning (error rates, engagement, etc.).
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AdaptiveResponseLength.

        Parameters
        ----------
        config_path : Optional[str], optional
            Path to YAML config file defining thresholds and suffixes.
            If not provided or invalid, defaults are used.
        """
        logger.info("Initializing AdaptiveResponseLength (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: ResponseLengthConfig = self._build_config_from_raw(raw_config)

        # Log the configuration for observability
        logger.info(
            "AdaptiveResponseLength configured with "
            "short_threshold=%d, long_threshold=%d, extend_suffix=%r, trim_suffix=%r",
            self._config.thresholds.short_threshold,
            self._config.thresholds.long_threshold,
            self._config.extend_suffix,
            self._config.trim_suffix,
        )

        logger.info("AdaptiveResponseLength initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> ResponseLengthConfig:
        """
        Build a ResponseLengthConfig from raw YAML data.

        This isolates details of the configuration structure and ensures
        all validation and defaults are consistently applied.

        Parameters
        ----------
        raw : Dict[str, Any]
            Parsed YAML configuration.

        Returns
        -------
        ResponseLengthConfig
            Validated and normalized configuration object.
        """
        # Extract raw values with defaults
        short_threshold = raw.get("short_threshold", 50)
        long_threshold = raw.get("long_threshold", 200)
        extend_suffix = raw.get("extend_suffix", "...")
        trim_suffix = raw.get("trim_suffix", "...")

        # Validate thresholds
        validate_thresholds(short_threshold, long_threshold)

        # Ensure suffixes are strings
        if not isinstance(extend_suffix, str):
            logger.warning(
                "Invalid extend_suffix type (%s). Falling back to default '...'.",
                type(extend_suffix).__name__,
            )
            extend_suffix = "..."
        if not isinstance(trim_suffix, str):
            logger.warning(
                "Invalid trim_suffix type (%s). Falling back to default '...'.",
                type(trim_suffix).__name__,
            )
            trim_suffix = "..."

        thresholds = ResponseLengthThresholds(
            short_threshold=short_threshold,
            long_threshold=long_threshold,
        )

        return ResponseLengthConfig(
            thresholds=thresholds,
            extend_suffix=extend_suffix,
            trim_suffix=trim_suffix,
        )

    # ------------------------------ Core Logic -------------------------------

    def _classify_length(self, text: str) -> str:
        """
        Classify the text as 'short', 'medium', or 'long' based on thresholds.

        Returns
        -------
        str
            One of: "short", "medium", "long"
        """
        length = len(text)
        st = self._config.thresholds.short_threshold
        lt = self._config.thresholds.long_threshold

        if length < st:
            return "short"
        if length > lt:
            return "long"
        return "medium"

    def _extend_short_text(self, text: str) -> str:
        """
        Extend a short text by appending the configured suffix.

        In Phase-1, we keep extension simple: we do not synthesize new content,
        only append a static suffix to signal continuation.
        """
        suffix = self._config.extend_suffix
        return text + suffix

    def _trim_long_text(self, text: str) -> str:
        """
        Trim a long text to the configured long_threshold and append suffix.
        """
        limit = self._config.thresholds.long_threshold
        suffix = self._config.trim_suffix

        if len(text) <= limit:
            return text

        trimmed = text[:limit]
        return trimmed + suffix

    # --------------------------- Public Interface ----------------------------

    def adjust_response(self, text: str) -> str:
        """
        Adjust the length of a response according to configured thresholds.

        Parameters
        ----------
        text : str
            Original response text.

        Returns
        -------
        str
            Length-normalized response text.

        Raises
        ------
        InvalidTextError
            If the input text is invalid.
        """
        validate_text(text)
        cleaned_text = sanitize_text(text)

        category = self._classify_length(cleaned_text)

        if category == "short":
            logger.info(
                "Response classified as 'short' (len=%d). Extending.", len(cleaned_text)
            )
            return self._extend_short_text(cleaned_text)

        if category == "long":
            logger.info(
                "Response classified as 'long' (len=%d). Trimming.", len(cleaned_text)
            )
            return self._trim_long_text(cleaned_text)

        logger.info(
            "Response classified as 'medium' (len=%d). Returning unchanged.",
            len(cleaned_text),
        )
        return cleaned_text

    # ------------------------------ Introspection ----------------------------

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a debug snapshot of current configuration and derived state.

        This can be safely logged or inspected in Phase-1, and extended in
        future phases as needed.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "short_threshold": self._config.thresholds.short_threshold,
            "long_threshold": self._config.thresholds.long_threshold,
            "extend_suffix": self._config.extend_suffix,
            "trim_suffix": self._config.trim_suffix,
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for AdaptiveResponseLength...")

    arl = AdaptiveResponseLength(config_path=None)

    # Short text example
    short_text = "Short reply"
    adjusted_short = arl.adjust_response(short_text)
    print("SHORT  IN :", repr(short_text))
    print("SHORT OUT :", repr(adjusted_short))

    # Long text example
    long_text = "This is a very long response " * 10
    adjusted_long = arl.adjust_response(long_text)
    print("LONG   IN :", repr(long_text[:80] + "..."))
    print("LONG  OUT :", repr(adjusted_long[:80] + "..."))

    # Medium text example
    medium_text = "This is a reasonably sized response that should not be altered."
    adjusted_medium = arl.adjust_response(medium_text)
    print("MEDIUM IN :", repr(medium_text))
    print("MEDIUMOUT :", repr(adjusted_medium))

    # Debug snapshot
    debug_info = arl.debug_state()
    logger.info("AdaptiveResponseLength debug state: %s", debug_info)

    logger.info("Phase-1 manual test for AdaptiveResponseLength completed.")
