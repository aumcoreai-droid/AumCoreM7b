"""
================================================================================
AumCore_AI - Dual Language Reply
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: dual_language_reply.py

Description:
    Phase-1 enterprise-grade implementation of a rule-based dual language
    reply helper. This module takes an input reply (in a "primary" language)
    and produces a dual-language formatted response (primary + secondary),
    using only deterministic, template-based logic.

    Phase-1 constraints:
        - No translation models, no external APIs
        - No embeddings, no language detection models
        - Pure rule-based formatting + lightweight heuristics
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Maintain configuration for primary and secondary language labels
        - Provide formatted dual-language wrappers for replies
        - Allow optional "hint text" for secondary language
        - Stay fully deterministic and inspectable

    Future phases may add:
        - Real translation support
        - User preference driven language ordering
        - Locale-aware formatting and directionality

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
__module_name__ = "dual_language_reply"
__description__ = (
    "Phase-1 rule-based Dual Language Reply helper with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("DualLanguageReply")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class DualLanguageError(Exception):
    """
    Base class for all dual language reply errors.
    """

    def __init__(self, message: str, *, code: str = "DUAL_LANGUAGE_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidTextError(DualLanguageError):
    """
    Raised when provided text is missing or invalid.
    """

    def __init__(self, message: str = "Text must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_TEXT")


class InvalidConfigError(DualLanguageError):
    """
    Raised when YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid dual language configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class LanguageLabelConfig:
    """
    Represents labels for primary and secondary languages.

    Attributes
    ----------
    primary_label : str
        Human-readable label for the primary language.
    secondary_label : str
        Human-readable label for the secondary language.
    """

    primary_label: str = "English"
    secondary_label: str = "Hindi"


@dataclass
class DualLanguageReplyConfig:
    """
    Represents the overall configuration for dual language replies.

    Attributes
    ----------
    labels : LanguageLabelConfig
        Labels for languages used in formatting.
    separator_line : str
        Line used to visually separate primary and secondary parts.
    show_language_labels : bool
        Whether to annotate each part with the language label.
    """

    labels: LanguageLabelConfig
    separator_line: str = "-" * 40
    show_language_labels: bool = True


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for dual language reply, if present.

    Expected structure (Phase-1, optional):

        labels:
          primary_label: "English"
          secondary_label: "Hindi"
        separator_line: "----------------------------------------"
        show_language_labels: true

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
        logger.info("No config path provided for DualLanguageReply. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Dual language config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded dual language config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load dual language YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_text(text: str) -> str:
    """
    Sanitize and normalize reply text.

    - Ensures string type
    - Strips leading/trailing whitespace
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")
    cleaned = text.strip()
    return cleaned


def sanitize_label(label: str) -> str:
    """
    Normalize language labels (for consistent formatting).
    """
    if not isinstance(label, str):
        raise InvalidConfigError("Language label must be a string.")
    cleaned = label.strip()
    if not cleaned:
        raise InvalidConfigError("Language label must not be empty.")
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


# ==============================================================================
# Chunk-1 + Chunk-2: Dual Language Reply (Phase-1 Only)
# ==============================================================================


class DualLanguageReply:
    """
    Phase-1 rule-based dual language reply formatter.

    NOTE:
        In Phase-1, this module does NOT perform real translation.
        It only wraps text into a dual-language template. The idea is:

            - Primary part: the actual reply (e.g., English)
            - Secondary part: either:
                * a manually provided hint/explanation, or
                * an empty stub placeholder, depending on usage

        This keeps behavior deterministic and clearly separated from any
        model-based translation that may be added in later phases.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DualLanguageReply helper.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing DualLanguageReply (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: DualLanguageReplyConfig = self._build_config_from_raw(raw_config)

        logger.info(
            "DualLanguageReply configured: primary_label=%r, secondary_label=%r, "
            "show_language_labels=%s",
            self._config.labels.primary_label,
            self._config.labels.secondary_label,
            self._config.show_language_labels,
        )

        logger.info("DualLanguageReply initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> DualLanguageReplyConfig:
        labels_raw = raw.get("labels", {}) or {}
        primary_label_raw = labels_raw.get("primary_label", "English")
        secondary_label_raw = labels_raw.get("secondary_label", "Hindi")

        primary_label = sanitize_label(primary_label_raw)
        secondary_label = sanitize_label(secondary_label_raw)

        separator_line = raw.get("separator_line", "-" * 40)
        if not isinstance(separator_line, str) or not separator_line.strip():
            logger.warning(
                "Invalid separator_line in config. Using default '----------'."
            )
            separator_line = "-" * 40

        show_language_labels = bool(raw.get("show_language_labels", True))

        labels = LanguageLabelConfig(
            primary_label=primary_label,
            secondary_label=secondary_label,
        )

        return DualLanguageReplyConfig(
            labels=labels,
            separator_line=separator_line,
            show_language_labels=show_language_labels,
        )

    # ---------------------------- Formatting Logic ---------------------------

    def _format_with_labels(
        self, primary_text: str, secondary_text: Optional[str] = None
    ) -> str:
        """
        Format reply with explicit language labels.
        """
        lines = []

        # Primary section
        primary_label = self._config.labels.primary_label
        lines.append(f"[{primary_label}]")
        lines.append(primary_text)

        # Separator
        lines.append(self._config.separator_line)

        # Secondary section
        secondary_label = self._config.labels.secondary_label
        if secondary_text is not None:
            lines.append(f"[{secondary_label}]")
            lines.append(secondary_text)
        else:
            # Explicit placeholder for Phase-1 to show that translation
            # was not generated automatically.
            lines.append(f"[{secondary_label}]")
            lines.append("(No secondary version provided in Phase-1.)")

        return "\n".join(lines)

    def _format_without_labels(
        self, primary_text: str, secondary_text: Optional[str] = None
    ) -> str:
        """
        Format reply without explicit language labels (minimal style).
        """
        lines = [primary_text, self._config.separator_line]

        if secondary_text is not None:
            lines.append(secondary_text)
        else:
            lines.append("(Secondary language version not provided.)")

        return "\n".join(lines)

    # --------------------------- Public Interface ----------------------------

    def build_dual_reply(
        self,
        primary_text: str,
        secondary_hint: Optional[str] = None,
    ) -> str:
        """
        Build a dual-language formatted reply.

        Parameters
        ----------
        primary_text : str
            The main reply text (e.g., in English).
        secondary_hint : Optional[str]
            Optional manually provided text or hint for the secondary language.
            In Phase-1, this is NOT auto-translated, just formatted.

        Returns
        -------
        str
            Formatted dual-language reply.
        """
        validate_text(primary_text)
        primary_clean = sanitize_text(primary_text)
        secondary_clean = (
            sanitize_text(secondary_hint) if secondary_hint is not None else None
        )

        if self._config.show_language_labels:
            formatted = self._format_with_labels(primary_clean, secondary_clean)
        else:
            formatted = self._format_without_labels(primary_clean, secondary_clean)

        logger.info("Built dual language reply (has_secondary=%s).", secondary_clean is not None)
        return formatted

    # ------------------------------ Introspection -----------------------------

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a debug snapshot of internal configuration.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "primary_label": self._config.labels.primary_label,
            "secondary_label": self._config.labels.secondary_label,
            "separator_line": self._config.separator_line,
            "show_language_labels": self._config.show_language_labels,
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for DualLanguageReply...")

    dlr = DualLanguageReply(config_path=None)

    base_reply = "Here is your requested information about the order status."
    secondary_hint = "Yeh aapke order status ki jankari hai."

    print("=== With secondary hint ===")
    print(dlr.build_dual_reply(base_reply, secondary_hint))

    print("\n=== Without secondary hint ===")
    print(dlr.build_dual_reply(base_reply, None))

    debug_info = dlr.debug_state()
    logger.info("DualLanguageReply debug state: %s", debug_info)

    logger.info("Phase-1 manual test for DualLanguageReply completed.")
