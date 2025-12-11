"""
================================================================================
AumCore_AI - Adaptive Prompt Builder
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: adaptive_prompt_builder.py

Description:
    Enterprise-grade Phase-1 implementation of an adaptive prompt builder
    used to construct prompts based on high-level "intent" and "context"
    information, using only rule-based logic.

    This module is designed as a foundational building block for later phases
    (Chunk-3 to Chunk-8), but in Phase-1 it is strictly limited to:

    - Chunk-1:
        * PEP8-compliant imports
        * Module metadata
        * Logging setup
        * Clear class structure

    - Chunk-2:
        * YAML config loader (optional)
        * Input sanitization helpers
        * Input validation helpers
        * Custom error hierarchy for prompt building
        * Safe rule-based fallback behavior

    IMPORTANT:
        - No model loading is allowed in Phase-1.
        - All behavior must be deterministic and rule-based.
        - This file is intentionally verbose and structured to support
          future extension without breaking the foundation.

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
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple

import yaml  # Phase-1 allows config handling, but no model loading

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "adaptive_prompt_builder"
__description__ = (
    "Phase-1 rule-based Adaptive Prompt Builder with Chunk-1 and Chunk-2 only."
)

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("AdaptivePromptBuilder")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class PromptBuilderError(Exception):
    """
    Base class for all prompt builder errors.

    This is intentionally broad so that higher-level systems can catch
    any prompt builder related issue with a single exception type, while
    still allowing more specific subclasses to be used where necessary.
    """

    def __init__(self, message: str, *, code: str = "PROMPT_BUILDER_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidIntentError(PromptBuilderError):
    """
    Raised when an invalid or empty intent value is provided.
    """

    def __init__(self, message: str = "Intent must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_INTENT")


class InvalidContextError(PromptBuilderError):
    """
    Raised when an invalid or empty context value is provided.
    """

    def __init__(self, message: str = "Context must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_CONTEXT")


class TemplateNotFoundError(PromptBuilderError):
    """
    Raised when no template exists for the given intent, and no fallback
    strategy can be applied.
    """

    def __init__(self, intent: str):
        super().__init__(
            f"No prompt template available for intent: '{intent}'",
            code="TEMPLATE_NOT_FOUND",
        )


class TemplateFormatError(PromptBuilderError):
    """
    Raised when a template string is malformed or missing required placeholders.
    """

    def __init__(self, template: str, message: str):
        super().__init__(
            f"Template format error: {message}. Template: {template!r}",
            code="TEMPLATE_FORMAT_ERROR",
        )


# ==============================================================================
# Chunk-2: Configuration Structures
# ==============================================================================


@dataclass
class PromptTemplateConfig:
    """
    Represents configuration for a single prompt template.

    Attributes
    ----------
    intent : str
        The logical intent identifier, normalized to lowercase.
    template : str
        The template string, which may contain placeholders like `{context}`.
    description : Optional[str]
        Human-friendly description of what this template is used for.
    """

    intent: str
    template: str
    description: Optional[str] = None


@dataclass
class PromptBuilderConfig:
    """
    Represents the overall configuration for AdaptivePromptBuilder.

    Attributes
    ----------
    templates : Dict[str, PromptTemplateConfig]
        Mapping of intent -> template configuration.
    fallback_template : str
        The default template used when no intent-specific template is available.
    """

    templates: Dict[str, PromptTemplateConfig]
    fallback_template: str = "Respond appropriately: {context}"


# ==============================================================================
# Chunk-2: Sanitization Utilities
# ==============================================================================


def sanitize_whitespace(text: str) -> str:
    """
    Normalize and collapse whitespace in a string.

    - Converts multiple spaces/tabs/newlines into a single space.
    - Strips leading and trailing whitespace.

    Parameters
    ----------
    text : str
        Input text to sanitize.

    Returns
    -------
    str
        Sanitized text.
    """
    if not isinstance(text, str):
        raise ValueError("sanitize_whitespace expects a string.")

    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def sanitize_placeholder_conflicts(text: str) -> str:
    """
    Remove or escape raw braces that might conflict with template placeholders.

    For Phase-1, we take a conservative approach:
    - Remove stray '{' and '}' that are not part of the `{context}` placeholder.
    """
    if not isinstance(text, str):
        raise ValueError("sanitize_placeholder_conflicts expects a string.")

    # Very simple handling: remove standalone braces
    cleaned = text.replace("{", "").replace("}", "")
    # Re-introduce the known placeholder, if it existed conceptually
    # (Phase-1 keeps it simple, we don't attempt advanced parsing.)
    if "context" in text and "{context}" not in cleaned:
        cleaned = cleaned.replace("context", "{context}")
    return cleaned


def sanitize_text(text: str) -> str:
    """
    High-level sanitization for general text inputs.

    Combines multiple sanitization operations into a single call for convenience.
    """
    text = sanitize_whitespace(text)
    text = sanitize_placeholder_conflicts(text)
    return text


# ==============================================================================
# Chunk-2: Validation Utilities
# ==============================================================================


def validate_non_empty_string(value: Any, *, field_name: str, min_length: int = 1):
    """
    Validate that a value is a non-empty string with a minimum length.

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field being validated (for error messages).
    min_length : int, optional
        Minimum required length for the string, by default 1.

    Raises
    ------
    PromptBuilderError
        If the value is invalid.
    """
    if not isinstance(value, str):
        raise PromptBuilderError(
            f"{field_name} must be a string, got: {type(value).__name__}",
            code="VALIDATION_ERROR",
        )
    if not value.strip():
        raise PromptBuilderError(
            f"{field_name} must be a non-empty string.", code="VALIDATION_ERROR"
        )
    if len(value.strip()) < min_length:
        raise PromptBuilderError(
            f"{field_name} must be at least {min_length} characters long.",
            code="VALIDATION_ERROR",
        )


def validate_intent(intent: str):
    """
    Validate an intent string.

    Rules (Phase-1):
    - Must be a non-empty string.
    - Must have at least 2 visible characters.
    """
    validate_non_empty_string(intent, field_name="Intent", min_length=2)


def validate_context(context: str):
    """
    Validate a context string.

    Rules (Phase-1):
    - Must be a non-empty string.
    - Must have at least 2 visible characters.
    """
    validate_non_empty_string(context, field_name="Context", min_length=2)


def validate_template(template: str):
    """
    Validate that the template is at least syntactically acceptable.

    For Phase-1:
    - Must be a non-empty string.
    - Should contain '{context}' if it expects dynamic context insertion.
      (We allow templates without it but log a warning.)
    """
    validate_non_empty_string(template, field_name="Template", min_length=4)


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file if it exists.

    For Phase-1:
    - Failure to load a config file does not break the system.
    - We log a warning and return an empty dict instead.

    The YAML file MAY include:
    - `fallback_template`: string
    - `templates`: mapping of intent -> { template: str, description: str }

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary (or empty dict on error).
    """
    if not path:
        logger.info("No config path provided. Using in-memory defaults only.")
        return {}

    if not os.path.exists(path):
        logger.warning("Config file not found at '%s'. Using defaults.", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded prompt builder config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load YAML config from '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-1 + Chunk-2: Adaptive Prompt Builder (Phase-1 Only)
# ==============================================================================


class AdaptivePromptBuilder:
    """
    Phase-1 compliant Adaptive Prompt Builder.

    Responsibilities (Phase-1):
    - Maintain a mapping from intent -> prompt template.
    - Provide a simple `build_prompt(intent, context)` method.
    - Apply input validation and sanitization.
    - Fall back to a rule-based default template when necessary.
    - NEVER load or call any model.

    This class is intentionally over-structured and documented to support
    future layering (Chunk-3 to Chunk-8) without needing to rewrite the
    foundation.
    """

    # ----------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AdaptivePromptBuilder.

        Parameters
        ----------
        config_path : Optional[str], optional
            Optional path to a YAML configuration file that defines initial
            templates and fallback behavior. If not provided or not found,
            the builder uses internal defaults.
        """
        logger.info("Initializing AdaptivePromptBuilder (Phase-1)...")

        raw_config = load_yaml_config(config_path) if config_path else {}
        self._config: PromptBuilderConfig = self._build_config_from_raw(raw_config)

        # Internal storage for templates: intent -> template string
        self._templates: Dict[str, str] = {
            intent: cfg.template for intent, cfg in self._config.templates.items()
        }

        logger.info("AdaptivePromptBuilder initialized successfully.")

    # ------------------------- Config Building Helpers ------------------------

    def _build_config_from_raw(self, raw: Dict[str, Any]) -> PromptBuilderConfig:
        """
        Build a PromptBuilderConfig from a raw dictionary loaded from YAML.

        This isolates YAML-specific structure from the rest of the code,
        making it easier to refactor or extend in later phases.
        """
        templates: Dict[str, PromptTemplateConfig] = {}
        fallback_template = raw.get("fallback_template", "Respond appropriately: {context}")

        raw_templates = raw.get("templates", {}) or {}
        if not isinstance(raw_templates, dict):
            logger.warning("Invalid 'templates' section in config. Ignoring it.")
            raw_templates = {}

        for intent, entry in raw_templates.items():
            try:
                if not isinstance(entry, dict):
                    logger.warning(
                        "Skipping non-dict template config for intent '%s'.", intent
                    )
                    continue

                template_str = entry.get("template")
                description = entry.get("description")

                if not template_str:
                    logger.warning(
                        "Skipping template for intent '%s' because 'template' is missing.",
                        intent,
                    )
                    continue

                validate_intent(intent)
                validate_template(template_str)

                normalized_intent = self._normalize_intent(intent)
                sanitized_template = sanitize_text(template_str)

                templates[normalized_intent] = PromptTemplateConfig(
                    intent=normalized_intent,
                    template=sanitized_template,
                    description=description,
                )
            except PromptBuilderError as exc:
                logger.warning(
                    "Skipping invalid template config for intent '%s': %s", intent, exc
                )

        # Validate and sanitize fallback template
        try:
            validate_template(fallback_template)
            fallback_template = sanitize_text(fallback_template)
        except PromptBuilderError as exc:
            logger.warning(
                "Invalid fallback template in config (%s). Using hardcoded default.", exc
            )
            fallback_template = "Respond appropriately: {context}"

        return PromptBuilderConfig(templates=templates, fallback_template=fallback_template)

    # ------------------------------ Normalization -----------------------------

    @staticmethod
    def _normalize_intent(intent: str) -> str:
        """
        Normalize an intent to a canonical form used as dictionary key.

        For Phase-1:
        - Lowercase the string.
        - Strip whitespace.
        """
        validate_intent(intent)
        return sanitize_whitespace(intent).lower()

    # ------------------------------- Add Template -----------------------------

    def add_template(
        self,
        intent: str,
        template: str,
        *,
        description: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Register or update a template for a given intent.

        Parameters
        ----------
        intent : str
            High-level intent name (e.g., "greeting", "farewell").
        template : str
            Template string with optional `{context}` placeholder.
        description : Optional[str], optional
            Human-readable description for documentation.
        overwrite : bool, optional
            Whether to replace existing template for the same intent.

        Raises
        ------
        PromptBuilderError
            If validation fails.
        """
        validate_intent(intent)
        validate_template(template)

        normalized_intent = self._normalize_intent(intent)
        sanitized_template = sanitize_text(template)

        if not overwrite and normalized_intent in self._templates:
            logger.info(
                "Template for intent '%s' already exists and overwrite=False. Skipping.",
                normalized_intent,
            )
            return

        self._templates[normalized_intent] = sanitized_template
        self._config.templates[normalized_intent] = PromptTemplateConfig(
            intent=normalized_intent, template=sanitized_template, description=description
        )

        logger.info("Template registered for intent '%s'.", normalized_intent)

    # ------------------------------- Get Template -----------------------------

    def get_template(self, intent: str) -> Optional[str]:
        """
        Get the template associated with an intent, if any.

        Parameters
        ----------
        intent : str
            Intent identifier.

        Returns
        -------
        Optional[str]
            Template string or None if not found.
        """
        normalized_intent = self._normalize_intent(intent)
        return self._templates.get(normalized_intent)

    def has_template(self, intent: str) -> bool:
        """
        Check whether a template is registered for the given intent.
        """
        normalized_intent = self._normalize_intent(intent)
        return normalized_intent in self._templates

    def list_intents(self) -> List[str]:
        """
        Return a list of all known intents in normalized form.
        """
        return sorted(self._templates.keys())

    # ------------------------------ Fallback Logic ----------------------------

    def _get_effective_template(self, intent: str) -> Tuple[str, bool]:
        """
        Internal helper to determine which template to use for a given intent.

        Returns
        -------
        Tuple[str, bool]
            (template_string, is_fallback)
        """
        normalized_intent = self._normalize_intent(intent)
        template = self._templates.get(normalized_intent)

        if template:
            return template, False

        logger.warning(
            "No template found for intent '%s'. Using fallback template.",
            normalized_intent,
        )
        return self._config.fallback_template, True

    # ------------------------------- Build Prompt -----------------------------

    def build_prompt(self, intent: str, context: str) -> str:
        """
        Build a prompt string from the given intent and context.

        For Phase-1:
        - This is a pure string-based transformation.
        - It does not call any model or external service.

        Parameters
        ----------
        intent : str
            Logical intent for the desired behavior.
        context : str
            Free-form context that is inserted into the template.

        Returns
        -------
        str
            Fully rendered prompt string.

        Raises
        ------
        PromptBuilderError
            If validation fails or template is malformed.
        """
        validate_intent(intent)
        validate_context(context)

        normalized_intent = self._normalize_intent(intent)
        sanitized_context = sanitize_text(context)

        template, is_fallback = self._get_effective_template(normalized_intent)

        # Simple template check: ensure "{context}" placeholder exists
        if "{context}" not in template:
            logger.warning(
                "Template for intent '%s' does not contain '{context}' placeholder.",
                normalized_intent,
            )

        try:
            prompt = template.replace("{context}", sanitized_context)
        except Exception as exc:
            raise TemplateFormatError(template, f"Failed to render template: {exc}")

        if is_fallback:
            logger.info(
                "Built prompt using fallback template for intent '%s'.", normalized_intent
            )
        else:
            logger.info(
                "Built prompt using registered template for intent '%s'.",
                normalized_intent,
            )

        return prompt

    # ------------------------------ Introspection -----------------------------

    def describe_intent(self, intent: str) -> Optional[str]:
        """
        Return the description for a given intent, if available.
        """
        normalized_intent = self._normalize_intent(intent)
        cfg = self._config.templates.get(normalized_intent)
        return cfg.description if cfg else None

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a debug snapshot of internal state.

        This is safe to log or inspect in Phase-1 and can be
        extended in later phases.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "intents": self.list_intents(),
            "fallback_template": self._config.fallback_template,
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for AdaptivePromptBuilder...")

    builder = AdaptivePromptBuilder(config_path=None)

    # Register some basic templates
    builder.add_template(
        "greeting",
        "Hello! How can I help you with {context}?",
        description="General greeting prompt.",
    )
    builder.add_template(
        "farewell",
        "Goodbye! Take care of {context}.",
        description="General farewell prompt.",
    )

    # Build prompts using known and unknown intents
    print(builder.build_prompt("greeting", "your query"))
    print(builder.build_prompt("unknown_intent", "some task"))
    print(builder.build_prompt("farewell", "your project"))

    # Show debug state
    debug_info = builder.debug_state()
    logger.info("Debug state: %s", debug_info)

    logger.info("Phase-1 manual test for AdaptivePromptBuilder completed.")
