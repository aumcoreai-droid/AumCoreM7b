"""
================================================================================
AumCore_AI - NLG Engine (Template-based)
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: nlg_engine.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    Natural Language Generation (NLG) engine. This module focuses on
    template-based response construction using structured inputs
    (intent, slots, tone, metadata) and pure rule logic.

    Phase-1 constraints:
        - No language models, no transformers, no neural NLG
        - No sampling, no stochastic decoding
        - Pure template selection + slot filling
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Maintain a registry of templates per "response_type" and "tone"
        - Fill templates with provided slot values
        - Provide safe fallbacks when slots are missing
        - Generate deterministic responses from structured inputs
        - Offer debug introspection of configured templates

    Future phases may add:
        - Model-assisted template selection and variation
        - Style adaptation, personalization, and localization
        - Context-aware blending with retrieval or summarization

================================================================================
"""

# ==============================================================================
# Chunk-1: Imports, Metadata, Logging Setup
# ==============================================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from string import Template
from typing import Optional, Dict, Any, List

import yaml

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "nlg_engine"
__description__ = (
    "Phase-1 rule-based template-based NLG Engine with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("NLGEngine")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class NLGError(Exception):
    """
    Base class for all NLG engine errors.
    """

    def __init__(self, message: str, *, code: str = "NLG_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidTemplateError(NLGError):
    """
    Raised when a template is invalid or cannot be rendered.
    """

    def __init__(self, message: str = "Invalid NLG template."):
        super().__init__(message, code="INVALID_TEMPLATE")


class InvalidConfigError(NLGError):
    """
    Raised when YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid NLG engine configuration."):
        super().__init__(message, code="INVALID_CONFIG")


class InvalidInputError(NLGError):
    """
    Raised when input to NLG engine is missing or malformed.
    """

    def __init__(self, message: str = "Invalid NLG input payload."):
        super().__init__(message, code="INVALID_INPUT")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class TemplateEntry:
    """
    Represents a single NLG template.

    Attributes
    ----------
    id : str
        Unique identifier for the template.
    response_type : str
        Logical type, e.g., "greeting", "answer_status", "fallback".
    tone : str
        Tone label, e.g., "neutral", "friendly", "formal".
    text : str
        Template text compatible with string.Template.
    description : Optional[str]
        Optional human-readable description.
    """

    id: str
    response_type: str
    tone: str
    text: str
    description: Optional[str] = None


@dataclass
class NLGTemplatesConfig:
    """
    Configuration containing all templates.

    Attributes
    ----------
    templates : Dict[str, TemplateEntry]
        Mapping from template id to TemplateEntry.
    default_tone : str
        Tone used when none is provided.
    fallback_template_id : str
        Template identifier used when no suitable template is found.
    """

    templates: Dict[str, TemplateEntry] = field(default_factory=dict)
    default_tone: str = "neutral"
    fallback_template_id: str = "fallback_default"


@dataclass
class NLGEngineConfig:
    """
    Overall configuration for the NLG engine.

    Attributes
    ----------
    templates_config : NLGTemplatesConfig
        Template registry and defaults.
    """

    templates_config: NLGTemplatesConfig


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for NLG engine, if present.

    Expected structure (Phase-1, optional):

        default_tone: "neutral"
        fallback_template_id: "fallback_default"
        templates:
          greeting_neutral:
            response_type: "greeting"
            tone: "neutral"
            text: "Hello, ${user_name}! How can I help you today?"
            description: "Simple neutral greeting."
          answer_status_friendly:
            response_type: "answer_status"
            tone: "friendly"
            text: "Hey ${user_name}, your order ${order_id} is currently ${order_status}."
            description: "Friendly order status answer."
          fallback_default:
            response_type: "fallback"
            tone: "neutral"
            text: "I am not fully sure about that, but based on what I know: ${message}."
            description: "Generic fallback response."

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
        logger.info("No config path provided for NLGEngine. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "NLG config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded NLG config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load NLG YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def _sanitize_str_value(value: Any, field_name: str) -> str:
    """
    Ensure a value is a non-empty string.
    """
    if not isinstance(value, str):
        raise InvalidConfigError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise InvalidConfigError(f"{field_name} must not be empty.")
    return cleaned


def sanitize_tone(tone: str) -> str:
    """
    Normalize tone labels to lowercase.
    """
    if not isinstance(tone, str):
        raise InvalidInputError("Tone must be a string.")
    cleaned = tone.strip().lower()
    if not cleaned:
        raise InvalidInputError("Tone must not be empty.")
    return cleaned


def sanitize_response_type(response_type: str) -> str:
    """
    Normalize response_type labels to lowercase.
    """
    if not isinstance(response_type, str):
        raise InvalidInputError("response_type must be a string.")
    cleaned = response_type.strip().lower()
    if not cleaned:
        raise InvalidInputError("response_type must not be empty.")
    return cleaned


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def _validate_templates_structure(templates_raw: Any) -> Dict[str, Any]:
    """
    Validate that templates section is a dict.
    """
    if templates_raw is None:
        return {}
    if not isinstance(templates_raw, dict):
        logger.warning(
            "Invalid 'templates' section in NLG config. Expected dict, got %s.",
            type(templates_raw).__name__,
        )
        return {}
    return templates_raw


# ==============================================================================
# Chunk-1 + Chunk-2: NLG Engine (Phase-1 Only)
# ==============================================================================


class NLGEngine:
    """
    Phase-1 rule-based template-based NLG engine.

    Core behavior:
        - Select a template based on response_type and tone (with fallbacks).
        - Fill the template using .safe_substitute(slot_values).
        - Ensure deterministic and safe output even if slots are missing.

    No probabilistic behavior is allowed in Phase-1. All branching
    is controlled by rules and simple fallbacks.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the NLGEngine.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing NLGEngine (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: NLGEngineConfig = self._build_config_from_raw(raw_config)

        logger.info(
            "NLGEngine configured with %d templates, default_tone=%r, fallback_template_id=%r",
            len(self._config.templates_config.templates),
            self._config.templates_config.default_tone,
            self._config.templates_config.fallback_template_id,
        )

        logger.info("NLGEngine initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> NLGEngineConfig:
        templates_raw = _validate_templates_structure(raw.get("templates"))
        default_tone = raw.get("default_tone", "neutral")
        fallback_template_id = raw.get("fallback_template_id", "fallback_default")

        try:
            default_tone = _sanitize_str_value(default_tone, "default_tone").lower()
        except InvalidConfigError as exc:
            logger.warning("%s Falling back to 'neutral'.", exc)
            default_tone = "neutral"

        try:
            fallback_template_id = _sanitize_str_value(
                fallback_template_id, "fallback_template_id"
            )
        except InvalidConfigError as exc:
            logger.warning("%s Falling back to 'fallback_default'.", exc)
            fallback_template_id = "fallback_default"

        templates: Dict[str, TemplateEntry] = {}

        for tmpl_id, entry in templates_raw.items():
            if not isinstance(entry, dict):
                logger.warning(
                    "Template '%s' entry must be a dict. Skipping.", tmpl_id
                )
                continue

            try:
                id_clean = _sanitize_str_value(tmpl_id, "template id")
                response_type_raw = entry.get("response_type", "generic")
                tone_raw = entry.get("tone", default_tone)
                text_raw = entry.get("text", "")
                desc_raw = entry.get("description")

                response_type = _sanitize_str_value(
                    response_type_raw, "response_type"
                ).lower()
                tone = _sanitize_str_value(tone_raw, "tone").lower()
                text = _sanitize_str_value(text_raw, "text")

                description = None
                if desc_raw is not None:
                    description = str(desc_raw).strip() or None

                templates[id_clean] = TemplateEntry(
                    id=id_clean,
                    response_type=response_type,
                    tone=tone,
                    text=text,
                    description=description,
                )

            except InvalidConfigError as exc:
                logger.warning(
                    "Skipping invalid template config for '%s': %s", tmpl_id, exc
                )

        templates_config = NLGTemplatesConfig(
            templates=templates,
            default_tone=default_tone,
            fallback_template_id=fallback_template_id,
        )

        return NLGEngineConfig(templates_config=templates_config)

    # ------------------------------ Template Lookup --------------------------

    def _find_template_id(
        self,
        response_type: str,
        tone: str,
    ) -> Optional[str]:
        """
        Find a template id matching response_type + tone with fallbacks.

        Fallback order (Phase-1):
            1. Exact match (response_type, tone)
            2. Any template with same response_type (ignore tone)
            3. Fallback template id (as configured)
        """
        rt = sanitize_response_type(response_type)
        t = sanitize_tone(tone)

        # 1. Exact match
        for tmpl_id, tmpl in self._config.templates_config.templates.items():
            if tmpl.response_type == rt and tmpl.tone == t:
                return tmpl_id

        # 2. Same response_type, any tone
        for tmpl_id, tmpl in self._config.templates_config.templates.items():
            if tmpl.response_type == rt:
                return tmpl_id

        # 3. Fallback
        fallback_id = self._config.templates_config.fallback_template_id
        if fallback_id in self._config.templates_config.templates:
            return fallback_id

        logger.warning(
            "No suitable template found for response_type=%r, tone=%r and no valid fallback.",
            rt,
            t,
        )
        return None

    # ------------------------------ Rendering Logic --------------------------

    @staticmethod
    def _safe_fill_template(text: str, slots: Dict[str, Any]) -> str:
        """
        Safely fill a string.Template with provided slot values.

        - Uses safe_substitute to leave unknown placeholders as-is.
        - Converts non-string values to strings.
        """
        if slots is None:
            slots = {}
        safe_slots: Dict[str, str] = {}
        for key, value in slots.items():
            safe_slots[str(key)] = "" if value is None else str(value)

        tmpl = Template(text)
        try:
            rendered = tmpl.safe_substitute(safe_slots)
        except ValueError as exc:
            raise InvalidTemplateError(f"Error in template formatting: {exc}") from exc

        return rendered

    # --------------------------- Public Interface ----------------------------

    def generate(
        self,
        response_type: str,
        slots: Optional[Dict[str, Any]] = None,
        tone: Optional[str] = None,
    ) -> str:
        """
        Generate a response using template-based NLG.

        Parameters
        ----------
        response_type : str
            Logical response type (e.g., 'greeting', 'answer_status').
        slots : Optional[Dict[str, Any]]
            Slot values used to fill the template (e.g., user_name, order_id).
        tone : Optional[str]
            Desired tone label. If not provided, uses default_tone.

        Returns
        -------
        str
            Rendered response text.

        Raises
        ------
        NLGError
            If no template can be found or template rendering fails.
        """
        if tone is None:
            tone = self._config.templates_config.default_tone

        tmpl_id = self._find_template_id(response_type, tone)
        if tmpl_id is None:
            raise InvalidTemplateError(
                f"No template available for response_type={response_type!r}, tone={tone!r} "
                f"and no valid fallback."
            )

        tmpl_entry = self._config.templates_config.templates.get(tmpl_id)
        if tmpl_entry is None:
            raise InvalidTemplateError(
                f"Template id {tmpl_id!r} resolved but no entry found."
            )

        rendered = self._safe_fill_template(tmpl_entry.text, slots or {})

        logger.info(
            "Generated NLG response using template_id=%r, response_type=%r, tone=%r",
            tmpl_id,
            tmpl_entry.response_type,
            tmpl_entry.tone,
        )
        return rendered

    # ------------------------------ Introspection -----------------------------

    def list_templates(self) -> List[str]:
        """
        Return list of all template ids.
        """
        return sorted(self._config.templates_config.templates.keys())

    def get_template_info(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Return metadata for a given template id, if available.
        """
        entry = self._config.templates_config.templates.get(template_id)
        if not entry:
            return None
        return {
            "id": entry.id,
            "response_type": entry.response_type,
            "tone": entry.tone,
            "text": entry.text,
            "description": entry.description,
        }

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of internal configuration.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "default_tone": self._config.templates_config.default_tone,
            "fallback_template_id": self._config.templates_config.fallback_template_id,
            "template_ids": self.list_templates(),
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for NLGEngine...")

    # Example in-memory configuration (used if no YAML is supplied)
    raw_fallback_config = {
        "default_tone": "neutral",
        "fallback_template_id": "fallback_default",
        "templates": {
            "greeting_neutral": {
                "response_type": "greeting",
                "tone": "neutral",
                "text": "Hello, ${user_name}! How can I help you today?",
                "description": "Simple neutral greeting.",
            },
            "greeting_friendly": {
                "response_type": "greeting",
                "tone": "friendly",
                "text": "Hey ${user_name}! Good to see you here. What can I do for you?",
                "description": "Casual, friendly greeting.",
            },
            "answer_status_friendly": {
                "response_type": "answer_status",
                "tone": "friendly",
                "text": "Hey ${user_name}, your order ${order_id} is currently ${order_status}.",
                "description": "Friendly order status answer.",
            },
            "fallback_default": {
                "response_type": "fallback",
                "tone": "neutral",
                "text": "I am not fully sure about that, but based on what I know: ${message}.",
                "description": "Generic fallback response.",
            },
        },
    }

    engine = NLGEngine(config_path=None)
    engine._config = engine._build_config_from_raw(raw_fallback_config)

    samples = [
        {
            "response_type": "greeting",
            "tone": "neutral",
            "slots": {"user_name": "Aum"},
        },
        {
            "response_type": "greeting",
            "tone": "friendly",
            "slots": {"user_name": "Aum"},
        },
        {
            "response_type": "answer_status",
            "tone": "friendly",
            "slots": {
                "user_name": "Aum",
                "order_id": "ORD-12345",
                "order_status": "in transit",
            },
        },
        {
            "response_type": "unknown_type",
            "tone": "neutral",
            "slots": {"message": "I don't have enough information."},
        },
    ]

    for sample in samples:
        resp = engine.generate(
            response_type=sample["response_type"],
            slots=sample.get("slots", {}),
            tone=sample.get("tone"),
        )
        print(f"\nResponseType={sample['response_type']!r}, Tone={sample.get('tone')!r}")
        print(" ->", resp)

    debug_info = engine.debug_state()
    logger.info("NLGEngine debug state: %s", debug_info)

    logger.info("Phase-1 manual test for NLGEngine completed.")
