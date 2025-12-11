"""
================================================================================
AumCore_AI - NLU Engine (Rule-based Orchestrator)
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: nlu_engine.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    Natural Language Understanding (NLU) engine. This module acts as an
    orchestrator that wires together multiple low-level NLU components:

        - intent detection
        - keyword extraction
        - mini knowledge synthesis (over messages)
        - topic switch detection (optional, via previous_topic)

    Phase-1 constraints:
        - No ML models, no embeddings, no transformers
        - No external APIs, no vector search
        - Only deterministic rule-based submodules
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Provide a single `analyze` entrypoint for raw user text
        - Invoke underlying submodules in a safe, predictable order
        - Produce a structured NLU result payload
        - Handle missing/optional dependencies gracefully
        - Expose debug state for inspection

    Future phases may add:
        - Confidence blending between rule-based and model-based signals
        - Dialogue-level memory and user profile conditioning
        - Multilingual support with translation middleware

================================================================================
"""

# ==============================================================================
# Chunk-1: Imports, Metadata, Logging Setup
# ==============================================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import yaml

# Phase-1: we import the rule-based components.
# They are expected to be in the same package or accessible on PYTHONPATH.
try:
    from intent_detection import IntentDetection, IntentDetectionError
except Exception:  # pragma: no cover - defensive import
    IntentDetection = None  # type: ignore
    IntentDetectionError = Exception  # type: ignore

try:
    from keyword_extractor import KeywordExtractor, KeywordExtractorError
except Exception:  # pragma: no cover - defensive import
    KeywordExtractor = None  # type: ignore
    KeywordExtractorError = Exception  # type: ignore

try:
    from mini_knowledge_synthesizer import (
        MiniKnowledgeSynthesizer,
        KnowledgeSynthError,
    )
except Exception:  # pragma: no cover - defensive import
    MiniKnowledgeSynthesizer = None  # type: ignore
    KnowledgeSynthError = Exception  # type: ignore

try:
    from auto_topic_switch_detector import (
        AutoTopicSwitchDetector,
        TopicSwitchError,
    )
except Exception:  # pragma: no cover - defensive import
    AutoTopicSwitchDetector = None  # type: ignore
    TopicSwitchError = Exception  # type: ignore


__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "nlu_engine"
__description__ = (
    "Phase-1 rule-based NLU Engine orchestrator with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("NLUEngine")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class NLUEngineError(Exception):
    """
    Base class for all NLU engine errors.
    """

    def __init__(self, message: str, *, code: str = "NLU_ENGINE_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidTextError(NLUEngineError):
    """
    Raised when incoming user text is invalid.
    """

    def __init__(self, message: str = "Text must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_TEXT")


class InvalidConfigError(NLUEngineError):
    """
    Raised when NLU engine YAML configuration is invalid.
    """

    def __init__(self, message: str = "Invalid NLU engine configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class NLUSubmoduleFlags:
    """
    Flags to enable or disable individual NLU submodules.

    Attributes
    ----------
    use_intent_detection : bool
        Whether to run the intent detection component.
    use_keyword_extractor : bool
        Whether to run the keyword extractor.
    use_mini_knowledge_synth : bool
        Whether to run mini knowledge synthesizer on message history.
    use_topic_switch_detector : bool
        Whether to run topic switch detector (requires previous_topic).
    """

    use_intent_detection: bool = True
    use_keyword_extractor: bool = True
    use_mini_knowledge_synth: bool = True
    use_topic_switch_detector: bool = True


@dataclass
class NLUEngineConfig:
    """
    NLU engine configuration.

    Attributes
    ----------
    flags : NLUSubmoduleFlags
        Flags controlling which submodules to use.
    max_history_messages : int
        Maximum number of previous messages used for synthesis.
    """

    flags: NLUSubmoduleFlags
    max_history_messages: int = 5


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for NLU engine, if present.

    Expected structure (Phase-1, optional):

        flags:
          use_intent_detection: true
          use_keyword_extractor: true
          use_mini_knowledge_synth: true
          use_topic_switch_detector: true
        max_history_messages: 5

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
        logger.info("No config path provided for NLUEngine. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "NLU engine config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded NLU engine config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load NLU engine YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_text(text: str) -> str:
    """
    Sanitize user text.

    - Ensures string type
    - Strips leading/trailing whitespace
    """
    if not isinstance(text, str):
        raise InvalidTextError("Text must be a string.")
    cleaned = text.strip()
    return cleaned


def sanitize_history(history: Optional[List[str]]) -> List[str]:
    """
    Sanitize a list of previous user messages for knowledge synthesis.

    - Ensures a list of strings
    - Strips whitespace
    - Drops empty entries
    """
    if history is None:
        return []

    if not isinstance(history, list):
        raise InvalidTextError("History must be provided as a list of strings.")

    cleaned_list: List[str] = []
    for item in history:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            cleaned_list.append(stripped)
    return cleaned_list


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


def _validate_bool(value: Any, default: bool, name: str) -> bool:
    if isinstance(value, bool):
        return value
    logger.warning(
        "Invalid %s value %r. Falling back to default %s.", name, value, default
    )
    return default


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
# Chunk-1 + Chunk-2: NLU Engine (Phase-1 Only)
# ==============================================================================


class NLUEngine:
    """
    Phase-1 rule-based NLU engine.

    This class orchestrates the underlying rule-based components to provide
    a single NLU `analyze` API that returns a structured understanding
    of the user's message.

    Expected output structure (Phase-1):

        {
            "text": "<original text>",
            "intent": {
                "name": "<intent_name>",
                "score": <float>,
            },
            "keywords": ["kw1", "kw2", ...],
            "topic": {
                "current_topic": "<topic_name or None>",
                "is_topic_switch": <bool or None>,
            },
            "synthesis": {
                "key_points": [...],
                "supporting_details": [...],
                "open_questions": [...],
            }
        }

    Any section may contain None or empty defaults if the corresponding
    submodule is disabled or unavailable.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        intent_detector: Optional[Any] = None,
        keyword_extractor: Optional[Any] = None,
        knowledge_synthesizer: Optional[Any] = None,
        topic_switch_detector: Optional[Any] = None,
    ):
        """
        Initialize the NLUEngine.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        intent_detector : Optional[Any]
            Optional pre-initialized IntentDetection instance.
        keyword_extractor : Optional[Any]
            Optional pre-initialized KeywordExtractor instance.
        knowledge_synthesizer : Optional[Any]
            Optional pre-initialized MiniKnowledgeSynthesizer instance.
        topic_switch_detector : Optional[Any]
            Optional pre-initialized AutoTopicSwitchDetector instance.
        """
        logger.info("Initializing NLUEngine (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: NLUEngineConfig = self._build_config_from_raw(raw_config)

        # Submodules (can be injected or lazily created)
        self._intent_detector = intent_detector
        self._keyword_extractor = keyword_extractor
        self._knowledge_synthesizer = knowledge_synthesizer
        self._topic_switch_detector = topic_switch_detector

        # Lazily initialize missing submodules using defaults
        self._ensure_submodules()

        logger.info(
            "NLUEngine configured: intent=%s, keywords=%s, synth=%s, topic_switch=%s, max_history_messages=%d",
            self._config.flags.use_intent_detection,
            self._config.flags.use_keyword_extractor,
            self._config.flags.use_mini_knowledge_synth,
            self._config.flags.use_topic_switch_detector,
            self._config.max_history_messages,
        )

        logger.info("NLUEngine initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> NLUEngineConfig:
        flags_raw = raw.get("flags", {}) or {}

        use_intent = _validate_bool(
            flags_raw.get("use_intent_detection", True),
            default=True,
            name="use_intent_detection",
        )
        use_keywords = _validate_bool(
            flags_raw.get("use_keyword_extractor", True),
            default=True,
            name="use_keyword_extractor",
        )
        use_synth = _validate_bool(
            flags_raw.get("use_mini_knowledge_synth", True),
            default=True,
            name="use_mini_knowledge_synth",
        )
        use_topic = _validate_bool(
            flags_raw.get("use_topic_switch_detector", True),
            default=True,
            name="use_topic_switch_detector",
        )

        flags = NLUSubmoduleFlags(
            use_intent_detection=use_intent,
            use_keyword_extractor=use_keywords,
            use_mini_knowledge_synth=use_synth,
            use_topic_switch_detector=use_topic,
        )

        max_history = _validate_positive_int(
            raw.get("max_history_messages", 5),
            default=5,
            name="max_history_messages",
        )

        return NLUEngineConfig(flags=flags, max_history_messages=max_history)

    # -------------------------- Submodule Management -------------------------

    def _ensure_submodules(self) -> None:
        """
        Lazily initialize missing submodules, respecting flags.
        """
        if self._config.flags.use_intent_detection and self._intent_detector is None:
            if IntentDetection is not None:
                try:
                    self._intent_detector = IntentDetection(config_path=None)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed to initialize IntentDetection: %s", exc)
                    self._intent_detector = None
            else:
                logger.warning("IntentDetection class not available in environment.")

        if self._config.flags.use_keyword_extractor and self._keyword_extractor is None:
            if KeywordExtractor is not None:
                try:
                    self._keyword_extractor = KeywordExtractor(config_path=None)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed to initialize KeywordExtractor: %s", exc)
                    self._keyword_extractor = None
            else:
                logger.warning("KeywordExtractor class not available in environment.")

        if (
            self._config.flags.use_mini_knowledge_synth
            and self._knowledge_synthesizer is None
        ):
            if MiniKnowledgeSynthesizer is not None:
                try:
                    self._knowledge_synthesizer = MiniKnowledgeSynthesizer(
                        config_path=None
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "Failed to initialize MiniKnowledgeSynthesizer: %s", exc
                    )
                    self._knowledge_synthesizer = None
            else:
                logger.warning(
                    "MiniKnowledgeSynthesizer class not available in environment."
                )

        if (
            self._config.flags.use_topic_switch_detector
            and self._topic_switch_detector is None
        ):
            if AutoTopicSwitchDetector is not None:
                try:
                    self._topic_switch_detector = AutoTopicSwitchDetector(
                        config_path=None
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "Failed to initialize AutoTopicSwitchDetector: %s", exc
                    )
                    self._topic_switch_detector = None
            else:
                logger.warning(
                    "AutoTopicSwitchDetector class not available in environment."
                )

    # ------------------------------ Core Helpers -----------------------------

    def _run_intent_detection(self, text: str) -> Dict[str, Any]:
        """
        Run intent detection submodule if enabled and available.
        """
        if not self._config.flags.use_intent_detection or self._intent_detector is None:
            return {"name": None, "score": 0.0}

        try:
            intent, score = self._intent_detector.detect_intent(text)
            return {"name": intent, "score": float(score)}
        except IntentDetectionError as exc:
            logger.error("Intent detection failed: %s", exc)
            return {"name": None, "score": 0.0}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error in intent detection: %s", exc)
            return {"name": None, "score": 0.0}

    def _run_keyword_extractor(self, text: str) -> List[str]:
        """
        Run keyword extractor submodule if enabled and available.
        """
        if not self._config.flags.use_keyword_extractor or self._keyword_extractor is None:
            return []

        try:
            keywords = self._keyword_extractor.extract_keywords(text)
            return list(keywords)
        except KeywordExtractorError as exc:
            logger.error("Keyword extraction failed: %s", exc)
            return []
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error in keyword extraction: %s", exc)
            return []

    def _run_topic_switch_detector(
        self,
        previous_topic: Optional[str],
        text: str,
    ) -> Dict[str, Any]:
        """
        Run topic switch detector if enabled, available, and previous_topic provided.
        """
        if (
            not self._config.flags.use_topic_switch_detector
            or self._topic_switch_detector is None
        ):
            return {"current_topic": None, "is_topic_switch": None}

        if previous_topic is None:
            return {"current_topic": None, "is_topic_switch": None}

        try:
            current_topic = self._topic_switch_detector.detect_topic(text)
            is_switch = self._topic_switch_detector.is_topic_switch(
                previous_topic, text
            )
            return {"current_topic": current_topic, "is_topic_switch": bool(is_switch)}
        except TopicSwitchError as exc:
            logger.error("Topic switch detection failed: %s", exc)
            return {"current_topic": None, "is_topic_switch": None}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error in topic switch detection: %s", exc)
            return {"current_topic": None, "is_topic_switch": None}

    def _run_knowledge_synthesizer(
        self,
        history: List[str],
        current_text: str,
    ) -> Dict[str, Any]:
        """
        Run mini knowledge synthesizer over a slice of history + current message.
        """
        if (
            not self._config.flags.use_mini_knowledge_synth
            or self._knowledge_synthesizer is None
        ):
            return {"key_points": [], "supporting_details": [], "open_questions": []}

        fragments: List[str] = []

        # Add truncated history (most recent messages)
        if history:
            limited_history = history[-self._config.max_history_messages :]
            fragments.extend(limited_history)

        # Add current text
        fragments.append(current_text)

        try:
            synthesis = self._knowledge_synthesizer.synthesize(fragments)
            return {
                "key_points": synthesis.get("key_points", []),
                "supporting_details": synthesis.get("supporting_details", []),
                "open_questions": synthesis.get("open_questions", []),
            }
        except KnowledgeSynthError as exc:
            logger.error("Knowledge synthesis failed: %s", exc)
            return {"key_points": [], "supporting_details": [], "open_questions": []}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unexpected error in knowledge synthesis: %s", exc)
            return {"key_points": [], "supporting_details": [], "open_questions": []}

    # --------------------------- Public Interface ----------------------------

    def analyze(
        self,
        text: str,
        *,
        previous_topic: Optional[str] = None,
        history_messages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run full NLU pipeline over a user message.

        Parameters
        ----------
        text : str
            Current user message.
        previous_topic : Optional[str]
            Previously detected topic label, if any.
        history_messages : Optional[List[str]]
            Previous user messages in the conversation (for synthesis).

        Returns
        -------
        Dict[str, Any]
            Structured NLU result.
        """
        validate_text(text)
        cleaned_text = sanitize_text(text)
        cleaned_history = sanitize_history(history_messages)

        intent_info = self._run_intent_detection(cleaned_text)
        keywords = self._run_keyword_extractor(cleaned_text)
        topic_info = self._run_topic_switch_detector(previous_topic, cleaned_text)
        synthesis = self._run_knowledge_synthesizer(cleaned_history, cleaned_text)

        result: Dict[str, Any] = {
            "text": cleaned_text,
            "intent": intent_info,
            "keywords": keywords,
            "topic": topic_info,
            "synthesis": synthesis,
        }

        logger.info(
            "NLUEngine analysis complete: intent=%r, topic=%r, keywords_count=%d",
            intent_info.get("name"),
            topic_info.get("current_topic"),
            len(keywords),
        )

        return result

    # ------------------------------ Introspection -----------------------------

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of internal configuration and submodules.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "flags": {
                "use_intent_detection": self._config.flags.use_intent_detection,
                "use_keyword_extractor": self._config.flags.use_keyword_extractor,
                "use_mini_knowledge_synth": self._config.flags.use_mini_knowledge_synth,
                "use_topic_switch_detector": self._config.flags.use_topic_switch_detector,
            },
            "max_history_messages": self._config.max_history_messages,
            "submodules_available": {
                "intent_detector": self._intent_detector is not None,
                "keyword_extractor": self._keyword_extractor is not None,
                "knowledge_synthesizer": self._knowledge_synthesizer is not None,
                "topic_switch_detector": self._topic_switch_detector is not None,
            },
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for NLUEngine...")

    # For manual test, we rely on default constructors of submodules.
    engine = NLUEngine(config_path=None)

    history = [
        "Hi, I want to check my order.",
        "The tracking page is not updating.",
    ]
    prev_topic = "shopping"

    samples = [
        "Can you tell me the status of my order?",
        "By the way, I also want to book a flight.",
    ]

    for msg in samples:
        result = engine.analyze(
            text=msg,
            previous_topic=prev_topic,
            history_messages=history,
        )
        print("\nMESSAGE:", msg)
        print("NLU RESULT:")
        print("  Intent :", result["intent"])
        print("  Keywords :", result["keywords"])
        print("  Topic :", result["topic"])
        print("  Key Points :", result["synthesis"]["key_points"])
        print("  Open Questions :", result["synthesis"]["open_questions"])

        # Update prev_topic and history for next iteration (simulating dialogue)
        prev_topic = result["topic"]["current_topic"] or prev_topic
        history.append(msg)

    debug_info = engine.debug_state()
    logger.info("NLUEngine debug state: %s", debug_info)

    logger.info("Phase-1 manual test for NLUEngine completed.")
