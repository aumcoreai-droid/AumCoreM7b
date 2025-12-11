"""
================================================================================
AumCore_AI - Task Understanding Engine
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: task_understanding.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    Task Understanding Engine. This module converts raw NLU-style signals
    (intent, keywords, simple NLU payload) into a structured "task spec"
    that higher layers can execute.

    Phase-1 constraints:
        - No ML models, no planners, no external APIs
        - No embeddings, no semantic planning graphs
        - Pure rule-based mapping from NLU → TaskSpec
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Accept a normalized NLU payload:
            * text
            * intent.name + score
            * keywords
            * topic info (optional)
        - Infer a coarse-grained task_type, e.g.:
            * "answer_question"
            * "summarize"
            * "generate_code"
            * "small_talk"
            * "unknown"
        - Attach additional task flags and parameters
        - Produce a deterministic, inspectable TaskSpec dictionary

    Future phases may add:
        - Multi-step planning and tool selection
        - Priority and cost estimation
        - User-profile-aware task shaping

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

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "task_understanding"
__description__ = (
    "Phase-1 rule-based Task Understanding Engine with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TaskUnderstandingEngine")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class TaskUnderstandingError(Exception):
    """
    Base class for all Task Understanding engine errors.
    """

    def __init__(self, message: str, *, code: str = "TASK_UNDERSTANDING_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidNLUPayloadError(TaskUnderstandingError):
    """
    Raised when the provided NLU payload is invalid.
    """

    def __init__(
        self,
        message: str = "NLU payload must be a valid non-empty dict with at least 'text'.",
    ):
        super().__init__(message, code="INVALID_NLU_PAYLOAD")


class InvalidConfigError(TaskUnderstandingError):
    """
    Raised when the YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid Task Understanding configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class IntentToTaskRule:
    """
    Rule mapping from intent (and optional topic) to task_type.

    Attributes
    ----------
    intent : str
        Canonical intent name (lowercase).
    task_type : str
        High-level task type label.
    min_score : float
        Minimum intent score needed to trigger this rule.
    topic : Optional[str]
        Optional topic filter (match only when topic.current_topic == topic).
    description : Optional[str]
        Optional human-readable description.
    """

    intent: str
    task_type: str
    min_score: float = 0.0
    topic: Optional[str] = None
    description: Optional[str] = None


@dataclass
class KeywordTaskRule:
    """
    Rule mapping from keyword presence to task_type.

    Attributes
    ----------
    task_type : str
        High-level task type label.
    required_keywords : List[str]
        Keywords that must all appear (case-insensitive).
    any_keywords : List[str]
        At least one of these keywords should appear (if non-empty).
    description : Optional[str]
        Optional human-readable description.
    """

    task_type: str
    required_keywords: List[str]
    any_keywords: List[str]
    description: Optional[str] = None


@dataclass
class TaskUnderstandingConfig:
    """
    Overall configuration for the Task Understanding engine.

    Attributes
    ----------
    default_task_type : str
        Fallback task type when no rule matches.
    min_intent_score_for_intent_rules : float
        Global minimum intent score to even consider intent-based rules.
    intent_rules : List[IntentToTaskRule]
        Ordered list of intent-based rules.
    keyword_rules : List[KeywordTaskRule]
        Ordered list of keyword-based rules.
    """

    default_task_type: str
    min_intent_score_for_intent_rules: float
    intent_rules: List[IntentToTaskRule]
    keyword_rules: List[KeywordTaskRule]


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for Task Understanding Engine, if present.

    Expected structure (Phase-1, optional):

        default_task_type: "answer_question"
        min_intent_score_for_intent_rules: 0.2

        intent_rules:
          - intent: "ask_status"
            task_type: "answer_question"
            min_score: 0.3
            topic: "shopping"
            description: "Order status questions in shopping topic."
          - intent: "summarize"
            task_type: "summarize"
            min_score: 0.2

        keyword_rules:
          - task_type: "generate_code"
            required_keywords: ["code"]
            any_keywords: ["python", "function", "class"]
            description: "Code generation tasks."
          - task_type: "small_talk"
            required_keywords: []
            any_keywords: ["how are you", "what's up"]
            description: "Casual chitchat."

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
        logger.info("No config path provided for TaskUnderstandingEngine. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Task understanding config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded Task Understanding config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error(
            "Failed to load Task Understanding YAML config '%s': %s", path, exc
        )
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def _sanitize_str(value: Any, field_name: str) -> str:
    """
    Ensure a value is a non-empty string.
    """
    if not isinstance(value, str):
        raise InvalidConfigError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise InvalidConfigError(f"{field_name} must not be empty.")
    return cleaned


def sanitize_intent_name(intent: Any) -> str:
    """
    Normalize intent names to lowercase.
    """
    if intent is None:
        return ""
    return str(intent).strip().lower()


def sanitize_topic_name(topic: Any) -> str:
    """
    Normalize topic names to lowercase.
    """
    if topic is None:
        return ""
    return str(topic).strip().lower()


def sanitize_keyword(keyword: Any) -> str:
    """
    Normalize keywords to lowercase (used for matching).
    """
    if keyword is None:
        return ""
    return str(keyword).strip().lower()


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def validate_nlu_payload(payload: Any) -> Dict[str, Any]:
    """
    Validate that payload is a non-empty dict with at least 'text'.

    Returns normalized payload.
    """
    if not isinstance(payload, dict):
        raise InvalidNLUPayloadError("NLU payload must be a dict.")

    if not payload:
        raise InvalidNLUPayloadError("NLU payload must not be empty.")

    if "text" not in payload:
        raise InvalidNLUPayloadError("NLU payload must contain 'text' field.")

    return payload


def _validate_float(value: Any, default: float, name: str) -> float:
    try:
        fvalue = float(value)
    except Exception:
        logger.warning(
            "Invalid %s value %r. Falling back to default %.3f.", name, value, default
        )
        return default
    return fvalue


def _validate_float_0_1(value: Any, default: float, name: str) -> float:
    fvalue = _validate_float(value, default, name)
    if fvalue < 0.0:
        logger.warning("%s < 0.0. Clamping to 0.0.", name)
        fvalue = 0.0
    elif fvalue > 1.0:
        logger.warning("%s > 1.0. Clamping to 1.0.", name)
        fvalue = 1.0
    return fvalue


def _validate_list(value: Any, default: List[Any], name: str) -> List[Any]:
    if value is None:
        return default
    if not isinstance(value, list):
        logger.warning(
            "Invalid %s value %r (not a list). Using default.", name, value
        )
        return default
    return value


# ==============================================================================
# Chunk-1 + Chunk-2: Task Understanding Engine (Phase-1 Only)
# ==============================================================================


class TaskUnderstandingEngine:
    """
    Phase-1 rule-based Task Understanding engine.

    Input (typical NLU payload shape):

        {
            "text": "Can you give me a Python function?",
            "intent": {
                "name": "generate_code",
                "score": 0.85
            },
            "keywords": ["python", "function", "code"],
            "topic": {
                "current_topic": "coding",
                "is_topic_switch": true
            }
        }

    Output (TaskSpec example):

        {
            "task_type": "generate_code",
            "reason": "intent_rule",
            "intent": {...original intent dict...},
            "topic": {...original topic dict...},
            "flags": {
                "needs_nlg": true,
                "needs_nlu": false,
                "is_small_talk": false,
                "is_question": true
            },
            "parameters": {
                "raw_text": "...",
                "keywords": [...],
                "inferred_language": "python"
            }
        }

    The engine does not execute tasks; it only clarifies WHAT should be done.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the TaskUnderstandingEngine.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing TaskUnderstandingEngine (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: TaskUnderstandingConfig = self._build_config_from_raw(raw_config)

        logger.info(
            "TaskUnderstandingEngine configured: default_task_type=%r, intent_rules=%d, keyword_rules=%d",
            self._config.default_task_type,
            len(self._config.intent_rules),
            len(self._config.keyword_rules),
        )

        logger.info("TaskUnderstandingEngine initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> TaskUnderstandingConfig:
        default_task_type_raw = raw.get("default_task_type", "answer_question")
        min_intent_score_raw = raw.get("min_intent_score_for_intent_rules", 0.2)

        try:
            default_task_type = _sanitize_str(default_task_type_raw, "default_task_type")
        except InvalidConfigError as exc:
            logger.warning("%s Falling back to 'answer_question'.", exc)
            default_task_type = "answer_question"

        min_intent_score = _validate_float_0_1(
            min_intent_score_raw,
            default=0.2,
            name="min_intent_score_for_intent_rules",
        )

        intent_rules_raw = _validate_list(
            raw.get("intent_rules", []), default=[], name="intent_rules"
        )
        keyword_rules_raw = _validate_list(
            raw.get("keyword_rules", []), default=[], name="keyword_rules"
        )

        intent_rules: List[IntentToTaskRule] = []
        for entry in intent_rules_raw:
            if not isinstance(entry, dict):
                logger.warning("Skipping invalid intent_rule (not dict): %r", entry)
                continue

            try:
                intent_name = sanitize_intent_name(entry.get("intent"))
                task_type = _sanitize_str(entry.get("task_type", ""), "task_type")
                min_score = _validate_float_0_1(
                    entry.get("min_score", 0.0), default=0.0, name="min_score"
                )
                topic = sanitize_topic_name(entry.get("topic")) or None
                description = entry.get("description")
                if not intent_name:
                    logger.warning("Skipping intent_rule with empty intent: %r", entry)
                    continue

                intent_rules.append(
                    IntentToTaskRule(
                        intent=intent_name,
                        task_type=task_type,
                        min_score=min_score,
                        topic=topic,
                        description=str(description).strip()
                        if description is not None
                        else None,
                    )
                )
            except InvalidConfigError as exc:
                logger.warning("Skipping invalid intent_rule %r: %s", entry, exc)

        keyword_rules: List[KeywordTaskRule] = []
        for entry in keyword_rules_raw:
            if not isinstance(entry, dict):
                logger.warning("Skipping invalid keyword_rule (not dict): %r", entry)
                continue

            try:
                task_type = _sanitize_str(entry.get("task_type", ""), "task_type")
                required_kw_raw = _validate_list(
                    entry.get("required_keywords", []),
                    default=[],
                    name="required_keywords",
                )
                any_kw_raw = _validate_list(
                    entry.get("any_keywords", []),
                    default=[],
                    name="any_keywords",
                )
                description = entry.get("description")

                required_kws = [sanitize_keyword(k) for k in required_kw_raw if k]
                any_kws = [sanitize_keyword(k) for k in any_kw_raw if k]

                keyword_rules.append(
                    KeywordTaskRule(
                        task_type=task_type,
                        required_keywords=required_kws,
                        any_keywords=any_kws,
                        description=str(description).strip()
                        if description is not None
                        else None,
                    )
                )
            except InvalidConfigError as exc:
                logger.warning("Skipping invalid keyword_rule %r: %s", entry, exc)

        return TaskUnderstandingConfig(
            default_task_type=default_task_type,
            min_intent_score_for_intent_rules=min_intent_score,
            intent_rules=intent_rules,
            keyword_rules=keyword_rules,
        )

    # ------------------------------ Core Helpers -----------------------------

    @staticmethod
    def _normalize_keywords(nlu_payload: Dict[str, Any]) -> List[str]:
        """
        Extract and normalize keywords from NLU payload.
        """
        raw = nlu_payload.get("keywords", [])
        if not isinstance(raw, list):
            return []
        return [sanitize_keyword(k) for k in raw if k is not None]

    @staticmethod
    def _extract_intent_info(nlu_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract intent info safely from NLU payload.
        """
        intent_raw = nlu_payload.get("intent", {}) or {}
        if not isinstance(intent_raw, dict):
            intent_raw = {}

        name = sanitize_intent_name(intent_raw.get("name"))
        score = 0.0
        try:
            score = float(intent_raw.get("score", 0.0))
        except Exception:
            score = 0.0

        return {"name": name, "score": score}

    @staticmethod
    def _extract_topic_info(nlu_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract topic info safely from NLU payload.
        """
        topic_raw = nlu_payload.get("topic", {}) or {}
        if not isinstance(topic_raw, dict):
            topic_raw = {}

        current_topic = sanitize_topic_name(topic_raw.get("current_topic"))
        is_switch = topic_raw.get("is_topic_switch", None)

        if is_switch not in (True, False):
            is_switch = None

        return {
            "current_topic": current_topic or None,
            "is_topic_switch": is_switch,
        }

    @staticmethod
    def _is_question(text: str) -> bool:
        """
        Simple heuristic: detect if text is a question.
        """
        if "?" in text:
            return True

        lower = text.lower().strip()
        question_starts = [
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "which ",
            "who ",
            "whom ",
            "whose ",
            "can ",
            "could ",
            "should ",
            "would ",
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
        ]
        for prefix in question_starts:
            if lower.startswith(prefix):
                return True
        return False

    # ------------------------------ Rule Matching ----------------------------

    def _match_intent_rule(
        self,
        intent_info: Dict[str, Any],
        topic_info: Dict[str, Any],
    ) -> Optional[IntentToTaskRule]:
        """
        Try to match an intent-based rule.

        Conditions:
            - intent score >= min_intent_score_for_intent_rules
            - rule.min_score <= intent score
            - if rule.topic is set, must match topic.current_topic
        """
        name = intent_info.get("name") or ""
        score = float(intent_info.get("score") or 0.0)

        if not name:
            return None

        if score < self._config.min_intent_score_for_intent_rules:
            return None

        topic_name = topic_info.get("current_topic") or ""

        for rule in self._config.intent_rules:
            if rule.intent != name:
                continue
            if score < rule.min_score:
                continue
            if rule.topic:
                if rule.topic != topic_name:
                    continue
            return rule

        return None

    def _match_keyword_rule(
        self,
        keywords: List[str],
    ) -> Optional[KeywordTaskRule]:
        """
        Try to match a keyword-based rule.

        Conditions:
            - All required_keywords present (if any)
            - At least one any_keywords present (if any_keywords non-empty)
        """
        if not keywords:
            return None

        keyword_set = set(k for k in keywords if k)

        for rule in self._config.keyword_rules:
            # Required keywords check
            if rule.required_keywords:
                if not all(k in keyword_set for k in rule.required_keywords):
                    continue

            # Any keywords check
            if rule.any_keywords:
                if not any(k in keyword_set for k in rule.any_keywords):
                    continue

            return rule

        return None

    # ---------------------- Task Type + Flags Derivation ---------------------

    def _derive_task_type(
        self,
        nlu_payload: Dict[str, Any],
        intent_info: Dict[str, Any],
        topic_info: Dict[str, Any],
        keywords: List[str],
    ) -> Dict[str, Any]:
        """
        Decide task_type + reason using intent and keyword rules.
        """
        # 1. Try intent rules
        intent_rule = self._match_intent_rule(intent_info, topic_info)
        if intent_rule is not None:
            return {
                "task_type": intent_rule.task_type,
                "reason": "intent_rule",
                "matched_rule": intent_rule,
            }

        # 2. Try keyword rules
        keyword_rule = self._match_keyword_rule(keywords)
        if keyword_rule is not None:
            return {
                "task_type": keyword_rule.task_type,
                "reason": "keyword_rule",
                "matched_rule": keyword_rule,
            }

        # 3. Simple fallback heuristics based on text form
        text = str(nlu_payload.get("text", "")).strip()
        if self._is_question(text):
            # Basic heuristic: treat as answer_question
            return {
                "task_type": "answer_question",
                "reason": "question_heuristic",
                "matched_rule": None,
            }

        # 4. Default
        return {
            "task_type": self._config.default_task_type,
            "reason": "default",
            "matched_rule": None,
        }

    def _derive_flags(
        self,
        task_type: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        Derive task flags (boolean properties) from task_type + text.
        """
        is_question = self._is_question(text)
        is_small_talk = task_type == "small_talk"
        needs_nlg = True  # In Phase-1, all tasks result in a textual reply
        needs_nlu = False  # Already done; kept for future multi-pass flows

        # More flags can be added in future phases
        return {
            "is_question": is_question,
            "is_small_talk": is_small_talk,
            "needs_nlg": needs_nlg,
            "needs_nlu": needs_nlu,
        }

    def _infer_language_from_keywords(self, keywords: List[str]) -> Optional[str]:
        """
        Very simple heuristic to guess programming language from keywords.
        """
        lowered = set(kw.lower() for kw in keywords)

        if "python" in lowered or "py" in lowered:
            return "python"
        if "javascript" in lowered or "js" in lowered or "node" in lowered:
            return "javascript"
        if "java" in lowered:
            return "java"
        if "c++" in lowered or "cpp" in lowered:
            return "cpp"
        if "c#" in lowered or "csharp" in lowered:
            return "csharp"

        return None

    # --------------------------- Public Interface ----------------------------

    def build_task_spec(self, nlu_payload: Any) -> Dict[str, Any]:
        """
        Build a TaskSpec from a normalized NLU payload.

        Parameters
        ----------
        nlu_payload : Any
            NLU analysis result (must be a dict with at least 'text').

        Returns
        -------
        Dict[str, Any]
            Task specification describing what the system should do next.
        """
        payload = validate_nlu_payload(nlu_payload)

        text = str(payload.get("text", "")).strip()
        intent_info = self._extract_intent_info(payload)
        topic_info = self._extract_topic_info(payload)
        keywords = self._normalize_keywords(payload)

        decision = self._derive_task_type(payload, intent_info, topic_info, keywords)
        task_type = str(decision.get("task_type") or self._config.default_task_type)
        reason = decision.get("reason") or "default"

        flags = self._derive_flags(task_type, text)
        inferred_language = self._infer_language_from_keywords(keywords)

        parameters: Dict[str, Any] = {
            "raw_text": text,
            "keywords": keywords,
        }
        if inferred_language:
            parameters["inferred_language"] = inferred_language

        task_spec: Dict[str, Any] = {
            "task_type": task_type,
            "reason": reason,
            "intent": intent_info,
            "topic": topic_info,
            "flags": flags,
            "parameters": parameters,
        }

        logger.info(
            "TaskUnderstandingEngine built task_spec: task_type=%r, reason=%r, intent=%r, topic=%r",
            task_type,
            reason,
            intent_info.get("name"),
            topic_info.get("current_topic"),
        )

        return task_spec

    # ------------------------------ Introspection -----------------------------

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of internal configuration.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "default_task_type": self._config.default_task_type,
            "min_intent_score_for_intent_rules": self._config.min_intent_score_for_intent_rules,
            "intent_rules_count": len(self._config.intent_rules),
            "keyword_rules_count": len(self._config.keyword_rules),
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for TaskUnderstandingEngine...")

    # Example in-memory configuration to demonstrate behavior
    raw_config = {
        "default_task_type": "answer_question",
        "min_intent_score_for_intent_rules": 0.2,
        "intent_rules": [
            {
                "intent": "ask_status",
                "task_type": "answer_question",
                "min_score": 0.3,
                "topic": "shopping",
                "description": "Order status questions in shopping topic.",
            },
            {
                "intent": "generate_code",
                "task_type": "generate_code",
                "min_score": 0.4,
                "description": "Code generation tasks.",
            },
            {
                "intent": "small_talk",
                "task_type": "small_talk",
                "min_score": 0.1,
                "description": "Casual small talk.",
            },
        ],
        "keyword_rules": [
            {
                "task_type": "generate_code",
                "required_keywords": ["code"],
                "any_keywords": ["python", "function", "class"],
                "description": "Code generation via keyword match.",
            },
            {
                "task_type": "summarize",
                "required_keywords": ["summary"],
                "any_keywords": [],
                "description": "Summarization request via keyword.",
            },
        ],
    }

    engine = TaskUnderstandingEngine(config_path=None)
    engine._config = engine._build_config_from_raw(raw_config)

    samples = [
        {
            "name": "order_status",
            "nlu": {
                "text": "What is the status of my order?",
                "intent": {"name": "ask_status", "score": 0.7},
                "keywords": ["order", "status", "track"],
                "topic": {"current_topic": "shopping", "is_topic_switch": False},
            },
        },
        {
            "name": "code_generation_intent",
            "nlu": {
                "text": "Can you write a Python function to sort a list?",
                "intent": {"name": "generate_code", "score": 0.9},
                "keywords": ["python", "function", "code", "sort"],
                "topic": {"current_topic": "coding", "is_topic_switch": True},
            },
        },
        {
            "name": "keyword_based_code_generation",
            "nlu": {
                "text": "I need some code in Python to parse JSON.",
                "intent": {"name": "unknown", "score": 0.1},
                "keywords": ["code", "python", "json"],
                "topic": {"current_topic": "coding", "is_topic_switch": False},
            },
        },
        {
            "name": "small_talk_intent",
            "nlu": {
                "text": "How are you doing today?",
                "intent": {"name": "small_talk", "score": 0.6},
                "keywords": ["how", "are", "you"],
                "topic": {"current_topic": "general", "is_topic_switch": False},
            },
        },
        {
            "name": "fallback_question_heuristic",
            "nlu": {
                "text": "When will it be delivered?",
                "intent": {"name": "unknown", "score": 0.0},
                "keywords": ["delivered"],
                "topic": {"current_topic": "", "is_topic_switch": None},
            },
        },
        {
            "name": "non_question_default",
            "nlu": {
                "text": "Thanks for your help.",
                "intent": {"name": "unknown", "score": 0.0},
                "keywords": ["thanks"],
                "topic": {"current_topic": "general", "is_topic_switch": False},
            },
        },
    ]

    for sample in samples:
        print("\n=== SAMPLE:", sample["name"], "===")
        spec = engine.build_task_spec(sample["nlu"])
        print("TaskSpec:")
        print("  task_type :", spec["task_type"])
        print("  reason    :", spec["reason"])
        print("  flags     :", spec["flags"])
        print("  params    :", spec["parameters"])

    debug_info = engine.debug_state()
    logger.info("TaskUnderstandingEngine debug state: %s", debug_info)

    logger.info("Phase-1 manual test for TaskUnderstandingEngine completed.")
