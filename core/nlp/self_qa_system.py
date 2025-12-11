"""
================================================================================
AumCore_AI - Self QA System
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: self_qa_system.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    Self QA System. This module lets the assistant "question and check"
    its own planned response using simple, handcrafted heuristics.

    IMPORTANT:
        - This is NOT model-on-model evaluation.
        - This is a pure rule-based sanity checker + metadata annotator.
        - It is designed to be cheap, predictable, and fully inspectable.

    Phase-1 constraints:
        - No ML models, no embeddings, no external APIs
        - No statistical scoring, no RL-based reward models
        - Pure rule-based checks on structured response objects
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Accept a structured "draft_response" payload from NLG / orchestrator
        - Run a fixed set of QA "questions" (checks) over the payload
        - Produce a structured QA report:
            * pass/fail per check
            * severity (info/warn/error)
            * aggregated verdict (ok / needs_review / reject)
        - Provide debug state for introspection

    Future phases may add:
        - Model-based quality scoring
        - Domain-specific check plugins
        - Learning from user feedback loops

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
__module_name__ = "self_qa_system"
__description__ = (
    "Phase-1 rule-based Self QA System with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("SelfQASystem")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class SelfQAError(Exception):
    """
    Base class for all Self QA system errors.
    """

    def __init__(self, message: str, *, code: str = "SELF_QA_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidPayloadError(SelfQAError):
    """
    Raised when the draft response payload is invalid.
    """

    def __init__(
        self,
        message: str = "Draft response payload must be a valid non-empty dict.",
    ):
        super().__init__(message, code="INVALID_PAYLOAD")


class InvalidConfigError(SelfQAError):
    """
    Raised when the YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid Self QA configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class QAThresholds:
    """
    Configuration for thresholds used in checks.

    Attributes
    ----------
    min_response_length : int
        Minimum allowed character count for a valid response.
    max_response_length : int
        Maximum allowed character count (beyond this triggers warning).
    max_warning_checks_for_ok : int
        Maximum number of 'warn' results allowed for overall verdict 'ok'.
    reject_on_error : bool
        If True, any 'error' check results in verdict 'reject'.
    """

    min_response_length: int = 20
    max_response_length: int = 8000
    max_warning_checks_for_ok: int = 2
    reject_on_error: bool = True


@dataclass
class QACheckFlags:
    """
    Flags to enable or disable specific QA checks.

    Attributes
    ----------
    check_non_empty : bool
        Ensure the response text is non-empty.
    check_min_length : bool
        Ensure response length >= min_response_length.
    check_max_length : bool
        Warn if response length > max_response_length.
    check_has_main_text_field : bool
        Ensure payload has a primary 'text' field.
    check_has_metadata : bool
        Warn if metadata dict is missing or empty.
    check_safety_placeholders : bool
        Check for unresolved placeholders like '<TODO>' or '???'.
    """

    check_non_empty: bool = True
    check_min_length: bool = True
    check_max_length: bool = True
    check_has_main_text_field: bool = True
    check_has_metadata: bool = True
    check_safety_placeholders: bool = True


@dataclass
class SelfQAConfig:
    """
    Overall Self QA System configuration.

    Attributes
    ----------
    thresholds : QAThresholds
        Threshold values used by checks.
    flags : QACheckFlags
        Check enable/disable flags.
    """

    thresholds: QAThresholds
    flags: QACheckFlags


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for Self QA System, if present.

    Expected structure (Phase-1, optional):

        thresholds:
          min_response_length: 20
          max_response_length: 8000
          max_warning_checks_for_ok: 2
          reject_on_error: true

        flags:
          check_non_empty: true
          check_min_length: true
          check_max_length: true
          check_has_main_text_field: true
          check_has_metadata: true
          check_safety_placeholders: true

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
        logger.info("No config path provided for SelfQASystem. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Self QA config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded Self QA config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load Self QA YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_text_value(value: Any) -> str:
    """
    Convert a value to a trimmed string (for text fields).
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned


def sanitize_metadata(value: Any) -> Dict[str, Any]:
    """
    Normalize metadata to a dict.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    # Anything else -> wrap in a dict under "raw"
    return {"raw": value}


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def validate_payload(payload: Any) -> Dict[str, Any]:
    """
    Validate that payload is a non-empty dict.

    Returns the payload as dict if valid, else raises InvalidPayloadError.
    """
    if not isinstance(payload, dict):
        raise InvalidPayloadError("Draft response payload must be a dict.")
    if not payload:
        raise InvalidPayloadError("Draft response payload must not be empty.")
    return payload


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


# ==============================================================================
# Chunk-1 + Chunk-2: Self QA System (Phase-1 Only)
# ==============================================================================


class SelfQASystem:
    """
    Phase-1 rule-based Self QA System.

    High-level design:
        - The orchestrator or NLG module passes a structured payload, e.g.:

            {
                "text": "<assistant reply string>",
                "metadata": {
                    "intent": "answer_status",
                    "role": "assistant",
                    "nlu": {...},
                    "nlg_template_id": "answer_status_friendly",
                }
            }

        - SelfQASystem runs deterministic checks:
            * Is there a 'text' field? Is it non-empty?
            * Is the length within reasonable bounds?
            * Does it contain suspicious placeholders?
            * Is there at least some metadata?

        - Returns a QA report:

            {
                "verdict": "ok" | "needs_review" | "reject",
                "checks": [
                    {
                        "name": "non_empty",
                        "passed": true,
                        "severity": "error" | "warn" | "info",
                        "message": "..."
                    },
                    ...
                ]
            }

    The module does NOT modify the response; it only annotates it.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SelfQASystem.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing SelfQASystem (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: SelfQAConfig = self._build_config_from_raw(raw_config)

        logger.info(
            "SelfQASystem configured: min_len=%d, max_len=%d, max_warn_for_ok=%d, reject_on_error=%s",
            self._config.thresholds.min_response_length,
            self._config.thresholds.max_response_length,
            self._config.thresholds.max_warning_checks_for_ok,
            self._config.thresholds.reject_on_error,
        )

        logger.info("SelfQASystem initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> SelfQAConfig:
        thresholds_raw = raw.get("thresholds", {}) or {}
        flags_raw = raw.get("flags", {}) or {}

        min_len = _validate_positive_int(
            thresholds_raw.get("min_response_length", 20),
            default=20,
            name="min_response_length",
        )
        max_len = _validate_positive_int(
            thresholds_raw.get("max_response_length", 8000),
            default=8000,
            name="max_response_length",
        )
        max_warn_ok = _validate_positive_int(
            thresholds_raw.get("max_warning_checks_for_ok", 2),
            default=2,
            name="max_warning_checks_for_ok",
        )
        reject_on_error = _validate_bool(
            thresholds_raw.get("reject_on_error", True),
            default=True,
            name="reject_on_error",
        )

        thresholds = QAThresholds(
            min_response_length=min_len,
            max_response_length=max_len,
            max_warning_checks_for_ok=max_warn_ok,
            reject_on_error=reject_on_error,
        )

        flag_non_empty = _validate_bool(
            flags_raw.get("check_non_empty", True),
            default=True,
            name="check_non_empty",
        )
        flag_min_length = _validate_bool(
            flags_raw.get("check_min_length", True),
            default=True,
            name="check_min_length",
        )
        flag_max_length = _validate_bool(
            flags_raw.get("check_max_length", True),
            default=True,
            name="check_max_length",
        )
        flag_has_text_field = _validate_bool(
            flags_raw.get("check_has_main_text_field", True),
            default=True,
            name="check_has_main_text_field",
        )
        flag_has_metadata = _validate_bool(
            flags_raw.get("check_has_metadata", True),
            default=True,
            name="check_has_metadata",
        )
        flag_placeholders = _validate_bool(
            flags_raw.get("check_safety_placeholders", True),
            default=True,
            name="check_safety_placeholders",
        )

        flags = QACheckFlags(
            check_non_empty=flag_non_empty,
            check_min_length=flag_min_length,
            check_max_length=flag_max_length,
            check_has_main_text_field=flag_has_text_field,
            check_has_metadata=flag_has_metadata,
            check_safety_placeholders=flag_placeholders,
        )

        return SelfQAConfig(thresholds=thresholds, flags=flags)

    # ------------------------------ Core Checks ------------------------------

    def _get_text_and_metadata(self, payload: Dict[str, Any]) -> (str, Dict[str, Any]):
        """
        Extract the primary text and metadata dict from payload.
        """
        text_raw = payload.get("text", "")
        meta_raw = payload.get("metadata", {})

        text = sanitize_text_value(text_raw)
        metadata = sanitize_metadata(meta_raw)
        return text, metadata

    def _check_has_main_text_field(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check: payload has 'text' key.
        """
        passed = "text" in payload
        result = {
            "name": "has_main_text_field",
            "passed": bool(passed),
            "severity": "error",
            "message": "Payload contains 'text' field." if passed else "Payload is missing 'text' field.",
        }
        return result

    def _check_non_empty(self, text: str) -> Dict[str, Any]:
        """
        Check: text is non-empty.
        """
        passed = len(text) > 0
        result = {
            "name": "non_empty",
            "passed": bool(passed),
            "severity": "error",
            "message": "Response text is non-empty." if passed else "Response text is empty.",
        }
        return result

    def _check_min_length(self, text: str) -> Dict[str, Any]:
        """
        Check: text length >= min_response_length.
        """
        min_len = self._config.thresholds.min_response_length
        length = len(text)
        passed = length >= min_len
        result = {
            "name": "min_length",
            "passed": bool(passed),
            "severity": "warn",
            "message": (
                f"Response length {length} >= minimum {min_len}."
                if passed
                else f"Response length {length} is below minimum {min_len}."
            ),
        }
        return result

    def _check_max_length(self, text: str) -> Dict[str, Any]:
        """
        Check: warn if text length > max_response_length.
        """
        max_len = self._config.thresholds.max_response_length
        length = len(text)
        passed = length <= max_len
        result = {
            "name": "max_length",
            "passed": bool(passed),
            "severity": "warn",
            "message": (
                f"Response length {length} within max {max_len}."
                if passed
                else f"Response length {length} exceeds max {max_len}."
            ),
        }
        return result

    def _check_has_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check: metadata dict is present and not empty.
        """
        passed = bool(metadata)
        result = {
            "name": "has_metadata",
            "passed": bool(passed),
            "severity": "warn",
            "message": "Metadata is present." if passed else "Metadata is missing or empty.",
        }
        return result

    def _check_safety_placeholders(self, text: str) -> Dict[str, Any]:
        """
        Check: detect unresolved placeholders like '<TODO>' or '???'.
        """
        suspicious_markers = ["<todo>", "???", "TBD", "<TBD>", "<FILL>", "<REPLACE>"]
        lowered = text.lower()
        found_any = False
        found_tokens: List[str] = []

        for marker in suspicious_markers:
            if marker.lower() in lowered:
                found_any = True
                found_tokens.append(marker)

        passed = not found_any
        message = (
            "No unresolved placeholders found."
            if passed
            else f"Found unresolved placeholder markers: {', '.join(found_tokens)}."
        )

        result = {
            "name": "safety_placeholders",
            "passed": bool(passed),
            "severity": "error",
            "message": message,
        }
        return result

    # ------------------------------ Verdict Logic ----------------------------

    def _compute_verdict(self, checks: List[Dict[str, Any]]) -> str:
        """
        Compute overall verdict based on individual check results.

        Rules (Phase-1):
            - If any check with severity 'error' has passed == False
              AND reject_on_error == True -> verdict 'reject'
            - Else, count number of checks where passed == False and severity == 'warn':
                * if count <= max_warning_checks_for_ok -> 'ok'
                * else -> 'needs_review'
        """
        reject_on_error = self._config.thresholds.reject_on_error
        max_warn_for_ok = self._config.thresholds.max_warning_checks_for_ok

        error_failed = any(
            (not c.get("passed")) and c.get("severity") == "error" for c in checks
        )
        if reject_on_error and error_failed:
            return "reject"

        warn_fail_count = sum(
            1
            for c in checks
            if (not c.get("passed")) and c.get("severity") == "warn"
        )

        if warn_fail_count <= max_warn_for_ok:
            return "ok"

        return "needs_review"

    # --------------------------- Public Interface ----------------------------

    def evaluate(self, draft_response: Any) -> Dict[str, Any]:
        """
        Run Self QA checks over a draft response payload.

        Parameters
        ----------
        draft_response : Any
            Structured payload containing at least a 'text' field.

        Returns
        -------
        Dict[str, Any]
            QA report with fields:
                - verdict: 'ok' | 'needs_review' | 'reject'
                - checks: List[ {name, passed, severity, message} ]
        """
        payload = validate_payload(draft_response)

        text, metadata = self._get_text_and_metadata(payload)

        checks: List[Dict[str, Any]] = []

        if self._config.flags.check_has_main_text_field:
            checks.append(self._check_has_main_text_field(payload))

        if self._config.flags.check_non_empty:
            checks.append(self._check_non_empty(text))

        if self._config.flags.check_min_length:
            checks.append(self._check_min_length(text))

        if self._config.flags.check_max_length:
            checks.append(self._check_max_length(text))

        if self._config.flags.check_has_metadata:
            checks.append(self._check_has_metadata(metadata))

        if self._config.flags.check_safety_placeholders:
            checks.append(self._check_safety_placeholders(text))

        verdict = self._compute_verdict(checks)

        logger.info(
            "SelfQASystem evaluation completed: verdict=%s (checks=%d)",
            verdict,
            len(checks),
        )

        return {
            "verdict": verdict,
            "checks": checks,
        }

    # ------------------------------ Introspection -----------------------------

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of internal configuration.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "thresholds": {
                "min_response_length": self._config.thresholds.min_response_length,
                "max_response_length": self._config.thresholds.max_response_length,
                "max_warning_checks_for_ok": self._config.thresholds.max_warning_checks_for_ok,
                "reject_on_error": self._config.thresholds.reject_on_error,
            },
            "flags": {
                "check_non_empty": self._config.flags.check_non_empty,
                "check_min_length": self._config.flags.check_min_length,
                "check_max_length": self._config.flags.check_max_length,
                "check_has_main_text_field": self._config.flags.check_has_main_text_field,
                "check_has_metadata": self._config.flags.check_has_metadata,
                "check_safety_placeholders": self._config.flags.check_safety_placeholders,
            },
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for SelfQASystem...")

    qa = SelfQASystem(config_path=None)

    samples = [
        {
            "name": "good_response",
            "payload": {
                "text": "Here is the status of your order: it is currently in transit and should arrive tomorrow.",
                "metadata": {
                    "intent": "answer_status",
                    "role": "assistant",
                    "nlg_template_id": "answer_status_friendly",
                },
            },
        },
        {
            "name": "too_short",
            "payload": {
                "text": "Okay.",
                "metadata": {"intent": "ack"},
            },
        },
        {
            "name": "placeholder_issue",
            "payload": {
                "text": "Your order status is: <TODO>.",
                "metadata": {"intent": "answer_status"},
            },
        },
        {
            "name": "missing_text_field",
            "payload": {
                "content": "I forgot to use 'text' key here.",
                "metadata": {"intent": "misc"},
            },
        },
    ]

    for sample in samples:
        name = sample["name"]
        payload = sample["payload"]
        print("\n=== SAMPLE:", name, "===")
        try:
            report = qa.evaluate(payload)
            print("Verdict:", report["verdict"])
            print("Checks:")
            for c in report["checks"]:
                status = "PASS" if c["passed"] else "FAIL"
                print(
                    f"  - [{status}] {c['name']} ({c['severity']}): {c['message']}"
                )
        except SelfQAError as exc:
            print("SelfQAError for sample", name, ":", exc)

    debug_info = qa.debug_state()
    logger.info("SelfQASystem debug state: %s", debug_info)

    logger.info("Phase-1 manual test for SelfQASystem completed.")
