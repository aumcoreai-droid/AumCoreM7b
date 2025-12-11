"""
================================================================================
AumCore_AI - Role Mode Switcher
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: role_mode_switcher.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    Role Mode Switcher. This module manages the current "role mode" of the
    assistant (e.g., helper, explainer, coder, summarizer) using a simple,
    inspectable state + rules system.

    Phase-1 constraints:
        - No ML models, no transformers
        - No embeddings, no intent-based RL policies
        - Pure rule-based matching on explicit events and hints
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Maintain a finite set of role modes in configuration
        - Track the current active role mode
        - Switch modes based on explicit events and optional hints
        - Expose a deterministic public API for querying/updating role mode
        - Provide debug state for higher layers (NLU/NLG/Orchestrator)

    Future phases may add:
        - Intent + topic based automatic role switching
        - User preference and profile-driven role priority
        - Confidence-based blending between multiple roles

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
__module_name__ = "role_mode_switcher"
__description__ = (
    "Phase-1 rule-based Role Mode Switcher with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("RoleModeSwitcher")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class RoleModeError(Exception):
    """
    Base class for all role mode switcher errors.
    """

    def __init__(self, message: str, *, code: str = "ROLE_MODE_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidRoleError(RoleModeError):
    """
    Raised when an invalid role name is referenced.
    """

    def __init__(self, message: str = "Invalid role name."):
        super().__init__(message, code="INVALID_ROLE")


class InvalidEventError(RoleModeError):
    """
    Raised when an invalid event is used for switching.
    """

    def __init__(self, message: str = "Invalid role switch event."):
        super().__init__(message, code="INVALID_EVENT")


class InvalidConfigError(RoleModeError):
    """
    Raised when the YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid role mode configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class RoleConfig:
    """
    Represents a single role configuration.

    Attributes
    ----------
    name : str
        Canonical role name (lowercase).
    description : Optional[str]
        Optional human-readable description for observability.
    category : Optional[str]
        Optional grouping (e.g., "nlu", "nlg", "system").
    """

    name: str
    description: Optional[str] = None
    category: Optional[str] = None


@dataclass
class RoleTransitionRule:
    """
    Represents a deterministic rule for switching roles.

    Attributes
    ----------
    from_role : str
        Role name from which the transition is allowed ("*" for any).
    event : str
        Symbolic event name (e.g., "user_request_code", "internal_error").
    to_role : str
        Target role name.
    description : Optional[str]
        Optional human-readable description of the rule.
    """

    from_role: str
    event: str
    to_role: str
    description: Optional[str] = None


@dataclass
class RoleModeSwitcherConfig:
    """
    Overall configuration for the role mode switcher.

    Attributes
    ----------
    roles : Dict[str, RoleConfig]
        Mapping of role name -> RoleConfig.
    transitions : List[RoleTransitionRule]
        List of deterministic role switch rules.
    initial_role : str
        Default initial role when the system starts.
    """

    roles: Dict[str, RoleConfig]
    transitions: List[RoleTransitionRule]
    initial_role: str = "assistant"


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for role mode switcher, if present.

    Expected structure (Phase-1, optional):

        initial_role: "assistant"
        roles:
          assistant:
            description: "General-purpose helper mode"
            category: "core"
          explainer:
            description: "Explain concepts and reasoning"
            category: "nlu"
          coder:
            description: "Code generation and refactoring"
            category: "nlg"
          summarizer:
            description: "Summarize content"
            category: "nlu"

        transitions:
          - from_role: "*"
            event: "user_request_explanation"
            to_role: "explainer"
            description: "Switch to explainer when user asks for explanation."
          - from_role: "*"
            event: "user_request_code"
            to_role: "coder"
          - from_role: "*"
            event: "user_request_summary"
            to_role: "summarizer"
          - from_role: "*"
            event: "back_to_assistant"
            to_role: "assistant"

    Parameters
    ----------
    path : Optional[str]
        Path to YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed dictionary or empty on error.
    """
    if not path:
        logger.info("No config path provided for RoleModeSwitcher. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Role mode switcher config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded role mode switcher config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error(
            "Failed to load role mode switcher YAML config '%s': %s", path, exc
        )
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_role_name(name: str) -> str:
    """
    Normalize role names to canonical lowercase form.
    """
    if not isinstance(name, str):
        raise InvalidRoleError("Role name must be a string.")
    cleaned = name.strip().lower()
    if not cleaned:
        raise InvalidRoleError("Role name must not be empty.")
    return cleaned


def sanitize_event_name(event: str) -> str:
    """
    Normalize event names to canonical lowercase form.
    """
    if not isinstance(event, str):
        raise InvalidEventError("Event name must be a string.")
    cleaned = event.strip().lower()
    if not cleaned:
        raise InvalidEventError("Event name must not be empty.")
    return cleaned


def sanitize_optional_str(value: Any) -> Optional[str]:
    """
    Convert value to a trimmed string or None.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    cleaned = value.strip()
    return cleaned or None


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def _validate_roles_section(raw_roles: Any) -> Dict[str, Any]:
    """
    Validate that roles section is a dict.
    """
    if raw_roles is None:
        return {}
    if not isinstance(raw_roles, dict):
        logger.warning(
            "Invalid 'roles' section in role mode config. Expected dict, got %s.",
            type(raw_roles).__name__,
        )
        return {}
    return raw_roles


def _validate_transitions_section(raw_transitions: Any) -> List[Dict[str, Any]]:
    """
    Validate that transitions section is a list of dicts.
    """
    if raw_transitions is None:
        return []
    if not isinstance(raw_transitions, list):
        logger.warning(
            "Invalid 'transitions' section in role mode config. Expected list, got %s.",
            type(raw_transitions).__name__,
        )
        return []
    result: List[Dict[str, Any]] = []
    for item in raw_transitions:
        if isinstance(item, dict):
            result.append(item)
        else:
            logger.warning(
                "Skipping invalid transition entry (not dict): %r", item
            )
    return result


def _validate_initial_role(initial_role: Any, roles: Dict[str, RoleConfig]) -> str:
    """
    Validate initial role; fall back to 'assistant' or any available role.
    """
    try:
        cleaned = sanitize_role_name(str(initial_role))
    except InvalidRoleError:
        cleaned = "assistant"

    if cleaned in roles:
        return cleaned

    if "assistant" in roles:
        logger.warning(
            "Initial role '%s' not in configured roles. Falling back to 'assistant'.",
            cleaned,
        )
        return "assistant"

    if roles:
        fallback = sorted(roles.keys())[0]
        logger.warning(
            "Initial role '%s' not in configured roles. Falling back to '%s'.",
            cleaned,
            fallback,
        )
        return fallback

    # No roles at all; fall back to hardcoded 'assistant'
    return "assistant"


# ==============================================================================
# Chunk-1 + Chunk-2: Role Mode Switcher (Phase-1 Only)
# ==============================================================================


class RoleModeSwitcher:
    """
    Phase-1 rule-based role mode switcher.

    Core behavior:
        - Maintains a current role name in memory.
        - Accepts symbolic events to trigger transitions.
        - Uses deterministic rules to select the next role.

    The module does not perform any NLU/NLG itself; it is purely
    a state + rules machine to coordinate higher-level behavior.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RoleModeSwitcher.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing RoleModeSwitcher (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: RoleModeSwitcherConfig = self._build_config_from_raw(raw_config)

        self._current_role: str = self._config.initial_role

        logger.info(
            "RoleModeSwitcher initialized with initial_role=%r, roles=%d, transitions=%d",
            self._current_role,
            len(self._config.roles),
            len(self._config.transitions),
        )

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> RoleModeSwitcherConfig:
        raw_roles = _validate_roles_section(raw.get("roles"))
        raw_transitions = _validate_transitions_section(raw.get("transitions"))
        initial_role_raw = raw.get("initial_role", "assistant")

        roles: Dict[str, RoleConfig] = {}

        for name, entry in raw_roles.items():
            if not isinstance(entry, dict):
                logger.warning(
                    "Role '%s' entry must be a dict. Skipping.", name
                )
                continue

            try:
                normalized_name = sanitize_role_name(name)
            except InvalidRoleError as exc:
                logger.warning(
                    "Skipping invalid role name %r: %s", name, exc
                )
                continue

            description = sanitize_optional_str(entry.get("description"))
            category = sanitize_optional_str(entry.get("category"))

            roles[normalized_name] = RoleConfig(
                name=normalized_name,
                description=description,
                category=category,
            )

        transitions: List[RoleTransitionRule] = []

        for t in raw_transitions:
            from_raw = t.get("from_role", "*")
            event_raw = t.get("event")
            to_raw = t.get("to_role")
            desc_raw = t.get("description")

            if event_raw is None or to_raw is None:
                logger.warning(
                    "Transition missing required fields 'event' or 'to_role': %r. Skipping.",
                    t,
                )
                continue

            try:
                from_role = from_raw.strip().lower() if isinstance(from_raw, str) else "*"
                if from_role != "*":
                    from_role = sanitize_role_name(from_role)

                event_name = sanitize_event_name(event_raw)
                to_role = sanitize_role_name(to_raw)

            except (InvalidRoleError, InvalidEventError) as exc:
                logger.warning(
                    "Skipping invalid transition %r due to error: %s", t, exc
                )
                continue

            description = sanitize_optional_str(desc_raw)

            transitions.append(
                RoleTransitionRule(
                    from_role=from_role,
                    event=event_name,
                    to_role=to_role,
                    description=description,
                )
            )

        initial_role = _validate_initial_role(initial_role_raw, roles)

        if not roles:
            logger.warning(
                "No roles configured. Creating default 'assistant' role."
            )
            roles["assistant"] = RoleConfig(
                name="assistant",
                description="Default assistant role",
                category="core",
            )
            if initial_role not in roles:
                initial_role = "assistant"

        return RoleModeSwitcherConfig(
            roles=roles,
            transitions=transitions,
            initial_role=initial_role,
        )

    # ------------------------------ Core Logic -------------------------------

    def get_current_role(self) -> str:
        """
        Return the current active role name.
        """
        return self._current_role

    def list_roles(self) -> List[str]:
        """
        Return sorted list of role names.
        """
        return sorted(self._config.roles.keys())

    def list_transitions(self) -> List[Dict[str, str]]:
        """
        Return a list of configured transitions in dict form.
        """
        result: List[Dict[str, str]] = []
        for t in self._config.transitions:
            result.append(
                {
                    "from_role": t.from_role,
                    "event": t.event,
                    "to_role": t.to_role,
                }
            )
        return result

    def _find_transition(self, from_role: str, event: str) -> Optional[RoleTransitionRule]:
        """
        Find a matching transition given a current role and event.

        Matching priority:
            1. Specific rule: from_role == current_role AND event == event
            2. Wildcard rule: from_role == "*" AND event == event
        """
        normalized_from = sanitize_role_name(from_role)
        normalized_event = sanitize_event_name(event)

        # 1. Specific match
        for rule in self._config.transitions:
            if rule.from_role != "*" and rule.from_role == normalized_from and rule.event == normalized_event:
                return rule

        # 2. Wildcard match
        for rule in self._config.transitions:
            if rule.from_role == "*" and rule.event == normalized_event:
                return rule

        return None

    def switch_role(self, event: str) -> str:
        """
        Apply an event to potentially switch the current role.

        Parameters
        ----------
        event : str
            Symbolic event name driving transitions.

        Returns
        -------
        str
            The new current role name (may be unchanged if no rule applies).
        """
        if not isinstance(event, str):
            raise InvalidEventError("Event must be a string.")

        normalized_event = sanitize_event_name(event)
        current = self._current_role

        rule = self._find_transition(current, normalized_event)
        if rule is None:
            logger.info(
                "No role transition rule matched for current_role=%r, event=%r. Staying in same role.",
                current,
                normalized_event,
            )
            return current

        if rule.to_role not in self._config.roles:
            logger.warning(
                "Transition rule leads to undefined role '%s'. Ignoring transition.",
                rule.to_role,
            )
            return current

        prev = self._current_role
        self._current_role = rule.to_role

        logger.info(
            "Role transition: '%s' --(%s)--> '%s'",
            prev,
            normalized_event,
            self._current_role,
        )
        return self._current_role

    def force_set_role(self, role: str) -> str:
        """
        Forcefully set the current role, bypassing rules.

        Parameters
        ----------
        role : str
            Target role name.

        Returns
        -------
        str
            The new current role name.

        Raises
        ------
        InvalidRoleError
            If the requested role is not configured.
        """
        normalized_role = sanitize_role_name(role)
        if normalized_role not in self._config.roles:
            raise InvalidRoleError(
                f"Role '{normalized_role}' is not defined in configuration."
            )

        prev = self._current_role
        self._current_role = normalized_role

        logger.info(
            "Role forcibly changed from '%s' to '%s'.", prev, self._current_role
        )
        return self._current_role

    # ------------------------------ Introspection -----------------------------

    def describe_role(self, role: str) -> Optional[Dict[str, Any]]:
        """
        Return metadata for a given role, if available.
        """
        try:
            normalized = sanitize_role_name(role)
        except InvalidRoleError:
            return None

        cfg = self._config.roles.get(normalized)
        if cfg is None:
            return None

        return {
            "name": cfg.name,
            "description": cfg.description,
            "category": cfg.category,
        }

    def debug_state(self) -> Dict[str, Any]:
        """
        Return debug snapshot of internal configuration and state.
        """
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "current_role": self._current_role,
            "roles": self.list_roles(),
            "transitions": self.list_transitions(),
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for RoleModeSwitcher...")

    # Example in-memory configuration (when no YAML is supplied)
    raw_config = {
        "initial_role": "assistant",
        "roles": {
            "assistant": {
                "description": "General-purpose helper mode",
                "category": "core",
            },
            "explainer": {
                "description": "Explain concepts and reasoning",
                "category": "nlu",
            },
            "coder": {
                "description": "Code generation and refactoring",
                "category": "nlg",
            },
            "summarizer": {
                "description": "Summarize content",
                "category": "nlu",
            },
        },
        "transitions": [
            {
                "from_role": "*",
                "event": "user_request_explanation",
                "to_role": "explainer",
                "description": "Switch to explainer when user asks for explanation.",
            },
            {
                "from_role": "*",
                "event": "user_request_code",
                "to_role": "coder",
                "description": "Switch to coder when user requests code.",
            },
            {
                "from_role": "*",
                "event": "user_request_summary",
                "to_role": "summarizer",
                "description": "Switch to summarizer when user requests summary.",
            },
            {
                "from_role": "*",
                "event": "back_to_assistant",
                "to_role": "assistant",
                "description": "Return to default assistant role.",
            },
        ],
    }

    switcher = RoleModeSwitcher(config_path=None)
    switcher._config = switcher._build_config_from_raw(raw_config)
    switcher._current_role = switcher._config.initial_role

    print("Initial role:", switcher.get_current_role())

    events = [
        "user_request_explanation",
        "user_request_code",
        "user_request_summary",
        "back_to_assistant",
        "unknown_event",
    ]

    for ev in events:
        new_role = switcher.switch_role(ev)
        print(f"Event: {ev!r} -> current_role: {new_role!r}")

    # Force role set
    try:
        switcher.force_set_role("coder")
        print("Force set role to 'coder':", switcher.get_current_role())
    except InvalidRoleError as exc:
        print("Error forcing role:", exc)

    debug_info = switcher.debug_state()
    logger.info("RoleModeSwitcher debug state: %s", debug_info)

    logger.info("Phase-1 manual test for RoleModeSwitcher completed.")
