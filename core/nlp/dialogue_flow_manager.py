"""
================================================================================
AumCore_AI - Dialogue Flow Manager
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: dialogue_flow_manager.py

Description:
    Phase-1 enterprise-grade implementation of a rule-based dialogue flow manager.
    This module orchestrates simple dialogue "states" and transitions between them
    based on user messages and high-level events.

    Phase-1 constraints:
        - Pure rule-based logic
        - No ML-based policy, no RL, no transformers
        - Deterministic state machine behavior
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Maintain a finite set of dialogue states
        - Define allowed transitions between states
        - Decide the next state based on events
        - Provide a simple public API to advance the dialogue

    Future phases may add:
        - Intent-based transitions
        - Confidence scores
        - User profile influence
        - Model-based policy blending

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
__module_name__ = "dialogue_flow_manager"
__description__ = (
    "Phase-1 rule-based Dialogue Flow Manager with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("DialogueFlowManager")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class DialogueFlowError(Exception):
    def __init__(self, message: str, *, code: str = "DIALOGUE_FLOW_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self):
        return f"[{self.code}] {super().__str__()}"


class InvalidStateError(DialogueFlowError):
    def __init__(self, message="Invalid dialogue state."):
        super().__init__(message, code="INVALID_STATE")


class InvalidEventError(DialogueFlowError):
    def __init__(self, message="Invalid dialogue event."):
        super().__init__(message, code="INVALID_EVENT")


class InvalidConfigError(DialogueFlowError):
    def __init__(self, message="Invalid dialogue flow configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class DialogueStateConfig:
    name: str
    description: Optional[str] = None
    is_terminal: bool = False


@dataclass
class DialogueTransitionConfig:
    from_state: str
    event: str
    to_state: str
    description: Optional[str] = None


@dataclass
class DialogueFlowConfig:
    states: Dict[str, DialogueStateConfig]
    transitions: List[DialogueTransitionConfig]
    initial_state: str = "idle"


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        logger.info("No config path provided for DialogueFlowManager. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Dialogue flow config file not found at '%s'. Using defaults.", path
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded dialogue flow config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load dialogue flow YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def validate_state_name(name: str):
    if not isinstance(name, str):
        raise InvalidStateError("State name must be a string.")
    if not name.strip():
        raise InvalidStateError("State name must not be empty.")


def validate_event_name(event: str):
    if not isinstance(event, str):
        raise InvalidEventError("Event name must be a string.")
    if not event.strip():
        raise InvalidEventError("Event name must not be empty.")


# ==============================================================================
# Chunk-1 + Chunk-2: Dialogue Flow Manager (Phase-1 Only)
# ==============================================================================


class DialogueFlowManager:
    """
    Phase-1 rule-based dialogue flow manager.

    Implements a basic finite state machine (FSM) with:
        - Named states
        - Labeled events
        - Deterministic transitions
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        logger.info("Initializing DialogueFlowManager (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: DialogueFlowConfig = self._build_config_from_raw(raw_config)

        self._current_state: str = self._config.initial_state

        logger.info(
            "DialogueFlowManager initialized with initial_state=%r.", self._current_state
        )

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> DialogueFlowConfig:
        states_raw = raw.get("states", {}) or {}
        transitions_raw = raw.get("transitions", []) or []
        initial_state_raw = raw.get("initial_state", "idle")

        states: Dict[str, DialogueStateConfig] = {}
        for name, entry in states_raw.items():
            if not isinstance(entry, dict):
                logger.warning(
                    "Invalid state entry for '%s'. Expected dict, got %s. Skipping.",
                    name,
                    type(entry).__name__,
                )
                continue

            desc = entry.get("description")
            is_terminal = bool(entry.get("is_terminal", False))

            validate_state_name(name)
            states[name] = DialogueStateConfig(
                name=name, description=desc, is_terminal=is_terminal
            )

        transitions: List[DialogueTransitionConfig] = []
        for t in transitions_raw:
            if not isinstance(t, dict):
                logger.warning(
                    "Invalid transition entry (not dict): %r. Skipping.", t
                )
                continue

            from_state = t.get("from_state")
            event = t.get("event")
            to_state = t.get("to_state")
            desc = t.get("description")

            if not from_state or not event or not to_state:
                logger.warning(
                    "Transition missing from_state/event/to_state: %r. Skipping.", t
                )
                continue

            validate_state_name(from_state)
            validate_state_name(to_state)
            validate_event_name(event)

            transitions.append(
                DialogueTransitionConfig(
                    from_state=from_state,
                    event=event,
                    to_state=to_state,
                    description=desc,
                )
            )

        if initial_state_raw not in states:
            logger.warning(
                "Initial state '%s' not in states. Falling back to 'idle'.",
                initial_state_raw,
            )
            initial_state_raw = "idle"
            if "idle" not in states:
                states["idle"] = DialogueStateConfig(
                    name="idle",
                    description="Default idle state",
                    is_terminal=False,
                )

        return DialogueFlowConfig(
            states=states,
            transitions=transitions,
            initial_state=initial_state_raw,
        )

    # ------------------------------ Core Logic -------------------------------

    def get_current_state(self) -> str:
        return self._current_state

    def is_terminal_state(self) -> bool:
        state_cfg = self._config.states.get(self._current_state)
        return bool(state_cfg and state_cfg.is_terminal)

    def _find_transition(self, from_state: str, event: str) -> Optional[DialogueTransitionConfig]:
        for t in self._config.transitions:
            if t.from_state == from_state and t.event == event:
                return t
        return None

    def apply_event(self, event: str) -> str:
        validate_event_name(event)

        if self.is_terminal_state():
            logger.info(
                "Current state '%s' is terminal. Ignoring event '%s'.",
                self._current_state,
                event,
            )
            return self._current_state

        transition = self._find_transition(self._current_state, event)
        if not transition:
            logger.info(
                "No transition found from state '%s' on event '%s'. Staying put.",
                self._current_state,
                event,
            )
            return self._current_state

        if transition.to_state not in self._config.states:
            raise InvalidStateError(
                f"Configured transition leads to unknown state '{transition.to_state}'."
            )

        prev_state = self._current_state
        self._current_state = transition.to_state

        logger.info(
            "Dialogue state transition: '%s' --(%s)--> '%s'",
            prev_state,
            event,
            self._current_state,
        )
        return self._current_state

    # ------------------------------ Introspection -----------------------------

    def list_states(self) -> List[str]:
        return sorted(self._config.states.keys())

    def list_transitions(self) -> List[Dict[str, str]]:
        result = []
        for t in self._config.transitions:
            result.append(
                {
                    "from_state": t.from_state,
                    "event": t.event,
                    "to_state": t.to_state,
                }
            )
        return result

    def debug_state(self) -> Dict[str, Any]:
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "current_state": self._current_state,
            "states": self.list_states(),
            "transitions": self.list_transitions(),
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for DialogueFlowManager...")

    raw_config = {
        "initial_state": "idle",
        "states": {
            "idle": {"description": "Waiting for user input", "is_terminal": False},
            "collecting_info": {
                "description": "Asking follow-up questions",
                "is_terminal": False,
            },
            "processing": {
                "description": "Processing user request",
                "is_terminal": False,
            },
            "completed": {"description": "Conversation done", "is_terminal": True},
        },
        "transitions": [
            {
                "from_state": "idle",
                "event": "user_message",
                "to_state": "collecting_info",
            },
            {
                "from_state": "collecting_info",
                "event": "got_enough_info",
                "to_state": "processing",
            },
            {
                "from_state": "processing",
                "event": "done",
                "to_state": "completed",
            },
        ],
    }

    manager = DialogueFlowManager(config_path=None)
    manager._config = manager._build_config_from_raw(raw_config)

    print("Initial state:", manager.get_current_state())
    print("Apply event 'user_message' ->", manager.apply_event("user_message"))
    print("Apply event 'got_enough_info' ->", manager.apply_event("got_enough_info"))
    print("Apply event 'done' ->", manager.apply_event("done"))
    print("Is terminal state:", manager.is_terminal_state())

    debug_info = manager.debug_state()
    logger.info("DialogueFlowManager debug state: %s", debug_info)

    logger.info("Phase-1 manual test for DialogueFlowManager completed.")
