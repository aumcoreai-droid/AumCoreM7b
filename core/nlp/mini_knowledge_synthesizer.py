"""
================================================================================
AumCore_AI - Mini Knowledge Synthesizer
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: mini_knowledge_synthesizer.py

Description:
    Phase-1 enterprise-grade implementation of a deterministic, rule-based
    "mini knowledge synthesizer". This module takes a small set of
    user-facing facts, notes, or snippets and produces a concise, structured
    synthesis using handcrafted heuristics.

    Phase-1 constraints:
        - No external knowledge bases, no web access
        - No embeddings, no retrieval-augmented generation
        - Pure rule-based grouping + aggregation
        - Chunk-1 + Chunk-2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase-1:
        - Accept a small list of text "fragments" (facts, notes, bullets)
        - Normalize and clean these fragments
        - Group similar fragments using basic keyword overlap
        - Produce a structured synthesis with sections:
            * key_points
            * supporting_details
            * open_questions
        - Provide deterministic behavior and debug information

    Future phases may add:
        - Real semantic clustering using embeddings
        - Integration with external knowledge graphs
        - Confidence scoring and provenance tracking

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
from typing import Optional, Dict, Any, List, Tuple

import yaml

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase-1 (Chunk-1 + Chunk-2)"
__module_name__ = "mini_knowledge_synthesizer"
__description__ = (
    "Phase-1 rule-based Mini Knowledge Synthesizer with Chunk-1 and Chunk-2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("MiniKnowledgeSynthesizer")

logger.info("Module %s initialized (Phase-1, Chunk-1 + Chunk-2).", __module_name__)

# ==============================================================================
# Chunk-2: Error Hierarchy
# ==============================================================================


class KnowledgeSynthError(Exception):
    """
    Base class for all mini knowledge synthesizer errors.
    """

    def __init__(self, message: str, *, code: str = "KNOWLEDGE_SYNTH_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


class InvalidFragmentError(KnowledgeSynthError):
    """
    Raised when a provided fragment is missing or invalid.
    """

    def __init__(self, message: str = "Each fragment must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_FRAGMENT")


class InvalidConfigError(KnowledgeSynthError):
    """
    Raised when YAML configuration is invalid or incomplete.
    """

    def __init__(self, message: str = "Invalid mini knowledge synthesizer configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk-2: Config Structures
# ==============================================================================


@dataclass
class GroupingConfig:
    """
    Configuration for grouping and similarity heuristics.

    Attributes
    ----------
    min_overlap_tokens : int
        Minimum number of overlapping tokens required to group fragments.
    max_groups : int
        Maximum number of groups (topics) to form.
    max_fragments_per_group : int
        Maximum number of fragments allowed per group.
    """

    min_overlap_tokens: int = 2
    max_groups: int = 5
    max_fragments_per_group: int = 10


@dataclass
class SynthesisConfig:
    """
    Configuration for the synthesis layout.

    Attributes
    ----------
    max_key_points : int
        Maximum number of key points to highlight.
    max_supporting_details : int
        Maximum number of supporting detail lines.
    detect_questions : bool
        Whether to detect fragments that look like questions.
    """

    max_key_points: int = 5
    max_supporting_details: int = 10
    detect_questions: bool = True


@dataclass
class MiniKnowledgeSynthesizerConfig:
    """
    Overall configuration for the mini knowledge synthesizer.

    Attributes
    ----------
    grouping : GroupingConfig
        Similarity and grouping configuration.
    synthesis : SynthesisConfig
        Output synthesis configuration.
    """

    grouping: GroupingConfig
    synthesis: SynthesisConfig


# ==============================================================================
# Chunk-2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration for mini knowledge synthesizer, if present.

    Expected structure (Phase-1, optional):

        grouping:
          min_overlap_tokens: 2
          max_groups: 5
          max_fragments_per_group: 10
        synthesis:
          max_key_points: 5
          max_supporting_details: 10
          detect_questions: true

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
        logger.info(
            "No config path provided for MiniKnowledgeSynthesizer. Using defaults."
        )
        return {}

    if not os.path.exists(path):
        logger.warning(
            "Mini knowledge synthesizer config file not found at '%s'. Using defaults.",
            path,
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded mini knowledge synthesizer config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error(
            "Failed to load mini knowledge synthesizer YAML config '%s': %s",
            path,
            exc,
        )
        return {}


# ==============================================================================
# Chunk-2: Sanitization Helpers
# ==============================================================================


def sanitize_fragment(fragment: str) -> str:
    """
    Sanitize and normalize a single fragment.

    - Ensures string type
    - Strips leading/trailing whitespace
    - Collapses internal whitespace
    """
    if not isinstance(fragment, str):
        raise InvalidFragmentError("Fragment must be a string.")
    cleaned = fragment.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def sanitize_fragments(fragments: List[str]) -> List[str]:
    """
    Sanitize a list of fragments, dropping empty ones.
    """
    cleaned_list: List[str] = []
    for frag in fragments:
        cleaned = sanitize_fragment(frag)
        if cleaned:
            cleaned_list.append(cleaned)
    return cleaned_list


# ==============================================================================
# Chunk-2: Validation Helpers
# ==============================================================================


def validate_fragments(fragments: List[str]) -> None:
    """
    Validate that fragments is a non-empty list of non-empty strings.
    """
    if not isinstance(fragments, list):
        raise InvalidFragmentError("Fragments must be provided as a list of strings.")

    if not fragments:
        raise InvalidFragmentError("Fragments list must not be empty.")

    for frag in fragments:
        if not isinstance(frag, str):
            raise InvalidFragmentError("Each fragment must be a string.")
        if not frag.strip():
            raise InvalidFragmentError(
                "Each fragment must be a meaningful non-empty string."
            )


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
# Chunk-1 + Chunk-2: Mini Knowledge Synthesizer (Phase-1 Only)
# ==============================================================================


class MiniKnowledgeSynthesizer:
    """
    Phase-1 rule-based mini knowledge synthesizer.

    Core behavior:
        1. Accept a small list of text fragments (facts, notes, bullets).
        2. Sanitize and validate all fragments.
        3. Tokenize each fragment into lowercase word-like tokens.
        4. Group fragments based on token overlap heuristics.
        5. Within groups, identify key points and supporting details.
        6. Detect open questions (optional).
        7. Return a structured synthesis object (dict).

    All logic is deterministic and uses only simple heuristics.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MiniKnowledgeSynthesizer.

        Parameters
        ----------
        config_path : Optional[str]
            Optional path to YAML configuration file.
        """
        logger.info("Initializing MiniKnowledgeSynthesizer (Phase-1)...")

        raw_config = load_yaml_config(config_path)
        self._config: MiniKnowledgeSynthesizerConfig = self._build_config_from_raw(
            raw_config
        )

        logger.info(
            "MiniKnowledgeSynthesizer configured: min_overlap_tokens=%d, max_groups=%d, "
            "max_fragments_per_group=%d, max_key_points=%d, max_supporting_details=%d, "
            "detect_questions=%s",
            self._config.grouping.min_overlap_tokens,
            self._config.grouping.max_groups,
            self._config.grouping.max_fragments_per_group,
            self._config.synthesis.max_key_points,
            self._config.synthesis.max_supporting_details,
            self._config.synthesis.detect_questions,
        )

        logger.info("MiniKnowledgeSynthesizer initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> MiniKnowledgeSynthesizerConfig:
        grouping_raw = raw.get("grouping", {}) or {}
        synthesis_raw = raw.get("synthesis", {}) or {}

        min_overlap = _validate_positive_int(
            grouping_raw.get("min_overlap_tokens", 2),
            default=2,
            name="min_overlap_tokens",
        )
        max_groups = _validate_positive_int(
            grouping_raw.get("max_groups", 5),
            default=5,
            name="max_groups",
        )
        max_frag_per_group = _validate_positive_int(
            grouping_raw.get("max_fragments_per_group", 10),
            default=10,
            name="max_fragments_per_group",
        )

        grouping = GroupingConfig(
            min_overlap_tokens=min_overlap,
            max_groups=max_groups,
            max_fragments_per_group=max_frag_per_group,
        )

        max_key_points = _validate_positive_int(
            synthesis_raw.get("max_key_points", 5),
            default=5,
            name="max_key_points",
        )
        max_supporting_details = _validate_positive_int(
            synthesis_raw.get("max_supporting_details", 10),
            default=10,
            name="max_supporting_details",
        )
        detect_questions = _validate_bool(
            synthesis_raw.get("detect_questions", True),
            default=True,
            name="detect_questions",
        )

        synthesis = SynthesisConfig(
            max_key_points=max_key_points,
            max_supporting_details=max_supporting_details,
            detect_questions=detect_questions,
        )

        return MiniKnowledgeSynthesizerConfig(grouping=grouping, synthesis=synthesis)

    # ------------------------------ Tokenization -----------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text into lowercase word-like tokens.
        """
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return tokens

    # ----------------------------- Grouping Logic ----------------------------

    def _compute_overlap(self, tokens_a: List[str], tokens_b: List[str]) -> int:
        """
        Compute token overlap count between two token lists.
        """
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        overlap = len(set_a.intersection(set_b))
        return overlap

    def _group_fragments(self, fragments: List[str]) -> List[List[str]]:
        """
        Group fragments based on token overlap heuristics.

        Algorithm (Phase-1, greedy):
            - Start with first fragment as seed of first group.
            - For each next fragment:
                * Compute overlap with each existing group representative.
                * If overlap >= min_overlap_tokens, join that group.
                * Otherwise, create a new group (until max_groups reached).
        """
        min_overlap = self._config.grouping.min_overlap_tokens
        max_groups = self._config.grouping.max_groups
        max_frag_per_group = self._config.grouping.max_fragments_per_group

        if not fragments:
            return []

        tokens_per_fragment = [self._tokenize(f) for f in fragments]

        groups: List[List[int]] = []
        group_representatives: List[int] = []

        groups.append([0])
        group_representatives.append(0)

        for idx in range(1, len(fragments)):
            tokens = tokens_per_fragment[idx]
            best_group_idx = None
            best_overlap = 0

            for g_idx, rep_idx in enumerate(group_representatives):
                rep_tokens = tokens_per_fragment[rep_idx]
                overlap = self._compute_overlap(tokens, rep_tokens)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_group_idx = g_idx

            if best_group_idx is not None and best_overlap >= min_overlap:
                if len(groups[best_group_idx]) < max_frag_per_group:
                    groups[best_group_idx].append(idx)
                else:
                    logger.info(
                        "Group %d reached max_fragments_per_group=%d. Fragment %d not added.",
                        best_group_idx,
                        max_frag_per_group,
                        idx,
                    )
            else:
                if len(groups) < max_groups:
                    groups.append([idx])
                    group_representatives.append(idx)
                else:
                    logger.info(
                        "Max_groups=%d reached. Fragment %d will not form a new group.",
                        max_groups,
                        idx,
                    )

        grouped_fragments: List[List[str]] = []
        for g in groups:
            grouped_fragments.append([fragments[i] for i in g])

        return grouped_fragments

    # ------------------------ Key Point Identification -----------------------

    @staticmethod
    def _is_question(fragment: str) -> bool:
        """
        Heuristic to detect if a fragment is a question.
        """
        if "?" in fragment:
            return True

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
        lower = fragment.lower().strip()
        for prefix in question_starts:
            if lower.startswith(prefix):
                return True
        return False

    def _select_key_points(
        self, groups: List[List[str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Select key points and supporting details from grouped fragments.

        Strategy (Phase-1):
            - For each group, the first fragment is treated as the key point.
            - Remaining fragments in the group are supporting details.
            - Flatten across groups up to configured limits.
        """
        max_key_points = self._config.synthesis.max_key_points
        max_supporting = self._config.synthesis.max_supporting_details

        key_points: List[str] = []
        supporting_details: List[str] = []

        for group in groups:
            if not group:
                continue

            if len(key_points) < max_key_points:
                key_points.append(group[0])

            for frag in group[1:]:
                if len(supporting_details) < max_supporting:
                    supporting_details.append(frag)

        return key_points, supporting_details

    def _collect_open_questions(self, fragments: List[str]) -> List[str]:
        """
        Collect fragments that look like open questions.
        """
        if not self._config.synthesis.detect_questions:
            return []

        questions: List[str] = []
        for frag in fragments:
            if self._is_question(frag):
                questions.append(frag)
        return questions

    # --------------------------- Public Interface ----------------------------

    def synthesize(self, fragments: List[str]) -> Dict[str, Any]:
        """
        Produce a structured synthesis from a list of fragments.

        Parameters
        ----------
        fragments : List[str]
            List of short textual units (facts, notes, bullets).

        Returns
        -------
        Dict[str, Any]
            Synthesis dictionary with keys:
                - key_points: List[str]
                - supporting_details: List[str]
                - open_questions: List[str]
        """
        validate_fragments(fragments)
        cleaned_fragments = sanitize_fragments(fragments)

        if not cleaned_fragments:
            logger.info("All fragments were empty after sanitization.")
            return {
                "key_points": [],
                "supporting_details": [],
                "open_questions": [],
            }

        groups = self._group_fragments(cleaned_fragments)
        key_points, supporting = self._select_key_points(groups)
        questions = self._collect_open_questions(cleaned_fragments)

        logger.info(
            "Synthesis result: key_points=%d, supporting_details=%d, open_questions=%d",
            len(key_points),
            len(supporting),
            len(questions),
        )

        return {
            "key_points": key_points,
            "supporting_details": supporting,
            "open_questions": questions,
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
            "min_overlap_tokens": self._config.grouping.min_overlap_tokens,
            "max_groups": self._config.grouping.max_groups,
            "max_fragments_per_group": self._config.grouping.max_fragments_per_group,
            "max_key_points": self._config.synthesis.max_key_points,
            "max_supporting_details": self._config.synthesis.max_supporting_details,
            "detect_questions": self._config.synthesis.detect_questions,
        }


# ==============================================================================
# Phase-1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase-1 manual test for MiniKnowledgeSynthesizer...")

    synthesizer = MiniKnowledgeSynthesizer(config_path=None)

    sample_fragments = [
        "User is asking about the current status of their order.",
        "They mention the tracking page is not updating.",
        "User wants to know when the delivery will arrive.",
        "There seems to be an issue with the logistics provider.",
        "What is causing the delay?",
        "Support might need to contact the shipping partner.",
        "User is concerned because the package contains important items.",
    ]

    synthesis = synthesizer.synthesize(sample_fragments)

    print("KEY POINTS:")
    for kp in synthesis["key_points"]:
        print(" -", kp)

    print("\nSUPPORTING DETAILS:")
    for sd in synthesis["supporting_details"]:
        print(" -", sd)

    print("\nOPEN QUESTIONS:")
    for q in synthesis["open_questions"]:
        print(" -", q)

    debug_info = synthesizer.debug_state()
    logger.info("MiniKnowledgeSynthesizer debug state: %s", debug_info)

    logger.info("Phase-1 manual test for MiniKnowledgeSynthesizer completed.")
