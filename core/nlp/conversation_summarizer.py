"""
================================================================================
AumCore_AI - Conversation Summarizer
Phase-1: Chunk-1 + Chunk-2 Only (Rule-based, No Model Load)
================================================================================
Module: conversation_summarizer.py

Description:
    Phase‑1 enterprise-grade implementation of a deterministic, rule‑based
    conversation summarizer. This module produces short, medium, or long
    summaries of a conversation transcript using ONLY handcrafted heuristics.

    Phase‑1 constraints:
        - No ML models, no embeddings, no transformers
        - No semantic similarity, no vector search
        - Pure rule-based extraction + compression
        - Chunk‑1 + Chunk‑2 only
        - 400+ line foundation for future expansion

    Core responsibilities in Phase‑1:
        - Maintain configurable summary lengths (short/medium/long)
        - Extract key sentences using simple heuristics
        - Compress redundant text
        - Provide deterministic summaries
        - Offer debug state for higher layers

    This file is intentionally verbose and structured to support future
    phases (Chunk‑3 to Chunk‑8), where advanced summarization models,
    embeddings, and contextual memory will be added.

================================================================================
"""

# ==============================================================================
# Chunk‑1: Imports, Metadata, Logging Setup
# ==============================================================================

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import yaml

__author__ = "AumCore_AI"
__version__ = "1.0.0"
__phase__ = "Phase‑1 (Chunk‑1 + Chunk‑2)"
__module_name__ = "conversation_summarizer"
__description__ = (
    "Phase‑1 rule‑based Conversation Summarizer with Chunk‑1 and Chunk‑2 only."
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ConversationSummarizer")

logger.info("Module %s initialized (Phase‑1, Chunk‑1 + Chunk‑2).", __module_name__)

# ==============================================================================
# Chunk‑2: Error Hierarchy
# ==============================================================================


class SummarizerError(Exception):
    def __init__(self, message: str, *, code: str = "SUMMARIZER_ERROR"):
        super().__init__(message)
        self.code = code

    def __str__(self):
        return f"[{self.code}] {super().__str__()}"


class InvalidTranscriptError(SummarizerError):
    def __init__(self, message="Transcript must be a meaningful non-empty string."):
        super().__init__(message, code="INVALID_TRANSCRIPT")


class InvalidSummaryModeError(SummarizerError):
    def __init__(self, message="Invalid summary mode. Must be short/medium/long."):
        super().__init__(message, code="INVALID_SUMMARY_MODE")


class InvalidConfigError(SummarizerError):
    def __init__(self, message="Invalid summarizer configuration."):
        super().__init__(message, code="INVALID_CONFIG")


# ==============================================================================
# Chunk‑2: Config Structures
# ==============================================================================


@dataclass
class SummaryLengthConfig:
    short_max_sentences: int = 2
    medium_max_sentences: int = 5
    long_max_sentences: int = 10


@dataclass
class ConversationSummarizerConfig:
    lengths: SummaryLengthConfig
    remove_filler_words: bool = True
    filler_words: List[str] = None


# ==============================================================================
# Chunk‑2: YAML Config Loader
# ==============================================================================


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        logger.info("No config path provided for summarizer. Using defaults.")
        return {}

    if not os.path.exists(path):
        logger.warning("Summarizer config file not found at '%s'. Using defaults.", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            logger.info("Loaded summarizer config from '%s'.", path)
            return data
    except Exception as exc:
        logger.error("Failed to load summarizer YAML config '%s': %s", path, exc)
        return {}


# ==============================================================================
# Chunk‑2: Sanitization Helpers
# ==============================================================================


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        raise InvalidTranscriptError("Transcript must be a string.")
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def sanitize_sentences(sentences: List[str]) -> List[str]:
    cleaned = []
    for s in sentences:
        s2 = s.strip()
        if s2:
            cleaned.append(s2)
    return cleaned


# ==============================================================================
# Chunk‑2: Validation Helpers
# ==============================================================================


def validate_transcript(text: str):
    if not isinstance(text, str):
        raise InvalidTranscriptError("Transcript must be a string.")
    if not text.strip():
        raise InvalidTranscriptError("Transcript must not be empty.")


def validate_summary_mode(mode: str):
    if mode not in ("short", "medium", "long"):
        raise InvalidSummaryModeError()


# ==============================================================================
# Chunk‑1 + Chunk‑2: Conversation Summarizer (Phase‑1 Only)
# ==============================================================================


class ConversationSummarizer:
    """
    Phase‑1 rule‑based conversation summarizer.

    Algorithm (Phase‑1):
        1. Split transcript into sentences.
        2. Remove filler words if enabled.
        3. Score sentences using simple heuristics:
            - length
            - presence of keywords
            - position in transcript
        4. Select top N sentences based on summary mode.
        5. Return deterministic summary.

    This is intentionally simple and deterministic for Phase‑1.
    """

    # ---------------------------- Initialization -----------------------------

    def __init__(self, config_path: Optional[str] = None):
        logger.info("Initializing ConversationSummarizer (Phase‑1)...")

        raw_config = load_yaml_config(config_path)
        self._config: ConversationSummarizerConfig = self._build_config_from_raw(
            raw_config
        )

        logger.info(
            "ConversationSummarizer configured: short=%d, medium=%d, long=%d",
            self._config.lengths.short_max_sentences,
            self._config.lengths.medium_max_sentences,
            self._config.lengths.long_max_sentences,
        )

        logger.info("ConversationSummarizer initialized successfully.")

    # ----------------------- Config Construction Helper ----------------------

    @staticmethod
    def _build_config_from_raw(raw: Dict[str, Any]) -> ConversationSummarizerConfig:
        lengths_raw = raw.get("lengths", {}) or {}

        short_len = lengths_raw.get("short_max_sentences", 2)
        med_len = lengths_raw.get("medium_max_sentences", 5)
        long_len = lengths_raw.get("long_max_sentences", 10)

        lengths = SummaryLengthConfig(
            short_max_sentences=short_len,
            medium_max_sentences=med_len,
            long_max_sentences=long_len,
        )

        remove_filler = raw.get("remove_filler_words", True)
        filler_words = raw.get(
            "filler_words",
            ["um", "uh", "like", "you know", "basically", "actually", "literally"],
        )

        return ConversationSummarizerConfig(
            lengths=lengths,
            remove_filler_words=remove_filler,
            filler_words=filler_words,
        )

    # ------------------------------ Sentence Split ----------------------------

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        parts = re.split(r"[.!?]+", text)
        return sanitize_sentences(parts)

    # ------------------------------ Filler Removal ----------------------------

    def _remove_filler_words(self, sentence: str) -> str:
        if not self._config.remove_filler_words:
            return sentence

        cleaned = sentence
        for fw in self._config.filler_words:
            pattern = r"\b" + re.escape(fw) + r"\b"
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # ------------------------------ Scoring Logic -----------------------------

    def _score_sentence(self, sentence: str, index: int, total: int) -> float:
        score = 0.0

        length_score = min(len(sentence) / 100.0, 1.0)
        score += length_score

        if index == 0:
            score += 0.5
        if index == total - 1:
            score += 0.3

        keywords = ["issue", "problem", "request", "question", "solution", "plan"]
        for kw in keywords:
            if kw in sentence.lower():
                score += 0.2

        return score

    # ------------------------------ Summary Logic -----------------------------

    def summarize(self, transcript: str, mode: str = "medium") -> str:
        validate_transcript(transcript)
        validate_summary_mode(mode)

        cleaned = sanitize_text(transcript)
        sentences = self._split_into_sentences(cleaned)

        if not sentences:
            return ""

        processed = [self._remove_filler_words(s) for s in sentences]

        scored = []
        total = len(processed)
        for idx, s in enumerate(processed):
            score = self._score_sentence(s, idx, total)
            scored.append((s, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if mode == "short":
            limit = self._config.lengths.short_max_sentences
        elif mode == "medium":
            limit = self._config.lengths.medium_max_sentences
        else:
            limit = self._config.lengths.long_max_sentences

        selected = [s for s, _ in scored[:limit]]
        summary = ". ".join(selected).strip()
        if summary and not summary.endswith("."):
            summary += "."

        logger.info("Generated %s summary with %d sentences.", mode, len(selected))
        return summary

    # ------------------------------ Introspection -----------------------------

    def debug_state(self) -> Dict[str, Any]:
        return {
            "module": __module_name__,
            "version": __version__,
            "phase": __phase__,
            "short_max": self._config.lengths.short_max_sentences,
            "medium_max": self._config.lengths.medium_max_sentences,
            "long_max": self._config.lengths.long_max_sentences,
            "remove_filler_words": self._config.remove_filler_words,
        }


# ==============================================================================
# Phase‑1 Test Block (Manual Verification Only)
# ==============================================================================

if __name__ == "__main__":
    logger.info("Running Phase‑1 manual test for ConversationSummarizer...")

    summarizer = ConversationSummarizer(config_path=None)

    transcript = (
        "Hello, I have an issue with my order. "
        "I tried contacting support but did not get a reply. "
        "Basically I want to know the status. "
        "Also, I have another question about delivery time. "
        "Thanks for your help."
    )

    print("SHORT SUMMARY:\n", summarizer.summarize(transcript, "short"))
    print("\nMEDIUM SUMMARY:\n", summarizer.summarize(transcript, "medium"))
    print("\nLONG SUMMARY:\n", summarizer.summarize(transcript, "long"))

    debug_info = summarizer.debug_state()
    logger.info("ConversationSummarizer debug state: %s", debug_info)

    logger.info("Phase‑1 manual test completed.")
