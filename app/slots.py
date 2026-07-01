import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.llm import LLMError, call_llm_json
from app.prompts.slot_extraction import (
    SLOT_EXTRACTION_SYSTEM,
    build_slot_extraction_user_prompt,
)
from app.utils.text import (
    looks_like_general_hiring_advice,
    looks_like_injection,
    looks_like_legal_question,
)

logger = logging.getLogger("shl_recommender.slots")

VALID_TURN_INTENTS = {
    "clarify_answer",
    "new_request",
    "refine_request",
    "compare_request",
    "confirm",
    "off_topic",
    "legal_question",
    "injection_attempt",
    "unclear",
}

VALID_TEST_TYPES = {"A", "B", "C", "D", "K", "P", "S"}


@dataclass
class Slots:
    role_title: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    test_type_prefs: List[str] = field(default_factory=list)
    job_level_hint: Optional[str] = None
    language: Optional[str] = None
    purpose: Optional[str] = None
    excluded_topics: List[str] = field(default_factory=list)
    must_include_topics: List[str] = field(default_factory=list)
    turn_intent: str = "unclear"
    compare_subjects: List[str] = field(default_factory=list)
    has_sufficient_context: bool = False


def _clean_str_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _parse_slots(raw: dict) -> Slots:
    turn_intent = raw.get("turn_intent")
    if turn_intent not in VALID_TURN_INTENTS:
        turn_intent = "unclear"

    test_type_prefs = [
        t.strip().upper()
        for t in raw.get("test_type_prefs", []) or []
        if isinstance(t, str) and t.strip().upper() in VALID_TEST_TYPES
    ]

    def _opt_str(v):
        return v.strip() if isinstance(v, str) and v.strip() else None

    return Slots(
        role_title=_opt_str(raw.get("role_title")),
        seniority=_opt_str(raw.get("seniority")),
        skills=_clean_str_list(raw.get("skills")),
        test_type_prefs=test_type_prefs,
        job_level_hint=_opt_str(raw.get("job_level_hint")),
        language=_opt_str(raw.get("language")),
        purpose=_opt_str(raw.get("purpose")),
        excluded_topics=_clean_str_list(raw.get("excluded_topics")),
        must_include_topics=_clean_str_list(raw.get("must_include_topics")),
        turn_intent=turn_intent,
        compare_subjects=_clean_str_list(raw.get("compare_subjects")),
        has_sufficient_context=bool(raw.get("has_sufficient_context", False)),
    )


def _rule_based_override(latest_user_message: str, slots: Slots) -> Slots:
    """Deterministic safety net on top of the LLM classification -- these
    checks never get relaxed by an LLM misclassification, which matters for
    the refusal/injection-resistance requirement."""
    if looks_like_injection(latest_user_message):
        slots.turn_intent = "injection_attempt"
    elif looks_like_legal_question(latest_user_message):
        slots.turn_intent = "legal_question"
    elif looks_like_general_hiring_advice(latest_user_message):
        slots.turn_intent = "off_topic"
    return slots


def extract_slots(history_text: str, latest_user_message: str) -> Slots:
    try:
        raw = call_llm_json(
            SLOT_EXTRACTION_SYSTEM,
            build_slot_extraction_user_prompt(history_text, latest_user_message),
        )
        slots = _parse_slots(raw)
    except LLMError as e:
        logger.error("Slot extraction failed, falling back to minimal slots: %s", e)
        slots = Slots(turn_intent="unclear", has_sufficient_context=False)

    return _rule_based_override(latest_user_message, slots)
