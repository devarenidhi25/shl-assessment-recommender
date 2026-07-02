import re
from typing import List

from app.schemas import Message

CONFIRMATION_PATTERNS = [
    r"\bconfirmed?\b",
    r"\bthat('s| is) good\b",
    r"\bthat works\b",
    r"\bperfect\b",
    r"\blocking (it|this) in\b",
    r"\bsounds good\b",
    r"\blooks good\b",
    r"\bgo(ing)? with (that|this|it)\b",
    r"\bfinal(ize|ise)?\b",
    r"\bagreed?\b",
    r"\byes,? that('s| is) (it|correct|right)\b",
    r"^\s*yes\.?\s*$",
    r"^\s*ok(ay)?\.?\s*$",
    r"\bwe('re| are) done\b",
    r"\bthat covers it\b",
]
_CONFIRMATION_RE = re.compile("|".join(CONFIRMATION_PATTERNS), re.IGNORECASE)

INJECTION_PATTERNS = [
    r"ignore (all|any|previous|the above|prior) instructions",
    r"disregard (all|any|previous|the above|prior) instructions",
    r"new instructions",
    r"override (your|the) (rules|instructions|guidelines)",
    r"you are now",
    r"system prompt",
    r"reveal your (prompt|instructions|system message)",
    r"print (your|the) (system|instructions|prompt)",
    r"repeat (everything|the words) (above|before)",
    r"act as (an?|the) (?!recruiter|hr)",
    r"pretend (you|to) (are|be)",
    r"forget (everything|all) (you|above)",
    r"jailbreak",
    r"developer mode",
    r"do anything now",
    r"\bDAN\b",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

LEGAL_PATTERNS = [
    r"legally required",
    r"legal (obligation|requirement|advice|liability)",
    r"is (it|this) legal",
    r"comply with (the )?law",
    r"lawsuit",
    r"sue\b",
    r"discriminat(e|ion) (claim|lawsuit)",
    r"eeoc",
    r"gdpr complian",
    r"satisfy (a |the )?(legal|regulatory) requirement",
]
_LEGAL_RE = re.compile("|".join(LEGAL_PATTERNS), re.IGNORECASE)

GENERAL_HIRING_ADVICE_PATTERNS = [
    r"how (much|do i) (should i )?pay",
    r"write (a|the) job (description|posting|ad)\b",
    r"salary (range|benchmark)",
    r"interview questions? for (?!.*assessment)",
    r"how to fire",
    r"how to negotiate (an? )?offer",
    r"onboarding plan",
    r"performance improvement plan",
]
_GENERAL_ADVICE_RE = re.compile("|".join(GENERAL_HIRING_ADVICE_PATTERNS), re.IGNORECASE)


def is_confirmation(text: str) -> bool:
    return bool(_CONFIRMATION_RE.search(text or ""))


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def looks_like_legal_question(text: str) -> bool:
    return bool(_LEGAL_RE.search(text or ""))


def looks_like_general_hiring_advice(text: str) -> bool:
    return bool(_GENERAL_ADVICE_RE.search(text or ""))


def last_user_message(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def format_history_for_prompt(messages: List[Message], max_chars: int = 6000) -> str:
    lines = []
    for m in messages:
        role = "User" if m.role == "user" else "Agent"
        lines.append(f"{role}: {m.content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def count_conversation_turns(messages: List[Message]) -> int:
    return len(messages)


def no_preference_expressed(text: str) -> bool:
    patterns = [
        r"no preference",
        r"don'?t know",
        r"not sure",
        r"you decide",
        r"up to you",
        r"whatever (works|you think)",
        r"doesn'?t matter",
    ]
    return bool(re.search("|".join(patterns), text or "", re.IGNORECASE))
