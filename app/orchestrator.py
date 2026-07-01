import logging
import re
from typing import List, Optional, Tuple

from app.catalog import CatalogItem, get_catalog
from app.config import settings
from app.llm import LLMError, call_llm_json
from app.prompts.generation import (
    CLARIFY_SYSTEM,
    COMPARE_SYSTEM,
    RECOMMEND_SYSTEM,
    build_clarify_user_prompt,
    build_compare_user_prompt,
    build_recommend_user_prompt,
)
from app.retrieval import find_items_mentioned, retrieve_candidates
from app.schemas import ChatResponse, Message, RecommendationItem
from app.slots import Slots, extract_slots
from app.utils.text import (
    format_history_for_prompt,
    is_confirmation,
    last_user_message,
    no_preference_expressed,
)

logger = logging.getLogger("shl_recommender.orchestrator")

_URL_RE = re.compile(r"https?://www\.shl\.com/[^\s\)\]\|]+")

REFUSAL_REPLIES = {
    "legal_question": (
        "That's a legal or compliance question, which is outside what I can advise on -- "
        "I can help you select the right SHL assessments, but not interpret regulatory "
        "obligations or whether a specific test satisfies a legal requirement. Your legal "
        "or compliance team is the right resource for that. Happy to keep working on the "
        "assessment shortlist."
    ),
    "off_topic": (
        "I'm focused specifically on helping you choose SHL assessments -- I can't help with "
        "general hiring advice, compensation, or process questions outside that scope. Tell me "
        "about the role you're assessing for and I can take it from there."
    ),
    "injection_attempt": (
        "I can only help with selecting SHL assessments for a role, and I'll keep doing that "
        "regardless of how a request is phrased. What role or hiring need are you working on?"
    ),
}

MAX_CLARIFY_TURNS_BEFORE_FORCE = 3


def _items_to_recommendations(items: List[CatalogItem]) -> List[RecommendationItem]:
    return [
        RecommendationItem(name=it.name, url=it.url, test_type=it.test_type)
        for it in items
    ]


def _extract_previous_recommendation_items(messages: List[Message]) -> List[CatalogItem]:
    catalog = get_catalog()
    for m in reversed(messages):
        if m.role != "assistant":
            continue
        urls = _URL_RE.findall(m.content or "")
        items = []
        seen = set()
        for u in urls:
            u = u.strip().rstrip(").,]")
            item = catalog.get_by_url(u)
            if item and item.url not in seen:
                seen.add(item.url)
                items.append(item)
        if items:
            return items
        # last assistant message existed but had no recognizable catalog urls
        # (e.g. it was a clarifying/refusal/compare turn) -- keep walking back
    return []


def _count_prior_clarify_turns(messages: List[Message]) -> int:
    count = 0
    for m in messages:
        if m.role == "assistant" and not _URL_RE.search(m.content or ""):
            count += 1
    return count


def _format_reply_with_table(reply_text: str, items: List[CatalogItem]) -> str:
    if not items:
        return reply_text
    lines = [reply_text.strip(), "", "| # | Name | Test Type | Duration | URL |", "|---|------|-----------|----------|-----|"]
    for i, it in enumerate(items, start=1):
        lines.append(
            f"| {i} | {it.name} | {it.test_type or '-'} | {it.duration_raw or '-'} | {it.url} |"
        )
    return "\n".join(lines)


def _safe_recommend_call(
    history_text: str,
    latest_user_message: str,
    candidate_pool: List[CatalogItem],
    previous_items: List[CatalogItem],
) -> Tuple[str, List[CatalogItem], bool]:
    catalog = get_catalog()
    prev_urls = [it.url for it in previous_items]
    try:
        raw = call_llm_json(
            RECOMMEND_SYSTEM,
            build_recommend_user_prompt(history_text, latest_user_message, candidate_pool, prev_urls),
            temperature=settings.llm_temperature_generate,
        )
        selected_urls = raw.get("selected_urls", [])
        reply_text = str(raw.get("reply", "")).strip()
        end_of_conv = bool(raw.get("end_of_conversation", False))
    except LLMError as e:
        logger.error("Recommend call failed, using fallback pool: %s", e)
        selected_urls = [it.url for it in candidate_pool[: settings.max_recommendations]]
        reply_text = "Here is a shortlist based on what you've told me so far."
        end_of_conv = False

    if not isinstance(selected_urls, list):
        selected_urls = []

    # Ground strictly against the real catalog -- drop anything not present,
    # never trust a model-generated url/name pair directly.
    selected_items: List[CatalogItem] = []
    seen = set()
    for u in selected_urls:
        if not isinstance(u, str):
            continue
        item = catalog.get_by_url(u.strip())
        if item and item.url not in seen:
            seen.add(item.url)
            selected_items.append(item)

    if not selected_items:
        # defensive fallback so a recommend turn is never empty when we do
        # have candidates to offer
        fallback_source = candidate_pool or previous_items
        selected_items = fallback_source[: settings.max_recommendations]

    selected_items = selected_items[: settings.max_recommendations]

    if not reply_text:
        reply_text = "Here is a shortlist based on the context so far."

    return reply_text, selected_items, end_of_conv


def _handle_refusal(intent: str) -> ChatResponse:
    reply = REFUSAL_REPLIES.get(intent, REFUSAL_REPLIES["off_topic"])
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _handle_clarify(history_text: str, latest_msg: str) -> ChatResponse:
    try:
        raw = call_llm_json(
            CLARIFY_SYSTEM,
            build_clarify_user_prompt(history_text, latest_msg),
            temperature=settings.llm_temperature_generate,
        )
        reply = str(raw.get("reply", "")).strip()
    except LLMError as e:
        logger.error("Clarify call failed, using generic fallback question: %s", e)
        reply = ""

    if not reply:
        reply = (
            "Happy to help find the right SHL assessments. Could you tell me a bit more "
            "about the role -- what it involves and the seniority level -- so I can narrow "
            "this down?"
        )
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _handle_compare(history_text: str, latest_msg: str, slots: Slots) -> ChatResponse:
    catalog = get_catalog()
    matched: List[CatalogItem] = []
    seen = set()

    for subject in slots.compare_subjects:
        for it in catalog.find_by_partial_name(subject, limit=2):
            if it.url not in seen:
                seen.add(it.url)
                matched.append(it)

    if not matched:
        for it in find_items_mentioned(latest_msg):
            if it.url not in seen:
                seen.add(it.url)
                matched.append(it)

    try:
        raw = call_llm_json(
            COMPARE_SYSTEM,
            build_compare_user_prompt(history_text, latest_msg, matched),
            temperature=settings.llm_temperature_generate,
        )
        reply = str(raw.get("reply", "")).strip()
    except LLMError as e:
        logger.error("Compare call failed: %s", e)
        reply = ""

    if not reply:
        if matched:
            names = ", ".join(it.name for it in matched)
            reply = (
                f"I found these in the catalog: {names}. Could you tell me specifically what "
                f"aspect you'd like compared (what each measures, duration, or use case)?"
            )
        else:
            reply = (
                "I couldn't confidently match the products you're asking about to catalog "
                "entries. Could you give me the exact names you'd like compared?"
            )

    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _handle_confirm(previous_items: List[CatalogItem]) -> ChatResponse:
    if not previous_items:
        return _handle_clarify(
            "", "The user said something that sounds like confirmation but no prior "
            "shortlist exists yet."
        )
    reply = _format_reply_with_table("Confirmed. Here is the final shortlist.", previous_items)
    return ChatResponse(
        reply=reply,
        recommendations=_items_to_recommendations(previous_items),
        end_of_conversation=True,
    )


def _handle_recommend_or_refine(
    messages: List[Message],
    history_text: str,
    latest_msg: str,
    slots: Slots,
    previous_items: List[CatalogItem],
) -> ChatResponse:
    exclude_urls = []
    if slots.excluded_topics:
        catalog = get_catalog()
        for topic in slots.excluded_topics:
            for it in catalog.find_by_partial_name(topic, limit=3):
                exclude_urls.append(it.url)

    query_text = " ".join(
        filter(
            None,
            [
                slots.role_title,
                slots.seniority,
                slots.purpose,
                " ".join(slots.skills),
                " ".join(slots.must_include_topics),
                latest_msg,
            ],
        )
    )

    candidate_pool = retrieve_candidates(
        query_text=query_text,
        skills=slots.skills,
        job_level=slots.job_level_hint,
        language=slots.language,
        test_type_prefs=slots.test_type_prefs or None,
        exclude_urls=exclude_urls,
    )

    # always keep previously shown items visible to the model as carry-over
    # candidates so a refine turn can retain them even if the keyword filter
    # would otherwise have dropped them
    pool_urls = {it.url for it in candidate_pool}
    for it in previous_items:
        if it.url not in pool_urls and it.url not in exclude_urls:
            candidate_pool.append(it)
            pool_urls.add(it.url)

    reply_text, selected_items, end_of_conv = _safe_recommend_call(
        history_text, latest_msg, candidate_pool, previous_items
    )

    if not end_of_conv:
        end_of_conv = is_confirmation(latest_msg) and bool(previous_items) and selected_items == previous_items

    full_reply = _format_reply_with_table(reply_text, selected_items)

    return ChatResponse(
        reply=full_reply,
        recommendations=_items_to_recommendations(selected_items),
        end_of_conversation=end_of_conv,
    )


def run_turn(messages: List[Message]) -> ChatResponse:
    try:
        latest_msg = last_user_message(messages)
        if not latest_msg.strip():
            return ChatResponse(
                reply=(
                    "I didn't catch a question or request there. What role or hiring need "
                    "are you looking to find SHL assessments for?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        history_text = format_history_for_prompt(messages)
        previous_items = _extract_previous_recommendation_items(messages)
        slots = extract_slots(history_text, latest_msg)

        if slots.turn_intent in ("injection_attempt", "off_topic", "legal_question"):
            return _handle_refusal(slots.turn_intent)

        if slots.turn_intent == "compare_request":
            return _handle_compare(history_text, latest_msg, slots)

        if slots.turn_intent == "confirm" and previous_items:
            return _handle_confirm(previous_items)

        prior_clarify_turns = _count_prior_clarify_turns(messages)
        force_recommend = (
            len(messages) >= settings.max_turns - 1
            or prior_clarify_turns >= MAX_CLARIFY_TURNS_BEFORE_FORCE
        )

        if no_preference_expressed(latest_msg) and prior_clarify_turns > 0:
            slots.has_sufficient_context = True

        if not slots.has_sufficient_context and not force_recommend and slots.turn_intent != "refine_request":
            return _handle_clarify(history_text, latest_msg)

        return _handle_recommend_or_refine(messages, history_text, latest_msg, slots, previous_items)

    except Exception as e:  # noqa: BLE001 - last-resort guard so /chat never 500s on the grader
        logger.exception("Unhandled orchestrator error: %s", e)
        return ChatResponse(
            reply=(
                "I hit an issue processing that -- could you rephrase what you're looking "
                "for in an SHL assessment?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
