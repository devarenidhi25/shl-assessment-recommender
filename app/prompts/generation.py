from typing import List

from app.catalog import CatalogItem
from app.prompts.system import AGENT_IDENTITY

RECOMMEND_SYSTEM = AGENT_IDENTITY + """

You are now producing a shortlist turn. You will be given a CANDIDATE POOL of real SHL \
catalog products (name + url + short facts) and the conversation context. You must select \
between 1 and 10 items from the CANDIDATE POOL ONLY -- never invent a product, never invent \
a URL, never modify a name or URL. If the user asked to add or remove specific items \
("refine"), carry over the previously shown items that are still relevant and only change \
what the user asked to change; do not silently drop unrelated items.

Write a short, direct reply (2-4 sentences) explaining the shortlist or the change made. If \
you included a personality measure as a sensible default the user didn't explicitly ask for, \
say so briefly and note it's optional. If the pool has no strong match for something the user \
asked for, say that plainly instead of forcing a weak match.

Output strictly a JSON object, no markdown fences:

{
  "reply": string,
  "selected_urls": array of strings (each MUST be an exact url copied from the candidate pool,
      1 to 10 items, ordered by relevance),
  "end_of_conversation": boolean (true only if the user's latest message clearly confirms /
      finalizes the shortlist; false otherwise)
}
"""


def build_recommend_user_prompt(
    history_text: str,
    latest_user_message: str,
    candidate_pool: List[CatalogItem],
    previous_recommendation_urls: List[str],
) -> str:
    pool_lines = []
    for it in candidate_pool:
        pool_lines.append(
            f"- name: {it.name} | url: {it.url} | test_type: {it.test_type} | "
            f"duration: {it.duration_raw or 'n/a'} | keys: {', '.join(it.keys)} | "
            f"description: {it.description[:220]}"
        )
    pool_text = "\n".join(pool_lines) if pool_lines else "(no candidates matched)"

    prev_text = ", ".join(previous_recommendation_urls) if previous_recommendation_urls else "(none yet)"

    return f"""Conversation so far:
{history_text}

Latest user message:
{latest_user_message}

Previously shown recommendation URLs (carry these over unless the user's latest message asks
to remove or replace them):
{prev_text}

CANDIDATE POOL (select only from these):
{pool_text}

Return only the JSON object described in the system instructions."""


CLARIFY_SYSTEM = AGENT_IDENTITY + """

You are now producing a clarifying turn. The conversation does not yet contain enough \
information to responsibly recommend SHL assessments. Ask ONE focused, natural clarifying \
question that will most efficiently unlock a good recommendation (e.g. seniority level, \
core skills/domain, purpose of assessment, language, or team size). Do not recommend any \
assessments in this turn. Keep it to 1-3 sentences, conversational, no bullet lists.

Output strictly a JSON object, no markdown fences:

{
  "reply": string
}
"""


def build_clarify_user_prompt(history_text: str, latest_user_message: str) -> str:
    return f"""Conversation so far:
{history_text}

Latest user message:
{latest_user_message}

Return only the JSON object described in the system instructions."""


COMPARE_SYSTEM = AGENT_IDENTITY + """

You are now producing a comparison turn. The user wants to understand the difference between \
specific SHL products. You will be given the matched catalog entries with their real \
descriptions, keys, and durations. Ground your answer STRICTLY in the provided facts -- do not \
use outside knowledge about these products. If a product the user mentioned was not found in \
the catalog data provided, say so plainly instead of guessing. Keep the answer to 3-6 \
sentences, clear and specific about what differs.

Output strictly a JSON object, no markdown fences:

{
  "reply": string
}
"""


def build_compare_user_prompt(
    history_text: str,
    latest_user_message: str,
    matched_items: List[CatalogItem],
) -> str:
    if matched_items:
        lines = []
        for it in matched_items:
            lines.append(
                f"- name: {it.name} | url: {it.url} | test_type: {it.test_type} | "
                f"duration: {it.duration_raw or 'n/a'} | keys: {', '.join(it.keys)} | "
                f"description: {it.description}"
            )
        matched_text = "\n".join(lines)
    else:
        matched_text = "(no matching catalog entries were found for the products mentioned)"

    return f"""Conversation so far:
{history_text}

Latest user message:
{latest_user_message}

Matched catalog entries (only source of truth for this comparison):
{matched_text}

Return only the JSON object described in the system instructions."""
