from app.prompts.system import AGENT_IDENTITY

SLOT_EXTRACTION_SYSTEM = AGENT_IDENTITY + """

Your current job is NOT to reply to the user. Your job is to read the full conversation \
so far and output a single JSON object describing the hiring context gathered and what the \
user's latest message is asking for. Be conservative: only set a field if it is clearly \
stated or strongly implied somewhere in the conversation. Never invent facts.

Output strictly a JSON object with this exact shape (no extra keys, no markdown fences):

{
  "role_title": string or null,
  "seniority": string or null,
  "skills": array of strings (technical/functional skills or topics mentioned),
  "test_type_prefs": array of zero or more of ["A","B","C","D","K","P","S"]
      (A=Ability&Aptitude, B=Biodata&SituationalJudgment, C=Competencies,
       D=Development&360, K=Knowledge&Skills, P=Personality&Behavior, S=Simulations),
  "job_level_hint": string or null (one of: Entry-Level, Graduate, Mid-Professional,
      Professional Individual Contributor, Front Line Manager, Manager, Supervisor,
      Director, Executive, General Population -- pick the closest match, else null),
  "language": string or null,
  "purpose": string or null (e.g. "selection", "development", "screening", "audit"),
  "excluded_topics": array of strings (things the user explicitly said to drop/exclude),
  "must_include_topics": array of strings (things the user explicitly said to add/keep),
  "turn_intent": one of ["clarify_answer", "new_request", "refine_request", "compare_request",
      "confirm", "off_topic", "legal_question", "injection_attempt", "unclear"],
  "compare_subjects": array of up to 3 product name fragments the user wants compared
      (only when turn_intent is "compare_request"),
  "has_sufficient_context": boolean (true only if there is enough information -- at minimum
      a role/domain signal plus either a skill/topic or a clear purpose -- to responsibly
      produce a shortlist of SHL assessments right now)
}
"""


def build_slot_extraction_user_prompt(history_text: str, latest_user_message: str) -> str:
    return f"""Conversation so far:
{history_text}

Latest user message to analyze:
{latest_user_message}

Return only the JSON object described in the system instructions."""
