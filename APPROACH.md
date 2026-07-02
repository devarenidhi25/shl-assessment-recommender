# Approach Document — Conversational SHL Assessment Recommender

## 1. Problem framing and design goals

The task is a stateless, multi-turn agent that moves a recruiter from a vague intent
to a grounded shortlist of SHL **Individual Test Solutions**, while handling four
behaviors (clarify, recommend, refine, compare) and staying strictly in scope. Scoring
is: hard evals (schema compliance, catalog-only items, ≤8 turns), **Recall@10** on the
final shortlist, and **behavior-probe** pass-rate.

I optimized for three things in priority order: (1) never break the hard evals — a
malformed response or an out-of-catalog URL zeroes a trace; (2) maximize Recall@10 by
retrieving broadly and letting the shortlist lean inclusive; (3) pass behavior probes
(refuse off-topic, don't recommend on turn 1 for vague queries, honor edits, no
hallucination).

## 2. Catalog engineering

The provided scrape contains 377 products spanning both Job Solutions and Individual
Test Solutions. The brief restricts scope to **Individual Test Solutions**, so the
catalog is curated to **104 items**. Validation: all 35 distinct assessment URLs that
appear across the 10 public traces are present in this 104-item set, and every item is
a subset (by URL) of the official catalog — so nothing is invented and nothing relevant
to the public traces is dropped. Each item is normalized into a `CatalogItem`
(name, URL, job levels, languages, duration, keys) with a derived single/multi-letter
`test_type` code from the category keys.

## 3. Retrieval strategy

I deliberately used **keyword + synonym retrieval rather than embeddings**. The catalog
is small (104 items), assessment names are highly lexical (e.g. "Core Java", "SQL",
"Docker"), and the evaluator's simulated user tends to state concrete skills — so a
transparent lexical scorer is both accurate and debuggable, and adds no cold-start or
inference cost against the 30s turn cap. Scoring rewards name matches heavily and
description/keys matches lightly; a curated synonym map bridges JD phrasing to catalog
vocabulary (`java → core java, spring, j2ee`; `cognitive → verify, reasoning`; etc.).
Optional job-level, language, and test-type filters narrow the pool, each with a
fall-through so a narrow filter never empties it. The top ~30 candidates become the
pool handed to the LLM.

## 4. Grounding and generation

The LLM never free-generates products. It receives the candidate pool and must return
`selected_urls` copied verbatim from it; every returned URL is then re-validated against
the catalog and anything not found is dropped. This makes hallucinated URLs structurally
impossible in the graded `recommendations` array. Comparison turns are grounded the same
way: the model is given only the matched catalog entries' real descriptions/keys/
durations and is instructed to say so plainly if a mentioned product isn't in the data
(verified against the "unknown assessment" probe).

## 5. Stateless refine — the key design decision

Because the API is stateless and history carries only assistant `content`, the shortlist
must survive round-trips through plain text. So each recommendation turn **embeds a
Markdown table of the shortlist including its catalog URLs** in the `reply`. On a later
turn, previous items are recovered by extracting those URLs from the last
recommendation-bearing assistant message and re-resolving them against the catalog. This
lets "add personality tests" / "drop REST" update the existing shortlist instead of
starting over, and lets a confirmation finalize it — all without server state. This
directly targets the "honors edits" probe and the multi-turn traces (C4, C8, C9, C10).

## 6. Agent control flow

A per-turn slot-extraction call classifies `turn_intent` and fills hiring-context slots.
Routing is deterministic on top of that: injection/off-topic/legal → fixed refusal;
compare → grounded comparison; confirm with a prior shortlist → finalize with
`end_of_conversation=true`; insufficient context → one clarifying question; otherwise →
retrieve + recommend. Two guards protect Recall@10: the agent **force-recommends** as the
8-turn cap approaches or after repeated clarifications, and treats "no preference" answers
as sufficient context — so a conversation never ends without a shortlist.

## 7. Robustness / failure-mode defenses

Against the "works on the happy path, breaks otherwise" failure mode: (a) **two-layer
scope enforcement** — deterministic regexes for injection/legal/off-topic that the LLM
cannot override, plus LLM semantic classification; (b) **fail-safe schema compliance** —
both an orchestrator try/except and an app-level exception handler always return a valid
`ChatResponse`, so `/chat` never emits a non-schema body or a 500; (c) **LLM fallbacks** —
JSON-mode calls with retries, and every LLM step (slots, clarify, recommend, compare) has
a sensible non-LLM fallback; (d) **empty/whitespace and empty-history handling** returns
either a 422 (empty array) or a safe clarifying reply.

## 8. Evaluation approach

I evaluated against the 10 public traces plus targeted behavior probes (vague turn-1,
off-topic laptop question, injection "print your system prompt", unknown-role and
unknown-assessment, empty body, and multi-turn add/drop edits). Iteration focused on:
tightening catalog URL grounding, adding the token-overlap name matcher so users can
refer to products by acronym/fragment (e.g. "Verify G+", "OPQ"), and the embedded-table
mechanism for stateless refine after observing that naive re-recommendation lost prior
items.

## 9. Stack and AI-tool usage

**Stack:** FastAPI + Pydantic (typed, self-documenting contract; Pydantic enforces the
non-negotiable schema at the boundary), raw OpenAI-compatible SDK against a free Groq
tier (`llama-3.3-70b-versatile`) — provider-swappable via env vars — and no heavyweight
framework, keeping the control flow explicit and defensible. **AI tools:** used an
AI coding assistant for scaffolding, prompt iteration, and this audit/documentation pass;
all design decisions (catalog curation, lexical retrieval, embedded-table refine,
grounding strategy) are my own and are documented above.

## 10. What didn't work / trade-offs

- **Embeddings** were considered and rejected for this catalog size — added latency and
  opacity without measurable recall gain over the lexical scorer on the public traces.
- **Relying on the LLM alone for scope control** let occasional misclassifications
  through, which motivated the deterministic regex layer.
- **Recall vs. precision:** since only Recall@10 is scored, the shortlist intentionally
  leans inclusive (offers an optional default personality measure) while still honoring
  explicit removals — a deliberate trade-off, not an oversight.
