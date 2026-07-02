# SHL Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent
(*"I'm hiring a Java developer"*) to a grounded shortlist of **SHL Individual Test
Solutions** through dialogue. It clarifies when a request is too vague, recommends
1–10 catalog assessments once it has enough context, refines the shortlist when
constraints change, compares products using catalog data, and refuses anything
outside SHL assessment selection.

Built for the SHL Labs "Build a Conversational SHL Assessment Recommender" take-home.

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration (environment variables)](#configuration-environment-variables)
- [Running locally](#running-locally)
- [API documentation](#api-documentation)
- [Deployment](#deployment)
- [Testing](#testing)
- [Design assumptions](#design-assumptions)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

---

## Features

| Behavior | How it is handled |
|----------|-------------------|
| **Clarify** vague queries | Does not recommend on turn 1 for under-specified requests; asks one focused question. |
| **Recommend** 1–10 assessments | Keyword + synonym retrieval builds a candidate pool; the LLM selects from that pool only. |
| **Refine** mid-conversation | Previously shown items are re-parsed from history and carried over; only what the user asks to change is changed. |
| **Compare** products | Grounded strictly in catalog descriptions/keys/durations for the matched items — no model prior. |
| **Stay in scope** | Deterministic rule-based guards + LLM classification refuse off-topic, legal/compliance, and prompt-injection requests. |
| **Grounding** | Every returned URL is validated against the catalog; model-invented names/URLs are dropped. |
| **Stateless** | No per-conversation server state; the full history is re-derived from the request each call. |

## Architecture

Each `POST /chat` turn runs a small, deterministic pipeline:

```
messages ──► slot extraction (LLM + rule-based overrides)
          │      → role, skills, seniority, language, turn_intent, ...
          │
          ├─ injection / off-topic / legal   ──► deterministic refusal
          ├─ compare_request                 ──► catalog-grounded comparison
          ├─ confirm (+ prior shortlist)     ──► finalize shortlist, end_of_conversation=true
          ├─ not enough context yet          ──► single clarifying question
          └─ enough context / forced         ──► retrieve candidates ──► LLM selects ──►
                                                  validate URLs against catalog ──► shortlist
```

Key design points:

- **Retrieval before generation.** A lightweight keyword/synonym scorer narrows the
  104-item catalog to a candidate pool. The LLM only *selects and orders* from that
  pool, so it can never invent a product or URL.
- **The reply embeds a Markdown table of the shortlist (with URLs).** Because the API
  is stateless and the history only carries assistant `content`, embedding the URLs in
  the reply is what lets a later refine/confirm turn recover the previous shortlist.
- **Two layers of scope enforcement.** Regex guards (`app/utils/text.py`) catch
  injection/legal/off-topic deterministically even if the LLM misclassifies; the LLM
  adds broader semantic coverage.
- **Fail-safe schema compliance.** `/chat` can never return a non-schema body: an
  orchestrator-level try/except and an app-level exception handler both fall back to a
  valid `ChatResponse`.
- **Turn-cap aware.** The agent force-produces a shortlist as the 8-turn cap approaches
  so a conversation never ends without recommendations.

## Project structure

```
shl-assessment-recommender/
├── app/
│   ├── main.py            # FastAPI app: /health and /chat, global error handler
│   ├── orchestrator.py    # Turn routing: clarify / recommend / refine / compare / confirm / refuse
│   ├── slots.py           # LLM slot extraction + deterministic intent overrides
│   ├── retrieval.py       # Keyword + synonym candidate retrieval and scoring
│   ├── catalog.py         # Catalog loading, URL/name lookup, test_type mapping
│   ├── llm.py             # OpenAI-compatible client with retries + JSON mode
│   ├── config.py          # Environment-driven settings
│   ├── schemas.py         # Pydantic request/response models (the graded contract)
│   ├── prompts/           # System identity + slot-extraction + generation prompts
│   └── utils/text.py      # History formatting + injection/legal/off-topic/confirm regexes
├── data/
│   └── shl_catalog.json   # 104 curated SHL Individual Test Solutions
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (developed on 3.11)
- An API key for any **OpenAI-compatible** chat-completions endpoint
  (Groq is the default; OpenAI, OpenRouter, local vLLM, etc. also work)

## Installation

```bash
git clone https://github.com/devarenidhi25/shl-assessment-recommender.git
cd shl-assessment-recommender

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration (environment variables)

Configuration is read directly from process environment variables
(`app/config.py`). **There is no `.env` auto-loading** — set the variables in your
shell or hosting platform's dashboard before starting the server.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_API_KEY` | *(empty)* | **Required.** API key for the LLM provider. |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible base URL. |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model name. |
| `LLM_TEMPERATURE_EXTRACT` | `0.0` | Temperature for slot extraction. |
| `LLM_TEMPERATURE_GENERATE` | `0.3` | Temperature for reply/recommendation generation. |
| `LLM_TIMEOUT_SECONDS` | `20` | Per-call HTTP timeout. |
| `LLM_MAX_RETRIES` | `2` | Retries on transient LLM errors. |
| `CATALOG_PATH` | `data/shl_catalog.json` | Path to the catalog file. |
| `MAX_TURNS` | `8` | Turn cap the agent designs around. |
| `MAX_RECOMMENDATIONS` | `10` | Hard ceiling on shortlist size. |
| `RETRIEVAL_POOL_SIZE` | `30` | Candidate pool size passed to the LLM. |

**Setting the required key:**

```bash
# Windows (PowerShell)
$env:LLM_API_KEY="your_key_here"

# Windows (cmd)
set LLM_API_KEY=your_key_here

# macOS / Linux
export LLM_API_KEY="your_key_here"
```

> Without a valid `LLM_API_KEY`, `/health` still returns 200 and `/chat` stays
> schema-compliant, but the agent falls back to generic clarifying questions and
> cannot produce recommendations.

## Running locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Interactive docs are available at `http://localhost:8000/docs`.

## API documentation

The API is **stateless**: every `POST /chat` call carries the full conversation
history and the service stores no per-conversation state.

### `GET /health`

Returns `200` with:

```json
{ "status": "ok" }
```

### `POST /chat`

**Request**

```json
{
  "messages": [
    { "role": "user", "content": "Hiring a Java developer who works with stakeholders" },
    { "role": "assistant", "content": "Sure. What is the seniority level?" },
    { "role": "user", "content": "Mid-level, around 4 years" }
  ]
}
```

- `messages` must be a non-empty array. Each message has `role`
  (`"user"` or `"assistant"`) and a string `content`.
- An empty `messages` array returns `422`.

**Response**

```json
{
  "reply": "Got it. Here are assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    { "name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K" },
    { "name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/...", "test_type": "P" }
  ],
  "end_of_conversation": false
}
```

- `recommendations` is an **empty array** while the agent is clarifying or refusing,
  and a **1–10 item array** once it commits to a shortlist. Each item is
  `{ name, url, test_type }` copied verbatim from the catalog.
- `test_type` is a comma-joined code string derived from the catalog category keys:
  `A` Ability & Aptitude, `B` Biodata & Situational Judgment, `C` Competencies,
  `D` Development & 360, `E` Assessment Exercises, `K` Knowledge & Skills,
  `P` Personality & Behavior, `S` Simulations.
- `end_of_conversation` is `true` only when the shortlist is finalized/confirmed.

## Deployment

Any platform that can run a Python web process works (Render, Railway, Fly, Hugging
Face Spaces, etc.). Start command:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set the environment variables from the table above in the platform dashboard
(at minimum `LLM_API_KEY`). The catalog loads at process startup so the first
`/health` after cold start is fast.

## Testing

The manual test matrix executed before submission covers: vague-query clarification,
single- and multi-turn recommendation, refine/edit honoring, comparison (real and
unknown products), off-topic refusal, prompt-injection resistance, empty-body
validation (`422`), and conversation-memory carry-over.

Quick smoke test with the bundled `TestClient`:

```bash
python -c "from fastapi.testclient import TestClient; from app.main import app; \
c=TestClient(app); print(c.get('/health').json()); \
print(c.post('/chat', json={'messages':[{'role':'user','content':'I need an assessment.'}]}).json())"
```

## Design assumptions

- **Catalog scope.** The catalog is restricted to **Individual Test Solutions**
  (104 items). Pre-packaged Job Solutions are intentionally excluded per the brief.
  All 35 distinct assessment URLs referenced across the 10 public traces are present
  in this catalog.
- **Reports/derived products count as recommendable items** (e.g. OPQ Leadership
  Report), matching the public traces.
- **The simulated evaluator user ends the conversation once a shortlist is shown**, so
  the agent optimizes for surfacing a correct, complete shortlist quickly rather than
  minimizing turns.
- **Recall-oriented shortlists.** Because scoring is Recall@10, the agent leans toward
  including all plausibly relevant items (a sensible default personality measure is
  offered and flagged as optional), while still honoring explicit removals.

## Limitations

- Retrieval is keyword + curated-synonym based, not embeddings; very unusual phrasing
  that shares no vocabulary with the catalog may under-retrieve before the LLM step.
- Recommendation and comparison quality depend on the configured LLM provider being
  reachable within the timeout budget.
- Off-topic detection beyond the hard-coded regex categories relies on LLM
  classification.

## Future improvements

- Add embedding/vector retrieval (FAISS/Chroma) as a recall booster alongside the
  keyword scorer.
- Add an automated Recall@10 harness that replays the public traces on every change.
- Cache slot-extraction results per prefix to cut latency under the 30s turn cap.
- Expand the synonym map from observed evaluator phrasings.
