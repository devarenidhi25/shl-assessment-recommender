import re
from typing import Dict, List, Optional

from app.catalog import CatalogItem, get_catalog
from app.config import settings

_TOKEN_RE = re.compile(r"[a-zA-Z0-9+#\.]+")

# Lightweight synonym expansion so common phrasing in job descriptions
# maps onto catalog vocabulary without needing embeddings.
SYNONYMS: Dict[str, List[str]] = {
    "java": ["java", "core java", "j2ee", "spring"],
    "js": ["javascript"],
    "reactjs": ["react"],
    "react": ["reactjs", "react"],
    "node": ["node.js", "nodejs"],
    "aws": ["amazon web services", "aws"],
    "k8s": ["kubernetes"],
    "cognitive": ["ability", "aptitude", "reasoning", "verify"],
    "personality": ["opq", "personality", "behavior", "behaviour"],
    "leadership": ["opq leadership", "leadership", "executive scenarios", "management scenarios"],
    "sql": ["sql", "database"],
    "excel": ["excel", "ms excel", "microsoft excel"],
    "word": ["word", "ms word", "microsoft word"],
    "customer service": ["customer service", "contact center", "call center", "call centre"],
    "spanish": ["spanish", "latin american spanish"],
    "safety": ["safety", "dependability", "dsi"],
    "sales": ["sales"],
    "graduate": ["graduate", "entry-level", "entry level"],
    "situational judgement": ["situational judgment", "biodata", "scenarios"],
    "situational judgment": ["situational judgment", "biodata", "scenarios"],
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def expand_query_terms(query: str) -> List[str]:
    tokens = tokenize(query)
    terms = set(tokens)
    lowered = (query or "").lower()
    for key, expansions in SYNONYMS.items():
        if key in lowered:
            terms.update(expansions)
    return list(terms)


def score_item(item: CatalogItem, terms: List[str]) -> float:
    if not terms:
        return 0.0
    blob = item.search_blob
    score = 0.0
    for term in terms:
        term = term.strip()
        if not term:
            continue
        if term in item.name.lower():
            score += 3.0
        occurrences = blob.count(term)
        if occurrences:
            score += min(occurrences, 3) * 1.0
    return score


def filter_by_job_level(items: List[CatalogItem], level_hint: Optional[str]) -> List[CatalogItem]:
    if not level_hint:
        return items
    level_hint_l = level_hint.lower()
    filtered = [
        it for it in items
        if not it.job_levels or any(level_hint_l in jl.lower() or jl.lower() in level_hint_l for jl in it.job_levels)
    ]
    return filtered if filtered else items


def filter_by_language(items: List[CatalogItem], language_hint: Optional[str]) -> List[CatalogItem]:
    if not language_hint:
        return items
    lang_l = language_hint.lower()
    filtered = [
        it for it in items
        if not it.languages or any(lang_l in l.lower() or l.lower() in lang_l for l in it.languages)
    ]
    return filtered if filtered else items


def filter_by_test_types(items: List[CatalogItem], type_codes: Optional[List[str]]) -> List[CatalogItem]:
    if not type_codes:
        return items
    wanted = set(c.upper() for c in type_codes)
    filtered = [it for it in items if wanted & set(it.test_type.split(",")) if it.test_type]
    return filtered if filtered else items


def retrieve_candidates(
    query_text: str,
    skills: Optional[List[str]] = None,
    job_level: Optional[str] = None,
    language: Optional[str] = None,
    test_type_prefs: Optional[List[str]] = None,
    exclude_urls: Optional[List[str]] = None,
    pool_size: Optional[int] = None,
) -> List[CatalogItem]:
    catalog = get_catalog()
    pool_size = pool_size or settings.retrieval_pool_size
    exclude_urls = set(exclude_urls or [])

    combined_query = " ".join([query_text or ""] + (skills or []))
    terms = expand_query_terms(combined_query)

    candidates = catalog.all()
    candidates = filter_by_job_level(candidates, job_level)
    candidates = filter_by_language(candidates, language)
    candidates = filter_by_test_types(candidates, test_type_prefs)

    scored = [(score_item(it, terms), it) for it in candidates if it.url not in exclude_urls]
    scored = [pair for pair in scored if pair[0] > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if not scored:
        # fall back to unfiltered scoring against the full catalog so a
        # narrow field filter never produces an empty pool
        scored = [
            (score_item(it, terms), it)
            for it in catalog.all()
            if it.url not in exclude_urls
        ]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)

    return [it for _, it in scored[:pool_size]]


_GENERIC_NAME_WORDS = {
    "new", "the", "and", "for", "of", "test", "assessment", "report", "development",
    "adaptive", "level", "entry", "general", "individual", "standard", "profile",
}


def find_items_mentioned(text: str, limit_each: int = 2) -> List[CatalogItem]:
    """Fuzzy-match catalog product names explicitly mentioned in free text
    (used as a fallback for the Compare behavior when the LLM did not
    extract clean compare_subjects). Matches on word overlap rather than
    requiring the full descriptive catalog name to appear verbatim, since
    users typically refer to products by a short acronym or fragment
    (e.g. "OPQ" for "Occupational Personality Questionnaire OPQ32r")."""
    catalog = get_catalog()
    text_l = (text or "").lower()

    # direct substring hits (short acronyms/fragments contained in the name)
    scored: List[tuple] = []
    for item in catalog.all():
        name_l = re.sub(r"\(new\)|\(adaptive\)", "", item.name.lower()).strip()
        if len(name_l) < 3:
            continue
        if name_l in text_l:
            scored.append((100, item))
            continue

        words = [w for w in tokenize(name_l) if len(w) >= 3 and w not in _GENERIC_NAME_WORDS]
        if not words:
            continue
        text_tokens = set(tokenize(text_l))
        overlap = sum(1 for w in words if w in text_tokens)
        if overlap >= 1 and overlap / len(words) >= 0.4:
            scored.append((overlap, item))

    scored.sort(key=lambda p: p[0], reverse=True)
    return [it for _, it in scored[: max(limit_each * 4, 6)]]
