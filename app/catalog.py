import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from app.config import settings

KEY_TO_CODE: Dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9+#\.]+")
_GENERIC_NAME_WORDS = {
    "new", "the", "and", "for", "of", "test", "assessment", "report", "development",
    "adaptive", "level", "entry", "general", "individual", "standard", "profile",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


@dataclass
class CatalogItem:
    entity_id: str
    name: str
    url: str
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration_raw: str = ""
    duration_minutes: Optional[int] = None
    remote: bool = True
    adaptive: bool = False
    description: str = ""
    keys: List[str] = field(default_factory=list)

    @property
    def test_type(self) -> str:
        codes = [KEY_TO_CODE.get(k, "") for k in self.keys]
        codes = [c for c in codes if c]
        # stable de-dup preserving order
        seen = []
        for c in codes:
            if c not in seen:
                seen.append(c)
        return ",".join(seen)

    @property
    def search_blob(self) -> str:
        return " ".join(
            [
                self.name,
                self.description,
                " ".join(self.keys),
                " ".join(self.job_levels),
                " ".join(self.languages),
            ]
        ).lower()


def _parse_duration_minutes(raw: str) -> Optional[int]:
    if not raw:
        return None
    match = re.search(r"(\d+)", raw)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _normalize_item(raw: dict) -> CatalogItem:
    return CatalogItem(
        entity_id=str(raw.get("entity_id", "")),
        name=str(raw.get("name", "")).strip(),
        url=str(raw.get("link", raw.get("url", ""))).strip(),
        job_levels=list(raw.get("job_levels", []) or []),
        languages=list(raw.get("languages", []) or []),
        duration_raw=str(raw.get("duration", "") or ""),
        duration_minutes=_parse_duration_minutes(str(raw.get("duration", "") or "")),
        remote=str(raw.get("remote", "yes")).lower() != "no",
        adaptive=str(raw.get("adaptive", "no")).lower() == "yes",
        description=str(raw.get("description", "") or "").strip(),
        keys=list(raw.get("keys", []) or []),
    )


class Catalog:
    def __init__(self, path: str) -> None:
        self.path = path
        self.items: List[CatalogItem] = []
        self._by_url: Dict[str, CatalogItem] = {}
        self._by_url_normalized: Dict[str, CatalogItem] = {}
        self._by_name_lower: Dict[str, CatalogItem] = {}
        self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        items = [_normalize_item(r) for r in raw_items]
        # de-dup by URL, keep first occurrence
        seen_urls = set()
        deduped = []
        for it in items:
            if not it.url or it.url in seen_urls:
                continue
            seen_urls.add(it.url)
            deduped.append(it)
        self.items = deduped
        self._by_url = {it.url: it for it in self.items}
        self._by_url_normalized = {_normalize_url(it.url): it for it in self.items}
        self._by_name_lower = {it.name.lower(): it for it in self.items}

    def all(self) -> List[CatalogItem]:
        return self.items

    def get_by_url(self, url: str) -> Optional[CatalogItem]:
        if not url:
            return None
        exact = self._by_url.get(url.strip())
        if exact:
            return exact
        # tolerate trailing-slash / whitespace / case drift from LLM output
        # without ever accepting a url that isn't actually in the catalog
        return self._by_url_normalized.get(_normalize_url(url))

    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        return self._by_name_lower.get(name.strip().lower())

    def find_by_partial_name(self, fragment: str, limit: int = 5) -> List[CatalogItem]:
        fragment_l = fragment.strip().lower()
        if not fragment_l:
            return []

        exact = self._by_name_lower.get(fragment_l)
        if exact:
            return [exact]

        # direct substring match (fragment fully contained in the catalog name,
        # or the catalog name fully contained in a longer fragment/sentence)
        substring_matches = [
            it for it in self.items
            if fragment_l in it.name.lower() or it.name.lower() in fragment_l
        ]
        if substring_matches:
            return substring_matches[:limit]

        # token-overlap fallback: handles cases like "Verify G+" not being a
        # contiguous substring of "SHL Verify Interactive G+" because a word
        # sits in between -- users reference products by fragments/acronyms
        # that don't preserve the catalog's exact word order
        fragment_tokens = set(t for t in _tokenize(fragment_l) if len(t) >= 2)
        if not fragment_tokens:
            return []

        scored = []
        for it in self.items:
            name_tokens = [
                t for t in _tokenize(it.name.lower())
                if len(t) >= 2 and t not in _GENERIC_NAME_WORDS
            ]
            if not name_tokens:
                continue
            overlap = sum(1 for t in name_tokens if t in fragment_tokens)
            if overlap == 0:
                continue
            ratio = overlap / len(name_tokens)
            if overlap >= 2 or ratio >= 0.5:
                scored.append((overlap + ratio, it))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [it for _, it in scored[:limit]]

    def is_valid_url(self, url: str) -> bool:
        return self.get_by_url(url) is not None

    def valid_urls(self) -> set:
        return set(self._by_url.keys())


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return Catalog(settings.catalog_path)
