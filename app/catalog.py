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
        self._by_name_lower = {it.name.lower(): it for it in self.items}

    def all(self) -> List[CatalogItem]:
        return self.items

    def get_by_url(self, url: str) -> Optional[CatalogItem]:
        return self._by_url.get(url.strip())

    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        return self._by_name_lower.get(name.strip().lower())

    def find_by_partial_name(self, fragment: str, limit: int = 5) -> List[CatalogItem]:
        fragment_l = fragment.strip().lower()
        if not fragment_l:
            return []
        exact = self._by_name_lower.get(fragment_l)
        if exact:
            return [exact]
        matches = [it for it in self.items if fragment_l in it.name.lower()]
        return matches[:limit]

    def is_valid_url(self, url: str) -> bool:
        return url.strip() in self._by_url

    def valid_urls(self) -> set:
        return set(self._by_url.keys())


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return Catalog(settings.catalog_path)
