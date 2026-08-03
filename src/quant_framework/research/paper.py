"""
Paper – immutable research paper record
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import hashlib
import json


@dataclass(frozen=True)
class Paper:
    """Immutable paper record – deterministic, reproducible"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str  # SSRN, Google Scholar, arXiv, etc.
    published_date: Optional[str]
    keywords: List[str]
    id: str = field(init=False)

    def __post_init__(self):
        # Deterministic ID based on title + url
        object.__setattr__(self, 'id', self._generate_id())

    def _generate_id(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.title.encode())
        hasher.update(self.url.encode())
        return hasher.hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date,
            "keywords": self.keywords
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(data: Dict) -> 'Paper':
        return Paper(
            title=data.get('title', ''),
            authors=data.get('authors', []),
            abstract=data.get('abstract', ''),
            url=data.get('url', ''),
            source=data.get('source', 'Unknown'),
            published_date=data.get('published_date'),
            keywords=data.get('keywords', [])
        )
