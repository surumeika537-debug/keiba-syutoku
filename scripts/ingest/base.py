"""Abstract fetcher interface. Concrete sources (netkeiba, JRA, JRA-VAN, etc.) implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseFetcher(ABC):
    """Defines the contract between the orchestrator and a data source.

    A new source = a new subclass. The orchestrator does not import any source-specific module.
    """

    name: str = "base"

    @abstractmethod
    def discover_race_ids(self, year: int, grades: tuple[str, ...]) -> list[str]:
        """Return race_ids in `year` whose grade is in `grades`. May be a best-effort crawl."""

    @abstractmethod
    def race_html_path(self, race_id: str) -> Path:
        """Where the raw HTML for `race_id` is stored on disk."""

    @abstractmethod
    def fetch_race_html(self, race_id: str) -> str:
        """Download (or read from cache) the HTML for `race_id`. Implementations should respect
        the rate limit configured in `src.config.FETCH_SLEEP_SECONDS` when making network calls."""
