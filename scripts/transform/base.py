"""Abstract parser interface. Concrete parsers must return a uniform dict regardless of source."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class ParsedRace:
    race: dict[str, Any]            # one race row
    entries: list[dict[str, Any]]   # one row per horse
    payouts: list[dict[str, Any]]   # one row per (bet_type, combination)


class BaseParser(ABC):
    name: str = "base"

    @abstractmethod
    def parse(self, html: str, race_id: str) -> ParsedRace:
        """Parse one race's HTML into structured rows.

        Implementations should:
        - never raise on missing optional fields — return None instead
        - normalize trifecta combination strings to 'h1-h2-h3' with integer numbers
        - tolerate minor HTML drift (try multiple selectors)
        """
