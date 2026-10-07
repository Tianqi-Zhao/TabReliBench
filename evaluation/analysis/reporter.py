"""Build a human-readable text summary of an analysis run."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


class SummaryReporter:
    """Tiny ordered text-section accumulator."""

    def __init__(self) -> None:
        self._sections: list[str] = []

    def add_header(self, title: str) -> None:
        self._sections.append("=" * 72)
        self._sections.append(title)
        self._sections.append("=" * 72)

    def add_kv(self, key: str, value) -> None:
        self._sections.append(f"{key:8s}: {value}")

    def add_section(self, title: str, content: str) -> None:
        self._sections.append("")
        self._sections.append(f"-- {title} ".ljust(72, "-"))
        self._sections.append(content)

    def add_blank(self) -> None:
        self._sections.append("")

    def add_lines(self, lines: Iterable[str]) -> None:
        for ln in lines:
            self._sections.append(ln)

    def render(self) -> str:
        return "\n".join(self._sections)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
        return path
