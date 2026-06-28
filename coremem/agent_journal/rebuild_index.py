"""Rebuild weekly, monthly, and index navigation files from daily pages."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path


def rebuild_index(root: str | Path) -> dict[str, int]:
    """Scan daily/ directory and regenerate weekly/, monthly/, and index.md.

    Returns counts of files written: {weekly, monthly, index}.
    """
    root = Path(root)
    daily_dir = root / "daily"
    if not daily_dir.exists():
        return {"weekly": 0, "monthly": 0, "index": 0}

    daily_files = sorted(daily_dir.glob("*.md"))
    if not daily_files:
        return {"weekly": 0, "monthly": 0, "index": 0}

    by_week: dict[str, list[tuple[str, str]]] = {}
    by_month: dict[str, list[str]] = {}

    for f in daily_files:
        stem = f.stem
        try:
            d = date.fromisoformat(stem)
        except ValueError:
            continue
        iso_year, iso_week, _ = d.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"
        month_key = f"{iso_year}-{d.month:02d}"
        desc = _first_section_title(f)
        by_week.setdefault(week_key, []).append((stem, desc))
        by_month.setdefault(month_key, []).append(week_key)

    weekly_dir = root / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    weekly_count = 0
    for week_key, entries in sorted(by_week.items()):
        year, w = week_key.split("-W")
        first_day = date.fromisocalendar(int(year), int(w), 1)
        last_day = first_day + timedelta(days=6)
        lines = [
            f"# Week {int(w)} ({first_day.isoformat()} to {last_day.isoformat()})",
            "",
        ]
        for stem, desc in entries:
            suffix = f" — {desc}" if desc else ""
            lines.append(f"- [{stem}](daily/{stem}.md){suffix}")
        lines.append("")
        (weekly_dir / f"{week_key}.md").write_text("\n".join(lines), encoding="utf-8")
        weekly_count += 1

    monthly_dir = root / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    monthly_count = 0
    for month_key, week_keys in sorted(by_month.items()):
        unique_weeks = sorted(set(week_keys))
        lines = [f"# {month_key}", ""]
        for wk in unique_weeks:
            first_entry = by_week[wk][0] if by_week.get(wk) else ""
            desc = by_week[wk][0][1] if by_week.get(wk) and by_week[wk][0][1] else ""
            suffix = f" — {desc}" if desc else ""
            lines.append(f"- [Week {wk.split('-W')[1]}](weekly/{wk}.md){suffix}")
        lines.append("")
        (monthly_dir / f"{month_key}.md").write_text("\n".join(lines), encoding="utf-8")
        monthly_count += 1

    index_lines = ["# AgentJournal Index", ""]
    for month_key in sorted(by_month):
        index_lines.append(f"- [{month_key}](monthly/{month_key}.md)")
    index_lines.append("")
    (root / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    return {"weekly": weekly_count, "monthly": monthly_count, "index": 1}


def _first_section_title(path: Path) -> str:
    """Extract the first section title from a daily page."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## \d{2}:\d{2} - (.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""
