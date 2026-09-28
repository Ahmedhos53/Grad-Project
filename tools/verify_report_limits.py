"""Verify final report chapter and total word limits from the Markdown source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "report_revision" / "CabInspector_Final_Report_Revised.md"
CHAPTER_LIMITS = {1: 1000, 2: 2500, 3: 2000, 4: 2500, 5: 2500, 6: 1000}
TOTAL_LIMIT = 10_500
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
CHAPTER_PATTERN = re.compile(r"^# Chapter ([1-6]):[^\n]*$", re.MULTILINE)


def _count_body_words(markdown: str) -> int:
    """Count chapter prose and table contents, excluding chapter titles and legends."""

    body_lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("!["):
            continue
        if stripped.startswith("```"):
            continue
        if re.match(r"^\*{0,2}\s*(?:Figure|Table)\s+\d", stripped, flags=re.IGNORECASE):
            continue
        body_lines.append(line)
    return len(WORD_PATTERN.findall(" ".join(body_lines)))


def verify_report(report_path: Path = REPORT_PATH) -> dict[str, object]:
    text = report_path.read_text(encoding="utf-8")
    matches = list(CHAPTER_PATTERN.finditer(text))
    chapter_numbers = [int(match.group(1)) for match in matches]
    references_match = re.search(r"^# References\s*$", text, flags=re.MULTILINE)
    chapters: dict[str, dict[str, object]] = {}
    total_words = 0
    if chapter_numbers == [1, 2, 3, 4, 5, 6] and references_match is not None:
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else references_match.start()
            number = int(match.group(1))
            count = _count_body_words(text[match.end() : end])
            total_words += count
            limit = CHAPTER_LIMITS[number]
            chapters[str(number)] = {
                "words": count,
                "limit": limit,
                "within_limit": count <= limit,
            }

    passed = (
        chapter_numbers == [1, 2, 3, 4, 5, 6]
        and references_match is not None
        and len(chapters) == 6
        and all(item["within_limit"] for item in chapters.values())
        and total_words <= TOTAL_LIMIT
    )
    return {
        "status": "report_word_limit_verification",
        "passed": passed,
        "report": str(report_path),
        "chapter_order": chapter_numbers,
        "chapters": chapters,
        "body_words": total_words,
        "total_limit": TOTAL_LIMIT,
        "count_exclusions": [
            "references, the report title, and chapter titles are outside the count",
            "Figure/Table captions and image lines are excluded; table contents and fenced-code contents are counted",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = verify_report(args.report)
    except OSError as exc:
        print(f"Could not read report: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Verification written: {args.output}")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
