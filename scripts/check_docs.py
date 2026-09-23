#!/usr/bin/env python3
"""Validate the authoritative documentation routing and active section plans."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = (
    Path("AGENTS.md"),
    Path("docs/PROJECT_PLAN.md"),
    Path("docs/CURRENT_STATUS.md"),
    Path("docs/requirements.md"),
    Path("docs/DOCS_INDEX.md"),
    Path("docs/plans/INDEX.md"),
    Path("docs/acceptance.md"),
    Path("docs/decisions.md"),
)
ACTIVE_PLANS = (
    Path("docs/plans/S01-scene-physics-checkpoints.md"),
    Path("docs/plans/S02-observation-target-selection.md"),
    Path("docs/plans/S03-approach-control-handoff.md"),
    Path("docs/plans/S04-contact-excavation-demonstrations.md"),
    Path("docs/plans/S05-imitation-and-rl.md"),
    Path("docs/plans/S06-three-scoop-objective-evaluation.md"),
    Path("docs/plans/S07-truck-reproduction-submission.md"),
)
RETIRED_DUPLICATES = (
    Path("复杂介质工况自主挖掘项目总体规划_V0.1.md"),
    Path("docs/START-HERE.md"),
)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
DECISION_HEADING = re.compile(r"^##\s+(D-\d{3})\b", re.MULTILINE)


def local_link_errors(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for raw in LINK.findall(text):
        target = raw.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            errors.append(
                f"broken link: {path.relative_to(ROOT)} -> {raw}")
    return errors


def main() -> int:
    errors: list[str] = []
    for relative in (*CORE, *ACTIVE_PLANS):
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing required document: {relative}")
    for relative in RETIRED_DUPLICATES:
        if (ROOT / relative).exists():
            errors.append(f"retired duplicate unexpectedly exists: {relative}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for relative in CORE[1:6]:
        if relative.as_posix() not in agents:
            errors.append(f"AGENTS routing omits {relative}")
    if "全部状态为`active`的小节方案" not in agents:
        errors.append("AGENTS does not require all active section plans")

    plan_index = (ROOT / "docs/plans/INDEX.md").read_text(encoding="utf-8")
    current = (ROOT / "docs/CURRENT_STATUS.md").read_text(encoding="utf-8")
    for relative in ACTIVE_PLANS:
        name = relative.name
        if name not in plan_index:
            errors.append(f"plan index omits {name}")
        if name not in current:
            errors.append(f"current status omits active plan {name}")
        text = (ROOT / relative).read_text(encoding="utf-8")
        for field in ("状态：", "权威范围：", "最后复核：", "更新触发："):
            if field not in text[:1200]:
                errors.append(f"{relative} missing metadata field {field}")

    authoritative = [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "docs/PROJECT_PLAN.md",
        ROOT / "docs/CURRENT_STATUS.md",
        ROOT / "docs/DOCS_INDEX.md",
        ROOT / "docs/plans/INDEX.md",
        ROOT / "docs/runbooks/a30-platform.md",
        *(ROOT / relative for relative in ACTIVE_PLANS),
    ]
    for path in authoritative:
        errors.extend(local_link_errors(path))

    decisions = (ROOT / "docs/decisions.md").read_text(encoding="utf-8")
    ids = DECISION_HEADING.findall(decisions)
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate decision headings: {duplicates}")
    if "D-098" not in decisions[:3000]:
        errors.append("current decision index does not expose D-098 near the top")

    acceptance = (ROOT / "docs/acceptance.md").read_text(encoding="utf-8")
    if not re.search(
            r"当前能力摘要（\d{4}-\d{2}-\d{2}）", acceptance[:3000]):
        errors.append("acceptance current summary is missing near the top")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "DOCS_CHECK_PASS "
        f"core={len(CORE)} active_plans={len(ACTIVE_PLANS)} "
        f"checked_links={len(authoritative)} decisions={len(ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
