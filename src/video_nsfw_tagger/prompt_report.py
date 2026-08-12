"""Aggregate per-prompt sidecars into a comparison report."""

import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def _discover_sidecars(run_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load all ``.nsfw.json`` sidecars nested under a run directory.

    Args:
        run_dir: Root directory of the experiment; expected layout is
            ``<run_dir>/<prompt_id>/<video>.nsfw.json``.

    Returns:
        ``(prompt_id, payload)`` tuples in sorted order.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    for prompt_dir in sorted(run_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_id = prompt_dir.name
        for path in prompt_dir.rglob("*.nsfw.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Allow the prompt_id to be overridden from the sidecar itself.
            vlm = payload.get("vlm") or {}
            if vlm.get("prompt_id"):
                prompt_id = vlm["prompt_id"]
            results.append((prompt_id, payload))
    return results


def _flatten_caption_text(caption: dict[str, Any]) -> str:
    """Return the raw text to match against the lexicon.

    If the caption is a JSON object (e.g. from a structured prompt),
    concatenate all string values so regex matching still works.
    """
    text = caption.get("caption", "") or ""
    if not text:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, dict):
        return " ".join(str(v) for v in _recursive_values(text) if v is not None)
    return str(text)


def _recursive_values(obj: object) -> Iterable[object]:
    """Yield all leaf values of a JSON-like object."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _recursive_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _recursive_values(v)
    else:
        yield obj


def _tag_hits(text: str, lexicon: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    """Case-insensitive whole-word matching like ``lexicon.find_matches``.

    Returns:
        Mapping of tag names to the patterns that matched ``text``.
    """
    lowered = text.lower()
    matches: dict[str, list[str]] = {}
    for tag, patterns in lexicon.items():
        hits = [
            str(pattern)
            for pattern in patterns
            if re.search(rf"(?<!\w){re.escape(str(pattern).lower())}(?!\w)", lowered)
        ]
        if hits:
            matches[tag] = hits
    return matches


def _video_captions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``captions`` list from the sidecar's VLM block, if any."""
    vlm = payload.get("vlm") or {}
    return vlm.get("captions") or []


def _compute_metrics(
    prompt_id: str,
    sidecars: list[dict[str, Any]],
    lexicon: dict[str, Iterable[str]],
) -> dict[str, Any]:
    """Compute metrics for one prompt across all of its sidecars.

    Returns:
        Dictionary of aggregate statistics plus the full caption dump.
    """
    attempted = 0
    succeeded = 0
    failures = 0
    lengths: list[int] = []
    elapsed_times: list[float] = []
    all_tags: set[str] = set()
    tagged_frames = 0
    pattern_counts: dict[str, int] = defaultdict(int)
    captions_for_dump: list[dict[str, Any]] = []

    for payload in sidecars:
        for caption in _video_captions(payload):
            attempted += 1
            text = _flatten_caption_text(caption)
            if not text:
                failures += 1
                continue
            succeeded += 1
            lengths.append(len(text))
            if "elapsed_s" in caption:
                elapsed_times.append(caption["elapsed_s"])
            matches = _tag_hits(text, lexicon)
            if not matches:
                matches = caption.get("matches") or {}
            if matches:
                tagged_frames += 1
                all_tags.update(matches)
                for tag, patterns in matches.items():
                    pattern_counts[tag] += len(patterns)
            captions_for_dump.append({
                "video_path": payload.get("video_path"),
                "frame": caption.get("frame"),
                "timestamp_s": caption.get("timestamp_s"),
                "caption": text,
                "tags": sorted(matches),
                "matches": matches,
                "elapsed_s": caption.get("elapsed_s"),
            })

    return {
        "prompt_id": prompt_id,
        "videos": len(sidecars),
        "attempted": attempted,
        "succeeded": succeeded,
        "failures": failures,
        "tagged_frames": tagged_frames,
        "tagged_frames_pct": (
            round(tagged_frames / attempted * 100, 2) if attempted else 0.0
        ),
        "distinct_tags": sorted(all_tags),
        "distinct_tags_count": len(all_tags),
        "pattern_counts": dict(pattern_counts),
        "avg_caption_length": (round(statistics.mean(lengths), 1) if lengths else 0.0),
        "median_caption_length": (
            round(statistics.median(lengths), 1) if lengths else 0.0
        ),
        "avg_elapsed_s": (
            round(statistics.mean(elapsed_times), 2) if elapsed_times else 0.0
        ),
        "captions": captions_for_dump,
    }


def collect(
    run_dir: Path,
    lexicon: dict[str, Iterable[str]],
) -> list[dict[str, Any]]:
    """Collect and compute metrics for every prompt in a run directory.

    Args:
        run_dir: Directory containing ``<prompt_id>/<video>.nsfw.json``.
        lexicon: Tag → patterns mapping.

    Returns:
        List of per-prompt metric dictionaries, sorted by prompt_id.
    """
    sidecars = _discover_sidecars(run_dir)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt_id, payload in sidecars:
        by_prompt[prompt_id].append(payload)

    rows = [
        _compute_metrics(prompt_id, payloads, lexicon)
        for prompt_id, payloads in sorted(by_prompt.items())
    ]
    return rows


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows by a composite score: distinct tags, then % tagged frames.

    Returns:
        Rows sorted descending by the composite score.
    """
    return sorted(
        rows,
        key=lambda r: (r["distinct_tags_count"], r["tagged_frames_pct"]),
        reverse=True,
    )


def print_table(
    rows: list[dict[str, Any]],
    lexicon: dict[str, Iterable[str]],
    console: Console | None = None,
) -> None:
    """Print a Rich table summarising prompt performance."""
    c = console or Console()
    table = Table(title="Prompt comparison report")
    table.add_column("Rank", justify="right")
    table.add_column("Prompt ID")
    table.add_column("Attempts", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Tagged %", justify="right")
    table.add_column("Distinct tags", justify="right")
    for tag in lexicon:
        table.add_column(tag, justify="right")
    table.add_column("Avg chars", justify="right")
    table.add_column("Avg s", justify="right")

    ranked = _rank(rows)
    for rank, row in enumerate(ranked, 1):
        tag_counts = row["pattern_counts"]
        table.add_row(
            str(rank),
            row["prompt_id"],
            str(row["attempted"]),
            str(row["succeeded"]),
            str(row["failures"]),
            f"{row['tagged_frames_pct']:.1f}",
            str(row["distinct_tags_count"]),
            *(str(tag_counts.get(tag, 0)) for tag in lexicon),
            f"{row['avg_caption_length']:.0f}",
            f"{row['avg_elapsed_s']:.1f}",
        )
    c.print(table)


def _markdown_report(
    rows: list[dict[str, Any]],
    lexicon: dict[str, Iterable[str]],
) -> str:
    """Build a human-readable Markdown report with tables + caption dumps.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = [
        "# Prompt Comparison Report",
        "",
        "## Summary",
        "",
        "| rank | prompt_id | attempts | success | failures | tagged% | distinct |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(_rank(rows), 1):
        lines.append(
            f"| {rank} | {row['prompt_id']} | {row['attempted']} | "
            f"{row['succeeded']} | {row['failures']} | "
            f"{row['tagged_frames_pct']:.1f} | {row['distinct_tags_count']} |"
        )
    lines.append("")
    lines.append("## Per-tag pattern counts")
    lines.append("")
    header = ["prompt_id", *lexicon.keys()]
    lines.append("| " + " | ".join(header) + " |")
    aligners = ["---:" if h != "prompt_id" else "---" for h in header]
    lines.append("| " + " | ".join(aligners) + " |")
    for row in rows:
        tag_counts = row["pattern_counts"]
        cells = [row["prompt_id"]]
        for tag in lexicon:
            cells.append(str(tag_counts.get(tag, 0)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Caption dumps")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['prompt_id']}")
        lines.append("")
        for c in row["captions"]:
            lines.append(
                f"- **{Path(c['video_path']).name}** "
                f"frame {c['frame']} (@{c['timestamp_s']:.0f}s): "
                f"{c['caption'][:160]}{'...' if len(c['caption']) > 160 else ''}"
            )
            if c["tags"]:
                lines.append(f"  - tags: {', '.join(c['tags'])}")
        lines.append("")
    return "\n".join(lines)


def write_reports(
    run_dir: Path,
    rows: list[dict[str, Any]],
    lexicon: dict[str, Iterable[str]],
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports under ``run_dir``.

    Returns:
        ``(report_json, report_md)`` paths.
    """
    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(rows, lexicon), encoding="utf-8")
    return json_path, md_path
