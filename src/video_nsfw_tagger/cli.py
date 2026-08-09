"""Typer CLI for the video-nsfw-tagger."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from video_nsfw_tagger import aggregate, classify, config, db, device, extract, sidecar

app = typer.Typer(help="Video NSFW Tagger", no_args_is_help=True)
console = Console()


def _discover_videos(target: Path, recursive: bool) -> List[Path]:
    """Return supported video files for a target path."""
    if target.is_file():
        return [target]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in target.glob(pattern)
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTS
    )


@app.command()
def scan(
    target: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            help="Video file or directory to scan",
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Scan directories recursively"),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="NSFW score threshold",
            min=0.0,
            max=1.0,
        ),
    ] = config.DEFAULT_THRESHOLD,
    fps: Annotated[
        float,
        typer.Option("--fps", help="Frame sampling rate (frames per second)", min=0.1),
    ] = config.DEFAULT_FPS,
    max_duration: Annotated[
        Optional[float],
        typer.Option(
            "--max-duration",
            help="Only process the first N seconds",
            min=1.0,
        ),
    ] = None,
    db_path: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path"),
    ] = Path(config.DEFAULT_DB),
    device_name: Annotated[
        str,
        typer.Option("--device", help="auto|cpu|cuda|mps"),
    ] = "auto",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="ViT inference batch size", min=1),
    ] = config.DEFAULT_BATCH_SIZE,
    vlm: Annotated[
        bool,
        typer.Option("--vlm", help="Enable VLM captioning (Phase B no-op)"),
    ] = False,
    vlm_model: Annotated[
        str,
        typer.Option("--vlm-model", help="VLM model ID"),
    ] = config.DEFAULT_VLM_MODEL,
    vlm_top_k: Annotated[
        Optional[int],
        typer.Option("--vlm-top-k", help="Max flagged frames to caption", min=1),
    ] = None,
    lexicon: Annotated[
        Path,
        typer.Option("--lexicon", help="Lexicon file for act tags"),
    ] = config.DEFAULT_LEXICON,
) -> None:
    """Scan video(s) and write sidecar + index entries."""
    if vlm:
        console.print(
            "[yellow]--vlm is a no-op in Phase A builds; "
            "VLM will be enabled in Phase B.[/yellow]"
        )

    videos = _discover_videos(target, recursive)
    if not videos:
        console.print("[yellow]No supported video files found.[/yellow]")
        raise typer.Exit(0)

    resolved = device.resolve_device(device_name)
    try:
        pipe = classify.load_pipeline(
            resolved,
            model_name=config.DEFAULT_VIT_MODEL,
            batch_size=batch_size,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load ViT model:[/red] {exc}")
        raise typer.Exit(1)

    conn = db.init_db(db_path)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning videos...", total=len(videos))
            for video in videos:
                progress.update(task, description=f"Scanning {video.name}")
                with extract.extracted_frames(
                    video, fps=fps, max_duration=max_duration
                ) as frames:
                    if not frames:
                        console.print(
                            f"[yellow]No frames extracted from {video}[/yellow]"
                        )
                        progress.advance(task)
                        continue

                    image_paths = [p for _, _, p in frames]
                    scores = classify.classify_batch(
                        pipe, image_paths, batch_size=batch_size
                    )
                    result = aggregate.aggregate(scores, threshold=threshold, fps=fps)
                    duration = extract.get_duration(video)
                    side = sidecar.sidecar_path(video)
                    created = datetime.now(timezone.utc).astimezone().isoformat()

                    payload = {
                        "schema_version": 1,
                        "video_path": str(video),
                        "duration_s": duration,
                        "fps_sampled": fps,
                        "frames_total": len(scores),
                        "threshold": threshold,
                        "device": resolved,
                        "model": config.DEFAULT_VIT_MODEL,
                        "nsfw_percent": round(result.nsfw_percent, 2),
                        "max_score": result.max_score,
                        "verdict": result.verdict,
                        "flagged_frames": result.flagged_frames,
                        "vlm": None,
                        "created_at": created,
                    }
                    sidecar.write_sidecar(side, payload)
                    db.upsert_video(
                        conn,
                        {
                            "path": str(video),
                            "duration_s": duration,
                            "frames_total": len(scores),
                            "nsfw_percent": round(result.nsfw_percent, 2),
                            "max_score": result.max_score,
                            "verdict": result.verdict,
                            "threshold": threshold,
                            "model": config.DEFAULT_VIT_MODEL,
                            "vlm_model": None,
                            "act_tags": "[]",
                            "sidecar_path": str(side),
                            "scanned_at": created,
                        },
                    )
                    console.print(
                        f"[green]Scanned[/green] {video} → {result.verdict} "
                        f"(max={result.max_score}, "
                        f"flagged={len(result.flagged_frames)})"
                    )
                progress.advance(task)
    finally:
        conn.close()


@app.command()
def report(
    db_path: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path"),
    ] = Path(config.DEFAULT_DB),
    verdict: Annotated[
        Optional[str],
        typer.Option("--verdict", help="Filter by verdict"),
    ] = None,
    min_percent: Annotated[
        Optional[float],
        typer.Option("--min-percent", help="Minimum NSFW percent"),
    ] = None,
) -> None:
    """Show a report of all indexed scans."""
    conn = db.init_db(db_path)
    try:
        rows = db.query_videos(conn, verdict=verdict, min_percent=min_percent)
        table = Table(title="NSFW scan report")
        table.add_column("Path", overflow="fold")
        table.add_column("Duration", justify="right")
        table.add_column("Frames", justify="right")
        table.add_column("% NSFW", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Verdict")
        table.add_column("Scanned")
        for r in rows:
            table.add_row(
                str(r["path"]),
                f"{r['duration_s']:.1f}s" if r["duration_s"] is not None else "-",
                str(r["frames_total"]) if r["frames_total"] is not None else "-",
                (
                    f"{r['nsfw_percent']:.2f}"
                    if r["nsfw_percent"] is not None
                    else "-"
                ),
                f"{r['max_score']:.3f}" if r["max_score"] is not None else "-",
                r["verdict"] or "-",
                r["scanned_at"] or "-",
            )
        console.print(table)
    finally:
        conn.close()


@app.command(name="config-show")
def config_show() -> None:
    """Print runtime configuration and resolved device."""
    import torch

    resolved = device.resolve_device("auto")
    table = Table(title="vnt configuration")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Auto device", resolved)
    table.add_row("Default threshold", str(config.DEFAULT_THRESHOLD))
    table.add_row("Default FPS", str(config.DEFAULT_FPS))
    table.add_row("Default batch size", str(config.DEFAULT_BATCH_SIZE))
    table.add_row("ViT model", config.DEFAULT_VIT_MODEL)
    table.add_row("VLM model", config.DEFAULT_VLM_MODEL)
    table.add_row("Default DB", config.DEFAULT_DB)
    console.print(table)


if __name__ == "__main__":
    app()
