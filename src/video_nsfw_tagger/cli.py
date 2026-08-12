"""Typer CLI for the video-nsfw-tagger."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from transformers import Pipeline

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from video_nsfw_tagger import (
    aggregate,
    classify,
    config,
    db,
    device,
    extract,
    sidecar,
)
from video_nsfw_tagger import (
    lexicon as lexicon_mod,
)
from video_nsfw_tagger import ollama as ollama_mod

app = typer.Typer(help="Video NSFW Tagger", no_args_is_help=True)
console = Console()


def _discover_videos(target: Path, recursive: bool) -> list[Path]:
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
        float | None,
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
        typer.Option("--vlm", "--ollama", help="Enable VLM captioning via Ollama"),
    ] = False,
    vlm_model: Annotated[
        str,
        typer.Option("--vlm-model", "--ollama-model", help="Ollama VLM model name"),
    ] = config.DEFAULT_OLLAMA_MODEL,
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama server URL"),
    ] = config.DEFAULT_OLLAMA_HOST,
    unload_vit_before_vlm: Annotated[
        bool,
        typer.Option(
            "--unload-vit-before-vlm",
            help="Unload the ViT before VLM calls (lower peak VRAM, slower)",
        ),
    ] = False,
    vlm_top_k: Annotated[
        int | None,
        typer.Option("--vlm-top-k", help="Max flagged frames to caption", min=1),
    ] = None,
    vlm_timeout: Annotated[
        float,
        typer.Option(
            "--vlm-timeout",
            help="Per-caption timeout in seconds (cold model load counts)",
            min=10.0,
        ),
    ] = ollama_mod.REQUEST_TIMEOUT_S,
    lexicon: Annotated[
        Path,
        typer.Option("--lexicon", help="Lexicon file for act tags"),
    ] = config.DEFAULT_LEXICON,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed progress for each step"),
    ] = False,
) -> None:
    """Scan video(s) and write sidecar + index entries.

    Raises:
        typer.Exit: If the VLM backend or lexicon is unavailable, no
            supported videos are found, or the ViT model fails to load.
    """

    def vlog(message: str) -> None:
        if verbose:
            stamp = datetime.now().strftime("%H:%M:%S")
            console.print(f"[dim]{stamp}[/dim] [blue]·[/blue] {message}")

    captioner = None
    lexicon_data = None
    if vlm:
        captioner = ollama_mod.OllamaCaptioner(
            model=vlm_model, host=ollama_host, timeout=vlm_timeout
        )
        vlog(f"Checking Ollama at {ollama_host} for model {vlm_model}...")
        try:
            captioner.check_available()
        except RuntimeError as exc:
            console.print(f"[red]VLM unavailable:[/red] {exc}")
            raise typer.Exit(1)
        try:
            lexicon_data = lexicon_mod.load_lexicon(lexicon)
        except Exception as exc:
            console.print(f"[red]Failed to load lexicon {lexicon}:[/red] {exc}")
            raise typer.Exit(1)
        vlog(f"Ollama OK; lexicon loaded ({len(lexicon_data)} tags from {lexicon})")

    videos = _discover_videos(target, recursive)
    if not videos:
        console.print("[yellow]No supported video files found.[/yellow]")
        raise typer.Exit(0)

    resolved = device.resolve_device(device_name)
    vlog(f"Found {len(videos)} video(s); device: {resolved}")

    def load_vit() -> "Pipeline":
        try:
            vlog(f"Loading ViT model {config.DEFAULT_VIT_MODEL}...")
            pipeline = classify.load_pipeline(
                resolved,
                model_name=config.DEFAULT_VIT_MODEL,
                batch_size=batch_size,
            )
            vlog("ViT model loaded")
            return pipeline
        except Exception as exc:
            console.print(f"[red]Failed to load ViT model:[/red] {exc}")
            raise typer.Exit(1)

    pipe = load_vit()

    conn = db.init_db(db_path)
    vlog(f"Database ready at {db_path}")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning videos...", total=len(videos))
            for video in videos:
                progress.update(task, description=f"Scanning {video.name}")
                if pipe is None:
                    pipe = load_vit()
                vlog(f"Extracting frames from {video.name} at {fps} fps...")
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
                    n_batches = -(-len(image_paths) // batch_size)
                    vlog(
                        f"Classifying {len(image_paths)} frames "
                        f"in {n_batches} batch(es) of {batch_size}..."
                    )
                    scores = classify.classify_batch(
                        pipe, image_paths, batch_size=batch_size
                    )
                    result = aggregate.aggregate(scores, threshold=threshold, fps=fps)
                    vlog(
                        f"Classification done: {len(result.flagged_frames)}/"
                        f"{len(scores)} frames flagged (max={result.max_score})"
                    )

                    # VLM stage (Phase B): caption flagged frames while they
                    # still exist inside the extracted_frames context.
                    vlm_block = None
                    act_tags: list[str] = []
                    if captioner is not None and result.flagged_frames:
                        selected = ollama_mod.select_flagged_frames(
                            frames, result.flagged_frames, top_k=vlm_top_k
                        )
                        vlog(
                            f"Selected {len(selected)} of "
                            f"{len(result.flagged_frames)} flagged frames "
                            f"for captioning"
                        )
                        if unload_vit_before_vlm:
                            vlog("Unloading ViT to free VRAM for the VLM...")
                            del pipe
                            pipe = None
                            if resolved == "cuda":
                                import torch

                                torch.cuda.empty_cache()
                        # Decision #16: degrade, don't abort the batch. A
                        # failed frame never discards captions already done.
                        captions = []
                        failures = 0
                        for i, (idx, ts, path) in enumerate(selected, 1):
                            progress.update(
                                task,
                                description=(
                                    f"Captioning {video.name} frame {i}/{len(selected)}"
                                ),
                            )
                            try:
                                caption = captioner.caption_frame(path)
                            except Exception as exc:
                                failures += 1
                                console.print(
                                    f"[yellow]Caption {i}/{len(selected)} "
                                    f"failed (frame {idx}): {exc}[/yellow]"
                                )
                                continue
                            matches = lexicon_mod.find_matches(caption, lexicon_data)
                            captions.append({
                                "frame": idx,
                                "timestamp_s": ts,
                                "caption": caption,
                                "tags": sorted(matches),
                                "matches": matches,
                            })
                            preview = caption.strip()[:80]
                            vlog(
                                f"Caption {i}/{len(selected)} done "
                                f"(frame {idx}, {ts:.0f}s): "
                                f"{preview or '(empty caption)'}..."
                            )
                            if matches:
                                formatted = "; ".join(
                                    f"{tag}: {', '.join(patterns)}"
                                    for tag, patterns in sorted(matches.items())
                                )
                                console.print(
                                    f"  [cyan]frame {idx} tags:[/cyan] {formatted}"
                                )
                        if failures:
                            console.print(
                                f"[yellow]{failures}/{len(selected)} captions "
                                f"failed; keeping the {len(captions)} that "
                                f"succeeded.[/yellow]"
                            )
                        if captions:
                            act_tags = sorted({
                                tag for c in captions for tag in c["tags"]
                            })
                            vlm_block = {
                                "backend": "ollama",
                                "model": vlm_model,
                                "host": ollama_host,
                                "captions": captions,
                                "act_tags": act_tags,
                            }
                            vlog(f"Act tags: {', '.join(act_tags) or 'none'}")
                        progress.update(task, description=f"Scanning {video.name}")

                    duration = extract.get_duration(video)
                    side = sidecar.sidecar_path(video)
                    created = datetime.now(timezone.utc).astimezone().isoformat()

                    # Decision #18: a non-VLM scan preserves existing VLM data.
                    if captioner is None and side.exists():
                        try:
                            vlm_block = sidecar.read_sidecar(side).get("vlm")
                        except Exception:
                            vlm_block = None

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
                        "vlm": vlm_block,
                        "created_at": created,
                    }
                    sidecar.write_sidecar(side, payload)
                    vlog(f"Sidecar written to {side}")

                    record = {
                        "path": str(video),
                        "duration_s": duration,
                        "frames_total": len(scores),
                        "nsfw_percent": round(result.nsfw_percent, 2),
                        "max_score": result.max_score,
                        "verdict": result.verdict,
                        "threshold": threshold,
                        "model": config.DEFAULT_VIT_MODEL,
                        "sidecar_path": str(side),
                        "scanned_at": created,
                    }
                    if vlm_block is not None:
                        record["vlm_model"] = vlm_block["model"]
                        record["act_tags"] = json.dumps(vlm_block["act_tags"])
                    db.upsert_video(conn, record)
                    console.print(
                        f"[green]Scanned[/green] {video} → {result.verdict} "
                        f"(max={result.max_score}, "
                        f"flagged={len(result.flagged_frames)})"
                    )
                    if act_tags:
                        console.print(f"  [cyan]Act tags:[/cyan] {', '.join(act_tags)}")
                progress.advance(task)
    finally:
        conn.close()
        if captioner is not None:
            vlog("Unloading Ollama model from VRAM...")
            captioner.unload()


@app.command()
def report(
    db_path: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path"),
    ] = Path(config.DEFAULT_DB),
    verdict: Annotated[
        str | None,
        typer.Option("--verdict", help="Filter by verdict"),
    ] = None,
    min_percent: Annotated[
        float | None,
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
                (f"{r['nsfw_percent']:.2f}" if r["nsfw_percent"] is not None else "-"),
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
    table.add_row("VLM model", config.DEFAULT_OLLAMA_MODEL)
    table.add_row("Ollama host", config.DEFAULT_OLLAMA_HOST)
    table.add_row("Default DB", config.DEFAULT_DB)
    console.print(table)


if __name__ == "__main__":
    app()
