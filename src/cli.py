#!/usr/bin/env python3
"""Command-line interface for vlog-voiceover.

Subcommands:
  parse     - Inspect and parse a script into dialogue lines, characters, and cues.
  generate  - Run full voiceover synthesis and timeline assembly.
  reroll    - Re-synthesize a specific line number and update the timeline.
  assemble  - Re-assemble master audio and subtitles from existing takes.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.markup import escape
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False
    escape = lambda x: x

from .parser import ScriptItem, parse_script_file, normalize_paralinguistics
from .engine import VoiceoverEngine, detect_device
from .timeline import TimelineAssembler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("text-to-voice")


def parse_voice_args(voice_args: Optional[List[str]]) -> Dict[str, Optional[str]]:
    """Parses list of 'CHARACTER=path' strings into a dictionary."""
    mapping: Dict[str, Optional[str]] = {}
    if not voice_args:
        return mapping
    for arg in voice_args:
        if "=" in arg:
            char, path = arg.split("=", 1)
            char_key = char.strip().upper()
            val = path.strip()
            if val.lower() in ("default", "none", "builtin", ""):
                mapping[char_key] = None
            else:
                mapping[char_key] = val
        else:
            mapping[arg.strip().upper()] = None
    return mapping


def cmd_parse(args: argparse.Namespace) -> int:
    """Handles the `parse` command."""
    script_path = Path(args.script)
    if not script_path.is_file():
        print(f"Error: Script file not found: {script_path}", file=sys.stderr)
        return 1

    items = parse_script_file(str(script_path), default_pause=args.pause, default_beat=args.beat)
    total_words = sum(len(it.text.split()) for it in items)
    est_duration_sec = (total_words / 150.0) * 60.0 + sum(it.pause_after_sec for it in items)

    # Character breakdown
    char_stats: Dict[str, Dict[str, int]] = {}
    for it in items:
        spk = it.speaker
        if spk not in char_stats:
            char_stats[spk] = {"lines": 0, "words": 0}
        char_stats[spk]["lines"] += 1
        char_stats[spk]["words"] += len(it.text.split())

    if args.json:
        data = {
            "script": str(script_path),
            "total_lines": len(items),
            "total_words": total_words,
            "est_duration_sec": round(est_duration_sec, 2),
            "characters": char_stats,
            "items": [it.to_dict() for it in items],
        }
        print(json.dumps(data, indent=2))
        return 0

    if HAS_RICH and console:
        console.print(Panel.fit(
            f"[bold cyan]Script Analysis:[/] {script_path.name}\n"
            f"[bold]Total Lines:[/] {len(items)}  |  "
            f"[bold]Total Words:[/] {total_words}  |  "
            f"[bold]Est. Duration:[/] {est_duration_sec / 60.0:.1f} min ({est_duration_sec:.0f}s)",
            title="text-to-voice parser"
        ))

        # Characters table
        char_table = Table(title="Detected Characters")
        char_table.add_column("Character", style="bold magenta")
        char_table.add_column("Lines", justify="right")
        char_table.add_column("Words", justify="right")
        for spk, stats in sorted(char_stats.items()):
            char_table.add_row(spk, str(stats["lines"]), str(stats["words"]))
        console.print(char_table)

        # Lines table
        line_table = Table(title="Parsed Dialogue Lines")
        line_table.add_column("#", justify="right", style="cyan", width=4)
        line_table.add_column("Speaker", style="magenta", width=12)
        line_table.add_column("Dialogue Text", style="white")
        line_table.add_column("Pause", justify="right", style="green", width=7)

        for it in items:
            line_table.add_row(
                str(it.index),
                escape(it.speaker),
                escape(it.text),
                f"{it.pause_after_sec:.1f}s",
            )
        console.print(line_table)
    else:
        print(f"=== Script: {script_path.name} ===")
        print(f"Total Lines: {len(items)} | Total Words: {total_words} | Est Duration: {est_duration_sec:.0f}s")
        print("\nCharacters:")
        for spk, stats in sorted(char_stats.items()):
            print(f"  - {spk}: {stats['lines']} lines, {stats['words']} words")
        print("\nDialogue:")
        for it in items:
            print(f"[{it.index:03d}] {it.speaker} ({it.pause_after_sec:.1f}s): {it.text}")

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Handles the `generate` command."""
    script_path = Path(args.script)
    if not script_path.is_file():
        print(f"Error: Script file not found: {script_path}", file=sys.stderr)
        return 1

    items = parse_script_file(str(script_path), default_pause=args.pause, default_beat=args.beat)
    if not items:
        print("Error: No dialogue items found in script.", file=sys.stderr)
        return 1

    # Setup output directory
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"output/run_{timestamp}").resolve()

    takes_dir = out_dir / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)

    # Save parsed script
    script_json_path = out_dir / "script_parsed.json"
    script_json_path.write_text(
        json.dumps([it.to_dict() for it in items], indent=2), encoding="utf-8"
    )

    # Parse voice mappings
    voice_map = parse_voice_args(args.voice)
    all_speakers = {it.speaker for it in items}

    print(f"\n[vlog-voiceover] Target output directory: {out_dir}")
    print(f"[vlog-voiceover] Total dialogue takes to synthesize: {len(items)}")

    # Initialize Engine
    print(f"[vlog-voiceover] Initializing TTS engine (device: {args.device or 'auto'})...")
    start_init = time.time()
    engine = VoiceoverEngine(device=args.device, nano=args.nano)
    print(f"[vlog-voiceover] Engine initialized on '{engine.device}' in {time.time() - start_init:.2f}s")

    # Register voices
    for spk in all_speakers:
        ref_path = voice_map.get(spk)
        engine.register_voice(spk, ref_path)

    # Pre-warm conditionals for all characters
    print("[vlog-voiceover] Pre-caching voice conditionals...")
    engine.preload_voices(list(all_speakers))
    print("[vlog-voiceover] Voice conditioning cached. Ready for synthesis.")

    # Timeline assembler
    assembler = TimelineAssembler(sample_rate=engine.sample_rate)

    # Synthesize takes
    start_gen = time.time()
    for idx, item in enumerate(items, start=1):
        take_filename = f"take_{item.index:03d}_{item.speaker}.wav"
        take_path = takes_dir / take_filename

        print(f"[{idx}/{len(items)}] Synthesizing take {item.index:03d} ({item.speaker}): \"{item.text}\"")
        wav_tensor, sr = engine.generate_line(
            text=item.text,
            character=item.speaker,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        engine.save_audio(wav_tensor, take_path, sr)

        assembler.add_take(
            index=item.index,
            speaker=item.speaker,
            text=item.text,
            audio=take_path,
            pause_after_sec=item.pause_after_sec,
            sample_rate=sr,
        )

    gen_duration = time.time() - start_gen
    print(f"\n[vlog-voiceover] All takes synthesized in {gen_duration:.2f}s!")

    # Assemble timeline
    print("[vlog-voiceover] Assembling master audio, subtitles, and metadata...")
    assembly_res = assembler.assemble(out_dir)

    print("\n" + "=" * 60)
    print("SUCCESS: Generation Complete!")
    print(f"Master Audio:    {assembly_res['master_audio_path']}")
    print(f"Subtitles (SRT): {assembly_res['srt_path']}")
    print(f"Subtitles (VTT): {assembly_res['vtt_path']}")
    print(f"Metadata JSON:   {assembly_res['metadata_path']}")
    print(f"Total Duration:  {assembly_res['total_duration_sec']:.2f} seconds")
    print("=" * 60 + "\n")

    return 0


def cmd_reroll(args: argparse.Namespace) -> int:
    """Handles the `reroll` command to re-synthesize a specific line."""
    out_dir = Path(args.output_dir).resolve()
    meta_path = out_dir / "metadata.json"
    if not meta_path.is_file():
        print(f"Error: metadata.json not found in output directory: {out_dir}", file=sys.stderr)
        return 1

    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    takes = meta_data.get("takes", [])

    target_take = None
    target_idx = None
    for i, t in enumerate(takes):
        if t["index"] == args.line:
            target_take = t
            target_idx = i
            break

    if target_take is None:
        print(f"Error: Line index {args.line} not found in metadata.", file=sys.stderr)
        return 1

    speaker = target_take["speaker"]
    take_text = args.text if args.text else target_take["text"]

    if args.text:
        # Normalize any stage directions in the new text
        norm_text, _ = normalize_paralinguistics(args.text)
        take_text = norm_text

    print(f"[vlog-voiceover] Re-rolling Line {args.line} ({speaker}): \"{take_text}\"")

    # Initialize Engine
    engine = VoiceoverEngine(device=args.device)

    # Check voice override
    voice_map = parse_voice_args(args.voice)
    ref_path = voice_map.get(speaker)
    engine.register_voice(speaker, ref_path)

    # Synthesize
    wav_tensor, sr = engine.generate_line(
        text=take_text,
        character=speaker,
        temperature=args.temperature or 0.8,
        top_p=args.top_p or 0.95,
    )

    take_file = Path(target_take["audio_file"])
    engine.save_audio(wav_tensor, take_file, sr)
    print(f"[vlog-voiceover] Overwrote take audio: {take_file}")

    # Update metadata record text
    target_take["text"] = take_text

    # Re-assemble timeline
    print("[vlog-voiceover] Re-assembling timeline...")
    assembler = TimelineAssembler(sample_rate=engine.sample_rate)
    for t in takes:
        assembler.add_take(
            index=t["index"],
            speaker=t["speaker"],
            text=t["text"],
            audio=t["audio_file"],
            pause_after_sec=t["pause_after_sec"],
        )

    res = assembler.assemble(out_dir)
    print(f"\nSUCCESS: Line {args.line} re-rolled! New master duration: {res['total_duration_sec']:.2f}s")
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    """Handles the `assemble` command to re-assemble takes without re-synthesizing."""
    out_dir = Path(args.output_dir).resolve()
    meta_path = out_dir / "metadata.json"
    if not meta_path.is_file():
        print(f"Error: metadata.json not found in output directory: {out_dir}", file=sys.stderr)
        return 1

    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    takes = meta_data.get("takes", [])
    if not takes:
        print("Error: No takes found in metadata.json", file=sys.stderr)
        return 1

    sr = meta_data.get("sample_rate", 24000)
    assembler = TimelineAssembler(sample_rate=sr)

    for t in takes:
        pause = args.pause if args.pause is not None else t.get("pause_after_sec", 0.4)
        assembler.add_take(
            index=t["index"],
            speaker=t["speaker"],
            text=t["text"],
            audio=t["audio_file"],
            pause_after_sec=pause,
        )

    res = assembler.assemble(out_dir)
    print(f"\nSUCCESS: Re-assembled master audio ({res['total_duration_sec']:.2f}s) and subtitles.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="text-to-voice",
        description="text-to-voice CLI: Text-to-speech voiceovers with Chatterbox-Turbo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. parse
    p_parse = subparsers.add_parser("parse", help="Parse and inspect script dialogue and cues.")
    p_parse.add_argument("script", help="Path to screenplay/script file (.fountain, .txt)")
    p_parse.add_argument("--pause", type=float, default=0.4, help="Default pause after dialogue (seconds)")
    p_parse.add_argument("--beat", type=float, default=0.8, help="Default beat duration (seconds)")
    p_parse.add_argument("--json", action="store_true", help="Output analysis in JSON format")

    # 2. generate
    p_gen = subparsers.add_parser("generate", help="Synthesize full voiceover from script.")
    p_gen.add_argument("script", help="Path to screenplay/script file (.fountain, .txt)")
    p_gen.add_argument("-o", "--output-dir", help="Directory to save takes and master outputs")
    p_gen.add_argument("-v", "--voice", action="append", help="Voice mapping, e.g. --voice HOST=voices/me.wav")
    p_gen.add_argument("--device", choices=["mps", "cuda", "cpu"], help="Hardware device override")
    p_gen.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    p_gen.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p (default: 0.95)")
    p_gen.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2)")
    p_gen.add_argument("--pause", type=float, default=0.4, help="Default pause between lines (seconds)")
    p_gen.add_argument("--beat", type=float, default=0.8, help="Default beat duration (seconds)")
    p_gen.add_argument("--nano", action="store_true", help="Use Chatterbox Nano model")

    # 3. reroll
    p_reroll = subparsers.add_parser("reroll", help="Re-synthesize a specific line take.")
    p_reroll.add_argument("--output-dir", required=True, help="Existing run output directory")
    p_reroll.add_argument("--line", type=int, required=True, help="Line number/index to re-roll")
    p_reroll.add_argument("--text", help="Optional replacement text for this line")
    p_reroll.add_argument("-v", "--voice", action="append", help="Voice mapping override")
    p_reroll.add_argument("--device", choices=["mps", "cuda", "cpu"], help="Hardware device override")
    p_reroll.add_argument("--temperature", type=float, help="Sampling temperature")
    p_reroll.add_argument("--top-p", type=float, help="Sampling top-p")

    # 4. assemble
    p_assemble = subparsers.add_parser("assemble", help="Re-assemble existing takes into master track.")
    p_assemble.add_argument("--output-dir", required=True, help="Existing run output directory")
    p_assemble.add_argument("--pause", type=float, help="Override pause duration between takes (seconds)")

    args = parser.parse_args()

    if args.command == "parse":
        sys.exit(cmd_parse(args))
    elif args.command == "generate":
        sys.exit(cmd_generate(args))
    elif args.command == "reroll":
        sys.exit(cmd_reroll(args))
    elif args.command == "assemble":
        sys.exit(cmd_assemble(args))


if __name__ == "__main__":
    main()
