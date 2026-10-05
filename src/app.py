"""Vlog Voiceover Studio - Interactive Gradio Web Application.

A modern, production-grade Web UI for multi-character text-to-speech
screenplay synthesis using Chatterbox-Turbo.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np

from .engine import VoiceoverEngine, detect_device
from .parser import ScriptItem, ScriptParser, normalize_paralinguistics
from .timeline import TimelineAssembler, TimelineTake

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vlog-studio")

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sample_script.fountain"
DEFAULT_VOICES_DIR = PROJECT_ROOT / "voices"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "studio_latest"

# Global TTS Engine singleton
_ENGINE_INSTANCE: Optional[VoiceoverEngine] = None


def get_engine(device: Optional[str] = None, nano: bool = False) -> VoiceoverEngine:
    """Retrieves or initializes the global VoiceoverEngine singleton."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        logger.info(f"Instantiating global VoiceoverEngine (device={device or 'auto'}, nano={nano})...")
        _ENGINE_INSTANCE = VoiceoverEngine(device=device, nano=nano)
    return _ENGINE_INSTANCE


def load_default_script() -> str:
    """Loads default sample script if present on disk."""
    if SAMPLE_SCRIPT_PATH.is_file():
        return SAMPLE_SCRIPT_PATH.read_text(encoding="utf-8")
    return """HOST: Hey everyone, welcome back to the channel! (chuckles)
HOST: Today we test local zero-subscription voiceovers. [pause: 1.2s]
NARRATOR: The benchmarks were about to speak for themselves.
HOST: [laugh] That sounds completely natural!
"""


def get_default_host_voice_path() -> Optional[str]:
    """Checks if voices/me.wav or voices/host.wav exists."""
    for candidate in [DEFAULT_VOICES_DIR / "me.wav", DEFAULT_VOICES_DIR / "host.wav"]:
        if candidate.is_file():
            return str(candidate)
    return ""


# Custom CSS for modern studio theme
STUDIO_CSS = """
/* Vlog Voiceover Studio Styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.studio-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #f8fafc;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #334155;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.studio-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin: 0 0 0.5rem 0;
    color: #38bdf8;
}
.studio-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}
.stat-pill {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    margin: 0.25rem 0.5rem 0.25rem 0;
    background: #0f172a;
    border: 1px solid #38bdf8;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
}
.card-notice {
    background: #1e293b;
    border-left: 4px solid #38bdf8;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 1rem 0;
    border-radius: 4px;
    font-size: 0.875rem;
    color: #cbd5e1;
}
.badge-ready {
    background: #065f46;
    color: #a7f3d0;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: bold;
}
"""


def create_app() -> gr.Blocks:
    """Builds and returns the Gradio Blocks application."""

    with gr.Blocks(title="Vlog Voiceover Studio") as demo:
        # Header banner and custom styles
        gr.HTML(
            f"""
            <style>{STUDIO_CSS}</style>
            <div class="studio-header">
                <div class="studio-title">🎙️ Vlog Voiceover Studio</div>
                <div class="studio-subtitle">
                    Automated Multi-Character Screenplay Synthesis with Chatterbox-Turbo
                    • Apple Silicon MPS & CUDA Accelerated • Sample-Accurate Timeline & Subtitles
                </div>
            </div>
            """
        )

        # Application State
        state_parsed_items = gr.State([])
        state_takes = gr.State([])
        state_output_dir = gr.State(str(DEFAULT_OUTPUT_DIR))
        state_assembler = gr.State(None)

        with gr.Tabs() as tabs:
            # =========================================================================
            # TAB 1: SCRIPT EDITOR
            # =========================================================================
            with gr.Tab("📝 1. Script Editor", id="tab_script"):
                gr.Markdown(
                    "Write or paste your script below. Supports both **Fountain format** "
                    "(screenplay cues & parentheticals) and **Colon format** (`HOST: Hello world!`). "
                    "Paralinguistic cues like `(chuckles)` or `[laugh]` and pause directives like `[pause: 1.5s]` or `(beat)` are fully supported."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        script_text_input = gr.Textbox(
                            label="Script Text",
                            value=load_default_script(),
                            lines=16,
                            placeholder="HOST: Hello world!\n(beat)\nNARRATOR: Welcome to the studio.",
                        )
                        with gr.Row():
                            btn_parse = gr.Button("🔍 Parse Script", variant="primary", scale=2)
                            btn_load_sample = gr.Button("📄 Load Sample Script", variant="secondary")
                            btn_clear_script = gr.Button("🗑️ Clear", variant="secondary")

                    with gr.Column(scale=2):
                        file_script_upload = gr.File(
                            label="Or Upload Script File (.fountain, .txt)",
                            file_types=[".fountain", ".txt"],
                        )
                        default_pause_input = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.4,
                            step=0.1,
                            label="Default Line Pause (seconds)",
                        )
                        default_beat_input = gr.Slider(
                            minimum=0.2,
                            maximum=3.0,
                            value=0.8,
                            step=0.1,
                            label="Default Beat Pause (seconds)",
                        )
                        script_stats_box = gr.Markdown(
                            "### 📊 Script Stats\n*Click 'Parse Script' to analyze character lines, word counts, and estimated duration.*"
                        )

                gr.Markdown("### 📋 Parsed Dialogue Takes")
                lines_dataframe = gr.Dataframe(
                    headers=["#", "Speaker", "Dialogue Text", "Pause After (s)"],
                    datatype=["number", "str", "str", "number"],
                    interactive=False,
                    wrap=True,
                )

            # =========================================================================
            # TAB 2: VOICES & CASTING
            # =========================================================================
            with gr.Tab("🎭 2. Voice Profiles", id="tab_voices"):
                gr.Markdown(
                    "Configure zero-shot reference voice profiles for each character. "
                    "Reference audio is encoded once and cached for high-speed multi-line synthesis."
                )

                gr.HTML(
                    """
                    <div class="card-notice">
                        <strong>💡 Reference Audio Best Practices:</strong><br>
                        • Must be <strong>greater than 5 seconds</strong> (10–15s recommended).<br>
                        • Clean, dry voice recording with no background music, room echo, or reverb.<br>
                        • Formats supported: <code>.wav</code> (preferred), <code>.mp3</code>, <code>.m4a</code>.
                    </div>
                    """
                )

                with gr.Row():
                    # HOST Character Profile
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎤 HOST Voice")
                        host_mode = gr.Radio(
                            choices=["Custom Audio Prompt", "Chatterbox Builtin"],
                            value="Custom Audio Prompt",
                            label="HOST Voice Mode",
                        )
                        host_audio_upload = gr.Audio(
                            label="Upload Reference Speech Clip (>5s clean speech)",
                            type="filepath",
                        )
                        host_path_input = gr.Textbox(
                            label="Or Reference File Path (e.g. voices/me.wav)",
                            value=get_default_host_voice_path(),
                            placeholder="voices/me.wav",
                        )

                    # NARRATOR Character Profile
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎙️ NARRATOR Voice")
                        narrator_mode = gr.Radio(
                            choices=["Chatterbox Builtin (Default)", "Custom Audio Prompt"],
                            value="Chatterbox Builtin (Default)",
                            label="NARRATOR Voice Mode",
                        )
                        narrator_audio_upload = gr.Audio(
                            label="Upload Narrator Speech Clip (>5s clean speech)",
                            type="filepath",
                            interactive=False,
                        )
                        narrator_path_input = gr.Textbox(
                            label="Or Reference File Path",
                            placeholder="voices/narrator.wav",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 👥 Additional / Custom Characters")
                        extra_voices_input = gr.Textbox(
                            label="Extra Character Voice Mappings (Format: CHARACTER=path/to/voice.wav, one per line)",
                            placeholder="GUEST=voices/guest.wav\nALEX=default",
                            lines=3,
                        )

                with gr.Row():
                    btn_warm_voices = gr.Button("⚡ Pre-Warm & Cache Voice Profiles", variant="primary")
                    voice_status_output = gr.Markdown("Status: Voice profiles ready for assignment.")

            # =========================================================================
            # TAB 3: TIMELINE & FULL GENERATION
            # =========================================================================
            with gr.Tab("⏱️ 3. Timeline & Full Generation", id="tab_timeline"):
                gr.Markdown(
                    "Synthesize all parsed script lines in sequence and concatenate into a sample-accurate master audio track and subtitles."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gen_output_dir_input = gr.Textbox(
                            label="Output Directory",
                            value=str(DEFAULT_OUTPUT_DIR),
                        )
                        gen_device_dropdown = gr.Dropdown(
                            label="Hardware Device",
                            choices=["auto", "mps", "cuda", "cpu"],
                            value="auto",
                        )
                        with gr.Row():
                            gen_temp_slider = gr.Slider(
                                minimum=0.1, maximum=1.5, value=0.8, step=0.05, label="Temperature"
                            )
                            gen_topp_slider = gr.Slider(
                                minimum=0.5, maximum=1.0, value=0.95, step=0.05, label="Top-P"
                            )
                        gen_rep_slider = gr.Slider(
                            minimum=1.0, maximum=2.0, value=1.2, step=0.05, label="Repetition Penalty"
                        )
                        btn_generate_full = gr.Button(
                            "🚀 Generate Full Voiceover", variant="primary", size="lg"
                        )
                        generation_status = gr.Markdown("Ready to synthesize.")

                    with gr.Column(scale=2):
                        gr.Markdown("### 🎧 Master Audio Player")
                        master_audio_player = gr.Audio(label="Master Concatenated Audio Track", type="filepath")

                        gr.Markdown("### 📜 Synchronized Subtitles (SRT)")
                        srt_subtitles_viewer = gr.Textbox(
                            label="Subtitles Preview",
                            lines=8,
                        )

                        with gr.Row():
                            download_master_wav = gr.File(label="Master WAV Audio")
                            download_srt = gr.File(label="SRT Subtitles")
                            download_vtt = gr.File(label="WebVTT Subtitles")
                            download_zip = gr.File(label="Full Project Archive (ZIP)")

            # =========================================================================
            # TAB 4: LINE TAKE STUDIO (REROLL)
            # =========================================================================
            with gr.Tab("🎛️ 4. Line Take Studio (Reroll)", id="tab_reroll"):
                gr.Markdown(
                    "Audition and fine-tune individual dialogue takes. Re-generate any take with "
                    "modified text or sampling parameters, and automatically re-assemble the master track."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        take_line_selector = gr.Dropdown(
                            label="Select Line to Audition / Reroll",
                            choices=[],
                            value=None,
                        )
                        take_speaker_display = gr.Textbox(label="Speaker", interactive=False)
                        take_text_input = gr.Textbox(
                            label="Dialogue Text (Edit to re-generate)",
                            lines=3,
                        )
                        take_pause_input = gr.Number(
                            label="Pause After Line (seconds)",
                            value=0.4,
                        )
                        with gr.Row():
                            reroll_temp_slider = gr.Slider(
                                minimum=0.1, maximum=1.5, value=0.8, step=0.05, label="Temperature"
                            )
                            reroll_topp_slider = gr.Slider(
                                minimum=0.5, maximum=1.0, value=0.95, step=0.05, label="Top-P"
                            )
                        btn_reroll = gr.Button("🎲 Reroll Line Take", variant="primary", size="lg")
                        reroll_status = gr.Markdown("Select a line above.")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🔊 Individual Line Take Audio")
                        single_take_audio_player = gr.Audio(label="Take Audio Player", type="filepath")

                        gr.Markdown("### 🔄 Master Timeline Sync")
                        reroll_master_player = gr.Audio(label="Updated Master Audio", type="filepath")
                        reroll_srt_viewer = gr.Textbox(label="Updated Subtitles (SRT)", lines=6)

        # =========================================================================
        # EVENT HANDLERS & LOGIC
        # =========================================================================

        # Helper to build voice map from UI inputs
        def resolve_voice_map(
            h_mode: str,
            h_audio: Optional[str],
            h_path: Optional[str],
            n_mode: str,
            n_audio: Optional[str],
            n_path: Optional[str],
            extra_lines: Optional[str],
        ) -> Dict[str, Optional[str]]:
            vmap: Dict[str, Optional[str]] = {}

            # HOST
            if h_mode == "Chatterbox Builtin":
                vmap["HOST"] = None
            else:
                if h_audio and Path(h_audio).is_file():
                    vmap["HOST"] = str(Path(h_audio).resolve())
                elif h_path and Path(h_path.strip()).is_file():
                    vmap["HOST"] = str(Path(h_path.strip()).resolve())
                else:
                    vmap["HOST"] = None

            # NARRATOR
            if n_mode.startswith("Chatterbox Builtin"):
                vmap["NARRATOR"] = None
            else:
                if n_audio and Path(n_audio).is_file():
                    vmap["NARRATOR"] = str(Path(n_audio).resolve())
                elif n_path and Path(n_path.strip()).is_file():
                    vmap["NARRATOR"] = str(Path(n_path.strip()).resolve())
                else:
                    vmap["NARRATOR"] = None

            # Extra lines
            if extra_lines:
                for line in extra_lines.splitlines():
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    char, path = line.split("=", 1)
                    char_k = char.strip().upper()
                    pval = path.strip()
                    if pval.lower() in ("default", "none", "builtin", ""):
                        vmap[char_k] = None
                    elif Path(pval).is_file():
                        vmap[char_k] = str(Path(pval).resolve())
                    else:
                        vmap[char_k] = None

            return vmap

        # 1. Parse Script Action
        def on_parse_script(script_text: str, default_pause: float, default_beat: float):
            if not script_text.strip():
                return (
                    [],
                    "### 📊 Script Stats\n*Error: Script text is empty.*",
                    [],
                    gr.Dropdown(choices=[]),
                )

            items = ScriptParser.parse(
                script_text, default_pause=default_pause, default_beat=default_beat
            )
            total_words = sum(len(it.text.split()) for it in items)
            est_duration_sec = (total_words / 150.0) * 60.0 + sum(it.pause_after_sec for it in items)
            characters = sorted(list({it.speaker for it in items}))

            # Format table data
            table_rows = []
            dropdown_choices = []
            for it in items:
                snippet = (it.text[:45] + "...") if len(it.text) > 45 else it.text
                dropdown_choices.append(f"Line {it.index}: [{it.speaker}] {snippet}")
                table_rows.append([it.index, it.speaker, it.text, round(it.pause_after_sec, 2)])

            char_badges = " ".join([f"<span class='stat-pill'>👤 {c}</span>" for c in characters])
            stats_markdown = f"""
### 📊 Script Analysis
{char_badges}

- **Total Dialogue Takes:** {len(items)}
- **Total Word Count:** {total_words} words
- **Estimated Master Duration:** {est_duration_sec / 60.0:.1f} min ({est_duration_sec:.1f}s)
            """

            return (
                [it.to_dict() for it in items],
                stats_markdown,
                table_rows,
                gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
            )

        btn_parse.click(
            fn=on_parse_script,
            inputs=[script_text_input, default_pause_input, default_beat_input],
            outputs=[state_parsed_items, script_stats_box, lines_dataframe, take_line_selector],
        )

        # Upload script file handler
        def on_upload_script_file(file_obj):
            if file_obj is None:
                return gr.update()
            file_path = Path(file_obj.name)
            try:
                content = file_path.read_text(encoding="utf-8")
                return content
            except Exception as e:
                return f"Error reading file: {e}"

        file_script_upload.change(
            fn=on_upload_script_file,
            inputs=[file_script_upload],
            outputs=[script_text_input],
        )

        # Reset / clear buttons
        btn_load_sample.click(
            fn=lambda: load_default_script(),
            inputs=[],
            outputs=[script_text_input],
        )
        btn_clear_script.click(
            fn=lambda: "",
            inputs=[],
            outputs=[script_text_input],
        )

        # Voice mode toggle interactivity
        def on_narrator_mode_change(mode):
            is_custom = "Custom" in mode
            return gr.update(interactive=is_custom), gr.update(interactive=is_custom)

        narrator_mode.change(
            fn=on_narrator_mode_change,
            inputs=[narrator_mode],
            outputs=[narrator_audio_upload, narrator_path_input],
        )

        # Pre-warm voice profiles action
        def on_prewarm_voices(
            h_mode, h_audio, h_path, n_mode, n_audio, n_path, extra_lines, device_opt
        ):
            vmap = resolve_voice_map(h_mode, h_audio, h_path, n_mode, n_audio, n_path, extra_lines)
            try:
                dev = None if device_opt == "auto" else device_opt
                engine = get_engine(device=dev)
                for char, p in vmap.items():
                    engine.register_voice(char, p)
                engine.preload_voices(list(vmap.keys()))
                configured = ", ".join([f"**{c}** ({'Default' if p is None else 'Custom'})" for c, p in vmap.items()])
                return f"✅ **Voice profiles pre-warmed successfully on {engine.device}!** Configured: {configured}"
            except Exception as e:
                logger.exception("Error during voice pre-warming")
                return f"❌ **Error pre-warming voices:** {e}"

        btn_warm_voices.click(
            fn=on_prewarm_voices,
            inputs=[
                host_mode,
                host_audio_upload,
                host_path_input,
                narrator_mode,
                narrator_audio_upload,
                narrator_path_input,
                extra_voices_input,
                gen_device_dropdown,
            ],
            outputs=[voice_status_output],
        )

        # 3. Full Generation Action
        def on_generate_full(
            parsed_items_data,
            script_raw_text,
            default_pause,
            default_beat,
            output_dir_str,
            device_opt,
            temperature,
            top_p,
            rep_penalty,
            h_mode,
            h_audio,
            h_path,
            n_mode,
            n_audio,
            n_path,
            extra_lines,
            progress=gr.Progress(track_tqdm=True),
        ):
            # Parse if state is empty
            if not parsed_items_data:
                if not script_raw_text.strip():
                    raise gr.Error("Script is empty! Please write or upload a script first.")
                items = ScriptParser.parse(
                    script_raw_text, default_pause=default_pause, default_beat=default_beat
                )
            else:
                items = [ScriptItem.from_dict(d) for d in parsed_items_data]

            if not items:
                raise gr.Error("No dialogue lines found in script.")

            out_dir = Path(output_dir_str).resolve()
            takes_dir = out_dir / "takes"
            takes_dir.mkdir(parents=True, exist_ok=True)

            # Initialize engine & register voices
            dev = None if device_opt == "auto" else device_opt
            progress(0.05, desc="Loading Chatterbox-Turbo TTS Engine...")
            engine = get_engine(device=dev)

            vmap = resolve_voice_map(h_mode, h_audio, h_path, n_mode, n_audio, n_path, extra_lines)
            all_speakers = {it.speaker for it in items}
            for spk in all_speakers:
                ref = vmap.get(spk)
                engine.register_voice(spk, ref)

            progress(0.15, desc="Pre-caching voice conditionals...")
            engine.preload_voices(list(all_speakers))

            # Assemble takes
            assembler = TimelineAssembler(output_dir=out_dir, sample_rate=engine.sample_rate)
            total_takes = len(items)

            for i, item in enumerate(items, start=1):
                pct = 0.15 + 0.70 * (i / total_takes)
                progress(pct, desc=f"Synthesizing Take {i}/{total_takes}: [{item.speaker}] {item.text[:25]}...")

                take_path = takes_dir / f"take_{item.index:03d}_{item.speaker}.wav"
                wav_tensor, sr = engine.generate_line(
                    text=item.text,
                    character=item.speaker,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                )
                engine.save_wav(wav_tensor, sr, take_path)

                take = assembler.create_take(
                    index=item.index,
                    speaker=item.speaker,
                    text=item.text,
                    audio_path=take_path,
                    sample_rate=sr,
                    pause_after_sec=item.pause_after_sec,
                )
                assembler.takes.append(take)

            progress(0.90, desc="Assembling master timeline and synchronized subtitles...")
            master_wav, srt_path, vtt_path, meta_path, duration = assembler.assemble_takes(
                takes=assembler.takes,
                default_pause_sec=default_pause,
                base_name="master",
                output_dir=out_dir,
            )

            progress(0.95, desc="Packaging project ZIP archive...")
            zip_path = assembler.export_zip()

            srt_content = srt_path.read_text(encoding="utf-8") if srt_path.is_file() else ""

            # Update Line dropdown choices
            dropdown_choices = []
            for it in items:
                snippet = (it.text[:45] + "...") if len(it.text) > 45 else it.text
                dropdown_choices.append(f"Line {it.index}: [{it.speaker}] {snippet}")

            status_msg = (
                f"✅ **Generation Complete!** Generated {total_takes} takes ({duration:.2f}s total duration) "
                f"saved to `{out_dir}`."
            )

            return (
                str(master_wav),
                srt_content,
                str(master_wav),
                str(srt_path),
                str(vtt_path),
                str(zip_path),
                str(out_dir),
                assembler,
                gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0] if dropdown_choices else None),
                status_msg,
            )

        btn_generate_full.click(
            fn=on_generate_full,
            inputs=[
                state_parsed_items,
                script_text_input,
                default_pause_input,
                default_beat_input,
                gen_output_dir_input,
                gen_device_dropdown,
                gen_temp_slider,
                gen_topp_slider,
                gen_rep_slider,
                host_mode,
                host_audio_upload,
                host_path_input,
                narrator_mode,
                narrator_audio_upload,
                narrator_path_input,
                extra_voices_input,
            ],
            outputs=[
                master_audio_player,
                srt_subtitles_viewer,
                download_master_wav,
                download_srt,
                download_vtt,
                download_zip,
                state_output_dir,
                state_assembler,
                take_line_selector,
                generation_status,
            ],
        )

        # 4. Line Selector Change Action
        def on_select_line(selected_line_str, output_dir_str, parsed_items_data):
            if not selected_line_str:
                return "", "", 0.4, None, "No line selected."

            try:
                # Format: "Line X: [SPEAKER] Text..."
                line_idx_str = selected_line_str.split(":", 1)[0].replace("Line", "").strip()
                line_idx = int(line_idx_str)
            except Exception:
                return "", "", 0.4, None, "Invalid line format."

            # Find in metadata or parsed items
            out_dir = Path(output_dir_str).resolve()
            meta_path = out_dir / "metadata.json"

            speaker = ""
            text = ""
            pause = 0.4
            audio_path = None

            if meta_path.is_file():
                try:
                    mdata = json.loads(meta_path.read_text(encoding="utf-8"))
                    for t in mdata.get("takes", []):
                        if t["index"] == line_idx:
                            speaker = t["speaker"]
                            text = t["text"]
                            pause = t.get("pause_after_sec", 0.4)
                            if t.get("audio_file") and Path(t["audio_file"]).is_file():
                                audio_path = str(Path(t["audio_file"]).resolve())
                            break
                except Exception as e:
                    logger.warning(f"Failed to read metadata.json: {e}")

            if not text and parsed_items_data:
                for d in parsed_items_data:
                    if d["index"] == line_idx:
                        speaker = d["speaker"]
                        text = d["text"]
                        pause = d.get("pause_after_sec", 0.4)
                        cand = out_dir / "takes" / f"take_{line_idx:03d}_{speaker}.wav"
                        if cand.is_file():
                            audio_path = str(cand)
                        break

            status = f"Loaded Line {line_idx} ({speaker}). Ready for playback or reroll."
            return speaker, text, pause, audio_path, status

        take_line_selector.change(
            fn=on_select_line,
            inputs=[take_line_selector, state_output_dir, state_parsed_items],
            outputs=[
                take_speaker_display,
                take_text_input,
                take_pause_input,
                single_take_audio_player,
                reroll_status,
            ],
        )

        # 5. Reroll Line Action
        def on_reroll_line(
            selected_line_str,
            edited_text,
            edited_pause,
            temperature,
            top_p,
            output_dir_str,
            device_opt,
            h_mode,
            h_audio,
            h_path,
            n_mode,
            n_audio,
            n_path,
            extra_lines,
        ):
            if not selected_line_str:
                raise gr.Error("Please select a line to reroll.")

            try:
                line_idx_str = selected_line_str.split(":", 1)[0].replace("Line", "").strip()
                line_idx = int(line_idx_str)
            except Exception:
                raise gr.Error("Invalid line format.")

            out_dir = Path(output_dir_str).resolve()
            meta_path = out_dir / "metadata.json"
            if not meta_path.is_file():
                raise gr.Error(f"No existing generation run found at `{out_dir}`. Generate the full voiceover first.")

            # Load assembler from metadata
            assembler = TimelineAssembler.load_from_metadata(meta_path, output_dir=out_dir)

            # Find target take
            target_take: Optional[TimelineTake] = None
            for t in assembler.takes:
                if t.index == line_idx:
                    target_take = t
                    break

            if target_take is None:
                raise gr.Error(f"Line {line_idx} not found in timeline metadata.")

            # Normalize text if stage directions exist
            norm_text, _ = normalize_paralinguistics(edited_text)
            speaker = target_take.speaker

            # Initialize engine & ensure voice is registered
            dev = None if device_opt == "auto" else device_opt
            engine = get_engine(device=dev)
            vmap = resolve_voice_map(h_mode, h_audio, h_path, n_mode, n_audio, n_path, extra_lines)
            engine.register_voice(speaker, vmap.get(speaker))

            # Generate new line take
            logger.info(f"Rerolling line {line_idx} ({speaker}): '{norm_text}'")
            wav_tensor, sr = engine.generate_line(
                text=norm_text,
                character=speaker,
                temperature=temperature,
                top_p=top_p,
            )

            # Target file path
            take_path = target_take.audio_path
            if not take_path:
                take_path = out_dir / "takes" / f"take_{line_idx:03d}_{speaker}.wav"

            engine.save_wav(wav_tensor, sr, take_path)

            # Update take in assembler
            target_take.text = norm_text
            target_take.audio_path = take_path
            target_take.audio_data = None  # Reload from disk during assembly
            target_take.pause_after_sec = float(edited_pause)

            # Re-assemble timeline
            master_wav, srt_path, vtt_path, m_path, new_duration = assembler.assemble_takes(
                takes=assembler.takes,
                default_pause_sec=float(edited_pause),
                base_name="master",
                output_dir=out_dir,
            )
            zip_path = assembler.export_zip()

            srt_content = srt_path.read_text(encoding="utf-8") if srt_path.is_file() else ""
            status_msg = f"🎉 **Line {line_idx} successfully re-rolled!** New master duration: {new_duration:.2f}s."

            return (
                str(take_path),
                str(master_wav),
                srt_content,
                str(master_wav),
                srt_content,
                str(master_wav),
                str(srt_path),
                str(vtt_path),
                str(zip_path),
                status_msg,
            )

        btn_reroll.click(
            fn=on_reroll_line,
            inputs=[
                take_line_selector,
                take_text_input,
                take_pause_input,
                reroll_temp_slider,
                reroll_topp_slider,
                state_output_dir,
                gen_device_dropdown,
                host_mode,
                host_audio_upload,
                host_path_input,
                narrator_mode,
                narrator_audio_upload,
                narrator_path_input,
                extra_voices_input,
            ],
            outputs=[
                single_take_audio_player,
                reroll_master_player,
                reroll_srt_viewer,
                master_audio_player,
                srt_subtitles_viewer,
                download_master_wav,
                download_srt,
                download_vtt,
                download_zip,
                reroll_status,
            ],
        )

    return demo


def main():
    """Main launcher for the Vlog Voiceover Studio Web UI."""
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
