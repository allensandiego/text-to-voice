# 🎙️ Text to Voice Studio

Automated multi-character screenplay voiceovers powered by local **Chatterbox-Turbo** neural text-to-speech.

Convert screenplay scripts directly into sample-accurate, multi-character master audio tracks with synchronized `.srt` / `.vtt` subtitles and a line-by-line take reroll workflow.

---

## ✨ Features

- **Local & Zero-Subscription**: Runs completely on-device with Apple Silicon (`mps`), NVIDIA (`cuda`), or `cpu` fallback. No recurring cloud API fees or external latency.
- **Multi-Character Casting**: Map different script characters (e.g., `HOST`, `NARRATOR`, `GUEST`) to zero-shot reference voice clips or Chatterbox-Turbo's builtin voice.
- **Cached Voice Conditioners**: Speaker and acoustic embeddings are computed once per character and cached in memory, eliminating redundant encoding overhead across lines.
- **Fountain & Colon Script Formats**: Supports standard screenplay formatting (Fountain) as well as quick colon-delimited lines (`HOST: Hello world!`).
- **Paralinguistic Emotional Cues**: Automatically maps stage directions like `(chuckles)` $\to$ `[chuckle]`, `(laughs)` $\to$ `[laugh]`, `(sighs)` $\to$ `[sigh]`, and `(whispers)` $\to$ `[whispering]`. Tags are voiced naturally by the TTS model but cleanly stripped from exported subtitles.
- **Pacing & Beat Directives**: Support for `[pause: 1.5s]`, `(pause 2.0s)`, and `(beat)` cues with sample-accurate silence insertion.
- **Line Take Studio (Reroll)**: Audition individual dialogue lines, adjust sampling temperature/top-p, edit line text, and re-roll a single take while automatically re-assembling the master audio track and subtitles.
- **Interactive Gradio Web Studio**: Modern 4-tab web application for script editing, voice casting, timeline assembly, and take auditing.
- **Complete Project Export**: Generates `master.wav`, `subtitles.srt`, `subtitles.vtt`, `metadata.json`, individual take files, and a bundled `.zip` archive.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Activate the project virtual environment (sharing dependencies with the local `chatterbox` installation):

```bash
git clone <owner>/text-to-voice.git
cd text-to-voice
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -m src.cli --help
```

---

## 🎤 Adding Your Voice (`voices/me.wav`)

Chatterbox-Turbo performs zero-shot voice cloning from a short reference audio prompt.

1. Record **10 to 15 seconds** (minimum >5.0 seconds) of natural, clear speech.
2. Ensure the recording is **dry and clean**: no background music, room reverb, fan noise, or echo.
3. Save or copy your audio file to `voices/me.wav`:
   ```bash
   cp /path/to/my_recording.wav voices/me.wav
   ```
4. **Voice Profiles**:
   - `voices/me.wav` (or any custom `.wav` path): Cloned custom voice.
   - `default` or `builtin` (or left blank for `NARRATOR`): Chatterbox-Turbo builtin default voice from `conds.pt`.

---

## 📝 Writing Scripts

Scripts can be written in either **Fountain Screenplay format** or simple **Colon style**.

### 1. Fountain Format (`.fountain`, `.txt`)
```fountain
Title: My Tech Audio Episode 1
Author: Studio Creator

INT. HOME STUDIO - DAY

HOST
Hey everyone, welcome back to the channel!
(chuckles)
Today we are testing zero-subscription local neural voiceovers.

(beat)

NARRATOR
He had spent hours trying cloud voiceovers, but local synthesis was instant.

HOST
Look at how smooth this transition sounds! [pause: 1.2s]
(laughs)
It even catches subtle vocal inflections and breathing pauses.
```

### 2. Colon Style Format
```text
HOST: Hey everyone, welcome back to the channel! (chuckles)
(beat)
NARRATOR: The benchmarks were about to speak for themselves.
HOST: Look at how smooth this transition sounds! [pause: 1.5s]
HOST: [laugh] That was completely natural!
```

### 3. Paralinguistic Tags & Stage Directions
Parentheticals and bracket cues are automatically mapped to Chatterbox-Turbo tags:
- `(laughs)`, `(laughing)`, `(haha)` $\to$ `[laugh]`
- `(chuckles)`, `(snickers)` $\to$ `[chuckle]`
- `(sighs)`, `(sighing)` $\to$ `[sigh]`
- `(gasps)` $\to$ `[gasp]`
- `(coughs)` $\to$ `[cough]`
- `(groans)`, `(grunts)` $\to$ `[groan]`
- `(whispers)`, `(whispering)` $\to$ `[whispering]`
- `(clears throat)` $\to$ `[clear throat]`
- `[sniff]`, `[crying]`, `[angry]`, `[surprised]`, `[dramatic]`

*Note: All TTS paralinguistic tags are voiced naturally by the TTS model, but automatically stripped from the generated `.srt` and `.vtt` files so viewers see clean, professional subtitles.*

### 4. Pause & Beat Directives
- `[pause: 1.5s]` or `(pause 2.0s)`: Inserts exact silence duration after the preceding segment.
- `(beat)` or `[beat]`: Inserts a dramatic beat pause (default: 0.8s).
- Default line pause between spoken lines: 0.4s (customizable).

---

## 🌐 Web UI Studio Guide

Launch the interactive Gradio Web UI studio with the convenience script:

```bash
python run_studio.py
```

Open your browser to:
**`http://localhost:7860`**

### Studio Workflow:
1. **📝 1. Script Editor**:
   - Paste or upload your `.fountain` / `.txt` script.
   - Click **"🔍 Parse Script"** to inspect detected characters, word counts, estimated master duration, and the line-by-line dialogue table.
2. **🎭 2. Voice Profiles**:
   - Assign voice reference files for `HOST` (e.g. `voices/me.wav` or upload directly).
   - Choose between Chatterbox builtin default voice or a custom clip for `NARRATOR`.
   - Add extra character mappings (e.g. `GUEST=voices/guest.wav`).
   - Click **"⚡ Pre-Warm & Cache Voice Profiles"** to compute conditioning tensors in advance.
3. **⏱️ 3. Timeline & Full Generation**:
   - Adjust sampling settings (Temperature, Top-P, Repetition Penalty).
   - Click **"🚀 Generate Full Voiceover"**.
   - Listen to the concatenated master track in the waveform player.
   - Preview synchronized SRT subtitles.
   - Download `master.wav`, `subtitles.srt`, `subtitles.vtt`, or the complete `.zip` project archive.
4. **🎛️ 4. Line Take Studio (Reroll)**:
   - Select any line number from the dropdown to audition that individual take.
   - Edit the line's dialogue text or tweak temperature/top-p.
   - Click **"🎲 Reroll Line Take"**: the engine re-synthesizes the line, updates the take file, automatically re-assembles the master audio track and subtitles, and refreshes the players.

---

## 💻 CLI Commands

The CLI allows complete headless scripting and automation in production pipelines.

### 1. `parse` - Inspect Script & Dialogue
```bash
python -m src.cli parse scripts/sample_script.fountain
```
*Options:*
- `--json`: Output structured JSON with all line items and character stats.
- `--pause <sec>`: Default inter-line pause (default: 0.4s).
- `--beat <sec>`: Default beat pause (default: 0.8s).

### 2. `generate` - Full Voiceover Synthesis
```bash
python -m src.cli generate scripts/sample_script.fountain \
  --voice HOST=voices/me.wav \
  --voice NARRATOR=default \
  -o output/audio_ep1
```
*Options:*
- `-o, --output-dir <dir>`: Target directory for master and take files.
- `-v, --voice <CHAR=path>`: Character voice mapping. Can be specified multiple times.
- `--device <mps|cuda|cpu>`: Hardware device override (default: auto).
- `--temperature <float>`: Sampling temperature (default: 0.8).
- `--top-p <float>`: Top-P nucleus sampling threshold (default: 0.95).
- `--repetition-penalty <float>`: Repetition penalty (default: 1.2).

### 3. `reroll` - Re-synthesize a Specific Take
Re-synthesize a single line take (e.g., line 3) without re-generating the entire script:

```bash
python -m src.cli reroll \
  --output-dir output/audio_ep1 \
  --line 3 \
  --text "He tried again with a slightly different wording." \
  --temperature 0.75
```
Automatically overwrites the take `.wav` file, updates metadata, and rebuilds `master.wav` and subtitle files!

### 4. `assemble` - Re-assemble Timeline Takes
Rebuild `master.wav` and subtitles from existing take audio files (e.g. to test different pause lengths without re-synthesizing):

```bash
python -m src.cli assemble \
  --output-dir output/audio_ep1 \
  --pause 0.6
```

---

## 📁 Project Structure

```text
text-to-voice/
├── README.md               # Complete documentation
├── requirements.txt        # Python package dependencies
├── run_studio.py           # Gradio Web UI launcher
├── scripts/
│   ├── sample_script.fountain  # Sample screenplay script
│   └── test_mini.fountain      # Mini test script
├── src/
│   ├── __init__.py         # Package exports
│   ├── app.py              # Interactive Gradio Web UI Studio
│   ├── cli.py              # Command-line interface
│   ├── engine.py           # Chatterbox-Turbo TTS engine & conditioning cache
│   ├── parser.py           # Fountain/Colon script parser & paralinguistics
│   └── timeline.py         # Sample-accurate concatenation & SRT/VTT export
├── voices/
│   ├── README.md           # Voice reference audio guide
│   └── me.wav              # (Optional) User reference speech clip (>5s)
└── output/
    └── studio_latest/      # Generated master tracks, subtitles & takes
        ├── master.wav
        ├── subtitles.srt
        ├── subtitles.vtt
        ├── metadata.json
        ├── takes/
        │   ├── take_001_HOST.wav
        │   └── take_002_NARRATOR.wav
        └── studio_latest_export.zip
```

---

## 📄 License & Attribution

Powered by [Chatterbox-Turbo](https://github.com/allen/chatterbox). Built for local neural speech generation.
