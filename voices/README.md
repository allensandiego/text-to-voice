# Reference Voices Directory

This directory stores audio reference clips used for zero-shot voice cloning in `text-to-voice`.

## Reference Audio Guidelines

Chatterbox-Turbo clones voices using a reference audio prompt. For optimal voice quality and natural inflections, follow these guidelines:

1. **Duration**: 
   - **Minimum**: Greater than 5.0 seconds (hard requirement by Chatterbox-Turbo).
   - **Recommended**: **10 to 15 seconds** of continuous, natural speech.
2. **Audio Quality**:
   - Clean, dry vocal recording (no reverb, room echo, background music, or background noise).
   - Clear articulation with natural pacing and consistent volume.
3. **Audio Format**:
   - Uncompressed `.wav` (preferred, 16-bit or 24-bit PCM).
   - Any sample rate is supported (the engine automatically resamples reference clips to 24 kHz and 16 kHz during embedding computation).

## How Voice Profiles Work

- **Custom Voices**: Place your reference voice clip here (e.g. `voices/me.wav` or `voices/host.wav`).
- **Default / Fallback Voice**: If no voice clip is specified for a character (or if you pass `default` or `none`), `text-to-voice` automatically uses Chatterbox-Turbo's builtin voice from `conds.pt`.
- **Conditionals Caching**: The engine computes acoustic and speaker conditioning once per character and caches it in memory. Multi-line generation across character changes is fast with zero re-encoding overhead.

## Usage in CLI

Map characters from your script to voice files using the `--voice` argument:

```bash
# Custom reference clip for HOST, builtin default voice for NARRATOR
python -m src.cli generate scripts/sample_script.fountain \
  --voice HOST=voices/me.wav \
  --voice NARRATOR=default \
  -o output/sample_ep1
```

If a character appears in the script but is omitted from `--voice`, it automatically falls back to the builtin default voice.
