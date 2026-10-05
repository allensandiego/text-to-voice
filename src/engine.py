"""Chatterbox-Turbo voiceover synthesis engine with cached conditionals and voice profiles.

Wraps ChatterboxTurboTTS with:
- Automatic hardware device detection (Apple Silicon MPS -> CUDA -> CPU fallback)
- Character voice profile mapping (reference audio path vs builtin default)
- Per-character Conditionals caching for fast multi-line synthesis
- Line generation with customizable sampling parameters
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals, S3GEN_SR
except ImportError as e:
    raise ImportError(
        f"Failed to import Chatterbox-Turbo: {e}. "
        "Ensure chatterbox is installed and virtualenv is active."
    )

logger = logging.getLogger(__name__)


def detect_device(preferred: Optional[str] = None) -> str:
    """Detects best available compute device, honoring preferred device if valid."""
    if preferred:
        pref = preferred.strip().lower()
        if pref == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available. Falling back to auto-detection.")
        elif pref == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
            logger.warning("MPS requested but not available. Falling back to auto-detection.")
        elif pref == "cpu":
            return "cpu"
        else:
            logger.warning(f"Unknown device '{preferred}'. Falling back to auto-detection.")

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


class VoiceoverEngine:
    """Manages Chatterbox-Turbo TTS model instance, character profiles, and cached conditionals."""

    def __init__(
        self,
        device: Optional[str] = None,
        model: Optional[ChatterboxTurboTTS] = None,
        nano: bool = False,
    ):
        """Initializes the voiceover engine.

        Args:
            device: Compute device ('mps', 'cuda', 'cpu') or None for auto-detection.
            model: Optional pre-instantiated ChatterboxTurboTTS instance.
            nano: If True, uses chatterbox-nano model instead of turbo.
        """
        self.device = detect_device(device)
        logger.info(f"Initializing VoiceoverEngine on device: {self.device}")

        if model is not None:
            self.tts = model
        else:
            self.tts = ChatterboxTurboTTS.from_pretrained(device=self.device, nano=nano)

        self.sample_rate = self.tts.sr or S3GEN_SR

        # Default builtin voice conditionals from conds.pt (e.g. for Narrator)
        self.default_conds: Optional[Conditionals] = self.tts.conds

        # Map of character_name -> audio_prompt_path
        self._voice_profiles: Dict[str, Optional[Path]] = {}

        # Cached Conditionals per character
        self._conditionals_cache: Dict[str, Conditionals] = {}

    def register_voice(
        self,
        character: str,
        voice_path: Optional[Union[str, Path]] = None,
        audio_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Registers reference audio for a character.

        Args:
            character: Character name (e.g. 'HOST', 'NARRATOR', 'ALEX')
            voice_path: Path to reference audio WAV (>5s), or None/'default' for builtin voice.
            audio_path: Alias for voice_path for backwards compatibility.
        """
        chosen_path = voice_path if voice_path is not None else audio_path
        char_key = character.strip().upper()
        if not chosen_path or str(chosen_path).strip().lower() in ("none", "default", "builtin"):
            self._voice_profiles[char_key] = None
            if self.default_conds is not None:
                self._conditionals_cache[char_key] = self.default_conds
            logger.info(f"Registered character '{char_key}' with default builtin voice.")
            return

        path = Path(chosen_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Reference voice audio not found for '{char_key}': {path}")

        self._voice_profiles[char_key] = path
        # Invalidate any previously cached conditionals for this character if path changed
        if char_key in self._conditionals_cache:
            del self._conditionals_cache[char_key]

        logger.info(f"Registered character '{char_key}' with voice clip: {path}")

    def register_voices_from_dict(self, voice_map: Dict[str, Optional[Union[str, Path]]]) -> None:
        """Registers multiple voice profiles at once."""
        for char, path in voice_map.items():
            self.register_voice(char, path)

    def get_conditionals(self, character: str, exaggeration: float = 0.5) -> Conditionals:
        """Retrieves or prepares cached Conditionals for a character.

        Caches the Conditionals object so that preparing reference features is only
        computed once per character rather than on every spoken line.
        """
        char_key = character.strip().upper()

        # Return from cache if already prepared
        if char_key in self._conditionals_cache:
            return self._conditionals_cache[char_key]

        ref_path = self._voice_profiles.get(char_key)

        if ref_path is not None:
            # Custom reference audio file provided
            logger.info(f"Computing conditionals for character '{char_key}' from: {ref_path}")
            self.tts.prepare_conditionals(
                wav_fpath=str(ref_path),
                exaggeration=exaggeration,
                norm_loudness=True,
            )
            conds = self.tts.conds
            if conds is None:
                raise RuntimeError(f"Failed to prepare conditionals for character '{char_key}'")
            self._conditionals_cache[char_key] = conds
            return conds
        else:
            # Fallback to builtin default voice (conds.pt)
            if self.default_conds is not None:
                logger.info(f"Using default builtin voice conditionals for character '{char_key}'")
                self._conditionals_cache[char_key] = self.default_conds
                return self.default_conds
            elif self.tts.conds is not None:
                self._conditionals_cache[char_key] = self.tts.conds
                return self.tts.conds
            else:
                raise ValueError(
                    f"No voice registered for character '{char_key}' and no default conds.pt available."
                )

    def preload_voices(self, characters: Optional[List[str]] = None) -> None:
        """Precomputes conditionals for all registered voices or specified character list."""
        chars_to_load = characters or list(self._voice_profiles.keys())
        for char in chars_to_load:
            self.get_conditionals(char)

    def generate_line(
        self,
        text: str,
        character: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        top_k: int = 1000,
    ) -> Tuple[torch.Tensor, int]:
        """Synthesizes a single dialogue line for a character.

        Args:
            text: Dialogue text (may include paralinguistic tags like [laugh], [chuckle]).
            character: Character name (e.g. 'HOST', 'NARRATOR').
            temperature: Sampling temperature (default: 0.8).
            top_p: Nucleus sampling threshold (default: 0.95).
            repetition_penalty: Repetition penalty (default: 1.2).
            top_k: Top-k sampling filter (default: 1000).

        Returns:
            Tuple of (audio_tensor, sample_rate) where audio_tensor has shape (1, num_samples).
        """
        conds = self.get_conditionals(character)
        # Point the model's active conditionals to this character's cached conditionals
        self.tts.conds = conds

        # Generate audio using the cached conditionals (passing no audio_prompt_path)
        wav = self.tts.generate(
            text=text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
        )

        return wav, self.sample_rate

    def generate_line_numpy(
        self,
        text: str,
        character: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        top_k: int = 1000,
    ) -> Tuple[np.ndarray, int]:
        """Synthesizes a line and returns a 1D float32 numpy array and sample rate."""
        wav_tensor, sr = self.generate_line(
            text=text,
            character=character,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
        )
        wav_np = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
        return wav_np, sr

    @staticmethod
    def save_audio(
        audio: Union[torch.Tensor, np.ndarray],
        output_path: Union[str, Path],
        sample_rate: int = 24000,
    ) -> Path:
        """Saves audio tensor or numpy array to a WAV file."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio, torch.Tensor):
            data = audio.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            data = np.asarray(audio, dtype=np.float32).squeeze()

        sf.write(str(out), data, sample_rate, format="WAV", subtype="PCM_16")
        return out

    @staticmethod
    def save_wav(
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate_or_path: Union[int, str, Path],
        out_path_or_sr: Optional[Union[str, Path, int]] = None,
    ) -> Path:
        """Saves audio tensor or numpy array to a WAV file.

        Supports both:
        - save_wav(audio, sample_rate, out_path)
        - save_wav(audio, out_path, sample_rate)
        """
        if isinstance(sample_rate_or_path, int):
            sr = sample_rate_or_path
            out = out_path_or_sr
        else:
            out = sample_rate_or_path
            sr = out_path_or_sr if isinstance(out_path_or_sr, int) else 24000
        if out is None:
            raise ValueError("Output path must be provided.")
        return VoiceoverEngine.save_audio(audio, out, sr)


if __name__ == "__main__":
    print("Testing VoiceoverEngine initialization...")
    engine = VoiceoverEngine()
    print(f"Engine device: {engine.device}")
    print(f"Sample rate: {engine.sample_rate}")
    print(f"Default conds available: {engine.default_conds is not None}")

    # Register voices
    engine.register_voice("NARRATOR", None)  # builtin
    engine.register_voice("HOST", "default")  # builtin fallback

    # Generate a short test line
    test_line = "Welcome back to the channel. [chuckle] This test is working great."
    print(f"Generating test line with character 'HOST': '{test_line}'")
    wav_tensor, sr = engine.generate_line(test_line, "HOST")
    print(f"Generated wav shape: {wav_tensor.shape}, sr: {sr}")

    test_out = Path("output/test_engine.wav")
    engine.save_audio(wav_tensor, test_out, sr)
    print(f"Saved test output to: {test_out} (file size: {test_out.stat().st_size} bytes)")
    print("VoiceoverEngine verification complete!")
