"""Timeline assembly and subtitle synchronization module.

Features:
- Sample-accurate concatenation of dialogue takes with inserted pauses.
- Master audio (.wav) export.
- Synchronized .srt and .vtt subtitle file generation with accurate millisecond timestamps.
- metadata.json generation documenting all takes, boundaries, and timeline stats.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Pattern to clean bracketed TTS paralinguistic tags for subtitle display
TAG_CLEAN_PATTERN = re.compile(r'\[[a-zA-Z0-9_\-\s]+\]')


@dataclass
class TimelineTake:
    """Represents a generated take ready for timeline assembly."""
    index: int
    speaker: str
    text: str
    audio_path: Optional[Path] = None
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 24000
    duration_sec: float = 0.0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    pause_after_sec: float = 0.4
    metadata: Dict[str, any] = field(default_factory=dict)


def clean_text_for_subtitles(text: str) -> str:
    """Removes TTS tags like [laugh], [chuckle], [sigh] for clean viewer-facing subtitles."""
    cleaned = TAG_CLEAN_PATTERN.sub('', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def format_srt_timestamp(seconds: float) -> str:
    """Formats seconds into standard SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    if secs >= 60:
        minutes += 1
        secs -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Formats seconds into standard WebVTT timestamp: HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    if secs >= 60:
        minutes += 1
        secs -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class TimelineAssembler:
    """Assembles generated audio takes, creates master audio, subtitles, and metadata."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path, int]] = None,
        sample_rate: int = 24000,
        initial_silence_sec: float = 0.0,
    ):
        if isinstance(output_dir, int):
            self.sample_rate = output_dir
            self.output_dir: Optional[Path] = None
        else:
            self.sample_rate = sample_rate
            self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.initial_silence_sec = initial_silence_sec
        self.takes: List[TimelineTake] = []

    def create_take(
        self,
        index: int,
        speaker: str,
        text: str,
        audio_path: Optional[Union[str, Path]] = None,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 24000,
        duration_sec: Optional[float] = None,
        pause_after_sec: float = 0.4,
        metadata: Optional[Dict[str, any]] = None,
    ) -> TimelineTake:
        """Creates and returns a TimelineTake instance."""
        p_path: Optional[Path] = None
        data: Optional[np.ndarray] = audio_data
        sr = sample_rate or self.sample_rate

        if audio_path is not None:
            p_path = Path(audio_path).resolve()
            if p_path.is_file() and data is None:
                d, file_sr = sf.read(str(p_path), dtype="float32")
                data = d
                sr = file_sr

        dur = duration_sec
        if data is not None:
            if data.ndim > 1:
                data = data.squeeze()
                if data.ndim > 1:
                    data = np.mean(data, axis=-1)
            dur = len(data) / sr
        elif dur is None:
            dur = 0.0

        return TimelineTake(
            index=index,
            speaker=speaker,
            text=text,
            audio_path=p_path,
            audio_data=data,
            sample_rate=sr,
            duration_sec=dur,
            pause_after_sec=max(0.0, pause_after_sec),
            metadata=metadata or {},
        )

    def add_take(
        self,
        index: int,
        speaker: str,
        text: str,
        audio: Union[Path, str, np.ndarray],
        pause_after_sec: float = 0.4,
        sample_rate: Optional[int] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> TimelineTake:
        """Adds a take to the timeline."""
        sr = sample_rate or self.sample_rate
        audio_path: Optional[Path] = None
        audio_data: Optional[np.ndarray] = None

        if isinstance(audio, (str, Path)):
            audio_path = Path(audio).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Take audio file not found: {audio_path}")
            data, file_sr = sf.read(str(audio_path), dtype="float32")
            if file_sr != self.sample_rate:
                logger.warning(
                    f"Take sample rate {file_sr} differs from timeline sample rate {self.sample_rate}"
                )
                sr = file_sr
            audio_data = data
        elif isinstance(audio, np.ndarray):
            audio_data = audio.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
            if audio_data.ndim > 1:
                # Average to mono if multi-channel
                audio_data = np.mean(audio_data, axis=-1)

        dur = len(audio_data) / sr

        take = TimelineTake(
            index=index,
            speaker=speaker,
            text=text,
            audio_path=audio_path,
            audio_data=audio_data,
            sample_rate=sr,
            duration_sec=dur,
            pause_after_sec=max(0.0, pause_after_sec),
            metadata=metadata or {},
        )
        self.takes.append(take)
        return take

    def assemble(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        master_audio_filename: str = "master.wav",
        srt_filename: str = "subtitles.srt",
        vtt_filename: str = "subtitles.vtt",
        metadata_filename: str = "metadata.json",
        include_speaker_in_subtitles: bool = True,
    ) -> Dict[str, any]:
        """Assembles all takes into master audio and outputs SRT, VTT, and metadata.

        Returns dictionary containing summary metrics and output file paths.
        """
        target = output_dir or self.output_dir
        if target is None:
            raise ValueError("output_dir must be provided for timeline assembly.")
        out_dir = Path(target).resolve()
        self.output_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.takes:
            raise ValueError("No takes added to the timeline for assembly.")

        # Audio segments buffer
        audio_segments: List[np.ndarray] = []
        current_sample_cursor = 0

        # Add initial silence if requested
        if self.initial_silence_sec > 0:
            init_silence_samples = int(round(self.initial_silence_sec * self.sample_rate))
            audio_segments.append(np.zeros(init_silence_samples, dtype=np.float32))
            current_sample_cursor += init_silence_samples

        timeline_records: List[Dict[str, any]] = []

        for take in self.takes:
            take_samples = take.audio_data
            if take_samples is None and take.audio_path:
                take_samples, _ = sf.read(str(take.audio_path), dtype="float32")
                if take_samples.ndim > 1:
                    take_samples = np.mean(take_samples, axis=-1)

            n_samples = len(take_samples)
            take_dur = n_samples / self.sample_rate

            start_time = current_sample_cursor / self.sample_rate
            end_time = (current_sample_cursor + n_samples) / self.sample_rate

            take.start_time_sec = start_time
            take.end_time_sec = end_time
            take.duration_sec = take_dur

            # Append take audio
            audio_segments.append(take_samples)
            current_sample_cursor += n_samples

            # Append trailing pause
            if take.pause_after_sec > 0:
                pause_samples_len = int(round(take.pause_after_sec * self.sample_rate))
                audio_segments.append(np.zeros(pause_samples_len, dtype=np.float32))
                current_sample_cursor += pause_samples_len

            clean_text = clean_text_for_subtitles(take.text)
            timeline_records.append({
                "index": take.index,
                "speaker": take.speaker,
                "text": take.text,
                "subtitle_text": f"{take.speaker}: {clean_text}" if include_speaker_in_subtitles else clean_text,
                "audio_file": str(take.audio_path) if take.audio_path else None,
                "start_time_sec": round(start_time, 3),
                "end_time_sec": round(end_time, 3),
                "duration_sec": round(take_dur, 3),
                "pause_after_sec": round(take.pause_after_sec, 3),
                "srt_start": format_srt_timestamp(start_time),
                "srt_end": format_srt_timestamp(end_time),
            })

        # Concatenate full master audio track
        master_audio = np.concatenate(audio_segments).astype(np.float32)
        total_duration = len(master_audio) / self.sample_rate

        # 1. Export Master Audio WAV
        master_wav_path = out_dir / master_audio_filename
        sf.write(str(master_wav_path), master_audio, self.sample_rate, format="WAV", subtype="PCM_16")
        logger.info(f"Saved master audio: {master_wav_path} ({total_duration:.2f}s)")

        # 2. Export SRT Subtitles
        srt_path = out_dir / srt_filename
        srt_lines: List[str] = []
        for i, rec in enumerate(timeline_records, start=1):
            srt_lines.append(str(i))
            srt_lines.append(f"{rec['srt_start']} --> {rec['srt_end']}")
            srt_lines.append(rec['subtitle_text'])
            srt_lines.append("")  # Empty line separator

        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        logger.info(f"Saved SRT subtitles: {srt_path}")

        # 3. Export WebVTT Subtitles
        vtt_path = out_dir / vtt_filename
        vtt_lines: List[str] = ["WEBVTT", ""]
        for i, rec in enumerate(timeline_records, start=1):
            vtt_lines.append(str(i))
            v_start = format_vtt_timestamp(rec['start_time_sec'])
            v_end = format_vtt_timestamp(rec['end_time_sec'])
            vtt_lines.append(f"{v_start} --> {v_end}")
            vtt_lines.append(rec['subtitle_text'])
            vtt_lines.append("")

        vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")
        logger.info(f"Saved WebVTT subtitles: {vtt_path}")

        # 4. Export Metadata JSON
        meta_path = out_dir / metadata_filename
        metadata = {
            "total_duration_sec": round(total_duration, 3),
            "total_takes": len(self.takes),
            "sample_rate": self.sample_rate,
            "master_audio_file": str(master_wav_path.name),
            "subtitles_srt_file": str(srt_path.name),
            "subtitles_vtt_file": str(vtt_path.name),
            "takes": timeline_records,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Saved timeline metadata: {meta_path}")

        return {
            "total_duration_sec": total_duration,
            "total_takes": len(self.takes),
            "master_audio_path": master_wav_path,
            "srt_path": srt_path,
            "vtt_path": vtt_path,
            "metadata_path": meta_path,
            "records": timeline_records,
        }

    def assemble_takes(
        self,
        takes: Optional[List[TimelineTake]] = None,
        default_pause_sec: float = 0.4,
        base_name: str = "master",
        output_dir: Optional[Union[str, Path]] = None,
        include_speaker_in_subtitles: bool = True,
    ) -> Tuple[Path, Path, Path, Path, float]:
        """Assembles takes into master audio, subtitles, and metadata.

        Returns:
            Tuple of (master_path, srt_path, vtt_path, metadata_path, master_duration)
        """
        if takes is not None:
            self.takes = list(takes)

        target_dir = output_dir or self.output_dir
        if target_dir is None:
            raise ValueError("output_dir must be specified for timeline assembly.")

        master_wav_name = f"{base_name}.wav"
        srt_name = f"{base_name}.srt" if base_name != "master" else "subtitles.srt"
        vtt_name = f"{base_name}.vtt" if base_name != "master" else "subtitles.vtt"
        meta_name = f"{base_name}_metadata.json" if base_name != "master" else "metadata.json"

        res = self.assemble(
            output_dir=target_dir,
            master_audio_filename=master_wav_name,
            srt_filename=srt_name,
            vtt_filename=vtt_name,
            metadata_filename=meta_name,
            include_speaker_in_subtitles=include_speaker_in_subtitles,
        )
        return (
            res["master_audio_path"],
            res["srt_path"],
            res["vtt_path"],
            res["metadata_path"],
            res["total_duration_sec"],
        )

    @classmethod
    def load_from_metadata(
        cls,
        metadata_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "TimelineAssembler":
        """Reconstructs a TimelineAssembler and its takes from metadata.json."""
        meta_file = Path(metadata_path).resolve()
        if not meta_file.is_file():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        target_dir = Path(output_dir).resolve() if output_dir else meta_file.parent
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        sr = data.get("sample_rate", 24000)

        assembler = cls(output_dir=target_dir, sample_rate=sr)
        for t_info in data.get("takes", []):
            take_path = Path(t_info["audio_file"]) if t_info.get("audio_file") else None
            if take_path and not take_path.is_file():
                candidate = target_dir / "takes" / take_path.name
                if candidate.is_file():
                    take_path = candidate
                elif (target_dir / take_path.name).is_file():
                    take_path = target_dir / take_path.name

            take = TimelineTake(
                index=t_info["index"],
                speaker=t_info["speaker"],
                text=t_info["text"],
                audio_path=take_path,
                sample_rate=sr,
                duration_sec=t_info.get("duration_sec", 0.0),
                start_time_sec=t_info.get("start_time_sec", 0.0),
                end_time_sec=t_info.get("end_time_sec", 0.0),
                pause_after_sec=t_info.get("pause_after_sec", 0.4),
                metadata=t_info.get("metadata", {}),
            )
            assembler.takes.append(take)
        return assembler

    def export_zip(self, zip_path: Optional[Union[str, Path]] = None) -> Path:
        """Packages master.wav, subtitles, and all takes into a zip file."""
        if not self.output_dir:
            raise ValueError("No output_dir set on TimelineAssembler.")
        out_dir = Path(self.output_dir).resolve()
        if not zip_path:
            zip_path = out_dir / f"{out_dir.name}_export.zip"
        else:
            zip_path = Path(zip_path).resolve()

        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in ["master.wav", "subtitles.srt", "subtitles.vtt", "metadata.json", "script_parsed.json"]:
                fpath = out_dir / fname
                if fpath.is_file():
                    zf.write(fpath, arcname=fname)
            takes_dir = out_dir / "takes"
            if takes_dir.is_dir():
                for take_file in sorted(takes_dir.glob("*.wav")):
                    zf.write(take_file, arcname=f"takes/{take_file.name}")
        return zip_path


if __name__ == "__main__":
    import tempfile

    print("--- Testing TimelineAssembler ---")
    sr = 24000
    assembler = TimelineAssembler(sample_rate=sr)

    # Synthetic takes
    t1_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)).astype(np.float32)  # 1.0s tone
    t2_audio = np.sin(2 * np.pi * 880 * np.linspace(0, 2.0, sr * 2)).astype(np.float32)  # 2.0s tone

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        t1_path = tmp_path / "take_001.wav"
        t2_path = tmp_path / "take_002.wav"
        sf.write(str(t1_path), t1_audio, sr)
        sf.write(str(t2_path), t2_audio, sr)

        assembler.add_take(
            index=1,
            speaker="HOST",
            text="Hello world! [laugh]",
            audio=t1_path,
            pause_after_sec=0.5,
        )
        assembler.add_take(
            index=2,
            speaker="NARRATOR",
            text="And now the test finishes. [chuckle]",
            audio=t2_path,
            pause_after_sec=1.0,
        )

        res = assembler.assemble(tmp_path)
        print("Assembly summary:")
        print(f"Total duration: {res['total_duration_sec']}s")
        print(f"Master file size: {res['master_audio_path'].stat().st_size} bytes")

        srt_content = res['srt_path'].read_text()
        print("\nGenerated SRT:")
        print(srt_content)

        vtt_content = res['vtt_path'].read_text()
        print("\nGenerated VTT:")
        print(vtt_content)

        # Assertions
        # Take 1: 0.0s to 1.0s. Pause: 0.5s.
        # Take 2: 1.5s to 3.5s. Pause: 1.0s.
        # Total duration: 4.5s.
        assert abs(res['total_duration_sec'] - 4.5) < 0.01, f"Expected 4.5s, got {res['total_duration_sec']}"
        assert "00:00:00,000 --> 00:00:01,000" in srt_content
        assert "00:00:01,500 --> 00:00:03,500" in srt_content
        assert "HOST: Hello world!" in srt_content
        assert "[laugh]" not in srt_content  # Paralinguistic tags cleaned from subtitles

        print("TimelineAssembler verification passed successfully!")
