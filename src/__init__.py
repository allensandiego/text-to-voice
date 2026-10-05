"""text-to-voice: Automated video blog voiceover generator using Chatterbox-Turbo."""

from .parser import ScriptParser, ScriptItem, parse_script, parse_script_file
from .engine import VoiceoverEngine, detect_device
from .timeline import TimelineAssembler, TimelineTake

__version__ = "0.1.0"

__all__ = [
    "ScriptParser",
    "ScriptItem",
    "parse_script",
    "parse_script_file",
    "VoiceoverEngine",
    "detect_device",
    "TimelineAssembler",
    "TimelineTake",
]
