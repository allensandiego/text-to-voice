"""Parser module for actor/screenplay text scripts into structured dialogue items.

Supports:
- Fountain screenplay format (UPPERCASE characters, parentheticals, dialogue)
- Colon style format (SPEAKER: Dialogue...)
- Mixed scripts and standalone pause directives
- Pause directives: [pause: 1.5s], (pause 2.0s), (beat)
- Paralinguistic tag normalization: (laughs) -> [laugh], (chuckles) -> [chuckle], etc.
- Preservation of existing native tags ([laugh], [sigh], etc.)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Supported Chatterbox-Turbo paralinguistic tags and their stage direction synonyms
STAGE_DIRECTION_MAP: Dict[str, str] = {
    # Laughs
    "laugh": "[laugh]",
    "laughs": "[laugh]",
    "laughing": "[laugh]",
    "giggle": "[laugh]",
    "giggles": "[laugh]",
    "giggling": "[laugh]",
    "haha": "[laugh]",
    "chuckle": "[chuckle]",
    "chuckles": "[chuckle]",
    "chuckling": "[chuckle]",
    "snicker": "[chuckle]",
    "snickers": "[chuckle]",
    # Sighs
    "sigh": "[sigh]",
    "sighs": "[sigh]",
    "sighing": "[sigh]",
    # Gasps
    "gasp": "[gasp]",
    "gasps": "[gasp]",
    "gasping": "[gasp]",
    # Coughs
    "cough": "[cough]",
    "coughs": "[cough]",
    "coughing": "[cough]",
    # Groans
    "groan": "[groan]",
    "groans": "[groan]",
    "groaning": "[groan]",
    "grunt": "[groan]",
    "grunts": "[groan]",
    # Sniff
    "sniff": "[sniff]",
    "sniffs": "[sniff]",
    "sniffing": "[sniff]",
    # Throat clearing
    "clear throat": "[clear throat]",
    "clears throat": "[clear throat]",
    "clearing throat": "[clear throat]",
    "throat clearing": "[clear throat]",
    "throat-clearing": "[clear throat]",
    # Whispering
    "whisper": "[whispering]",
    "whispers": "[whispering]",
    "whispering": "[whispering]",
    # Sarcastic
    "sarcastic": "[sarcastic]",
    "sarcastically": "[sarcastic]",
    "sarcasm": "[sarcastic]",
    # Crying
    "cry": "[crying]",
    "cries": "[crying]",
    "crying": "[crying]",
    "sob": "[crying]",
    "sobs": "[crying]",
    "sobbing": "[crying]",
    # Emotion
    "angry": "[angry]",
    "angrily": "[angry]",
    "fear": "[fear]",
    "fearful": "[fear]",
    "happy": "[happy]",
    "happily": "[happy]",
    "surprised": "[surprised]",
    "surprise": "[surprised]",
    "dramatic": "[dramatic]",
    "dramatically": "[dramatic]",
    "narration": "[narration]",
    "narrator": "[narration]",
    "shush": "[shush]",
    "shushing": "[shush]",
    "ad": "[advertisement]",
    "advertisement": "[advertisement]",
}

# Regex to match pause directives like [pause: 1.5s], (pause 2.0s), (beat), [beat]
PAUSE_PATTERN = re.compile(
    r'(?:\[|\()(?:pause\s*:?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?|(beat))(?:\]|\))',
    re.IGNORECASE,
)

# Regex to detect bracket or parenthetical tags
PARENTHETICAL_OR_TAG_PATTERN = re.compile(
    r'(\([^\)]+\)|\[[^\]]+\]|\*[^\*]+\*)'
)

# Fountain scene headings / transitions
SCENE_HEADING_PATTERN = re.compile(
    r'^(?:INT\.|EXT\.|EST\.|INT/EXT\.|INT\./EXT\.|I/E\.)\s+',
    re.IGNORECASE,
)
TRANSITION_PATTERN = re.compile(
    r'^(?:CUT TO:|FADE IN:|FADE OUT\.|FADE TO BLACK\.|DISSOLVE TO:)$',
    re.IGNORECASE,
)


@dataclass
class ScriptItem:
    """Represents a single parsed line/take in the voiceover script."""
    index: int
    speaker: str
    text: str
    pause_after_sec: float = 0.4
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "speaker": self.speaker,
            "text": self.text,
            "pause_after_sec": self.pause_after_sec,
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScriptItem":
        return cls(
            index=data["index"],
            speaker=data["speaker"],
            text=data["text"],
            pause_after_sec=data.get("pause_after_sec", 0.4),
            raw_text=data.get("raw_text", ""),
            metadata=data.get("metadata", {}),
        )


def parse_pause_directive(text: str, default_pause: float = 0.4, default_beat: float = 0.8) -> Optional[float]:
    """Extracts pause duration in seconds if text is a pause directive, else None."""
    match = PAUSE_PATTERN.fullmatch(text.strip())
    if not match:
        return None
    sec_str, is_beat = match.groups()
    if is_beat:
        return default_beat
    if sec_str is not None:
        return float(sec_str)
    return default_pause


def normalize_paralinguistics(text: str) -> Tuple[str, List[str]]:
    """Converts stage direction parentheticals into native Chatterbox tags or strips non-vocal cues.

    Preserves existing native bracket tags like [laugh], [sigh], etc.
    Unmapped acting directions like (smiling) or (looking at camera) are stripped
    from spoken dialogue to prevent TTS artifacting and returned in the stripped list.

    Returns:
        (normalized_text, list_of_stripped_non_vocal_cues)
    """
    stripped_cues: List[str] = []

    def replace_tag(match: re.Match) -> str:
        token = match.group(1)
        # Already a bracket tag like [laugh] or [pause: 1s]
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1].strip().lower()
            # If already a known tag or pause, preserve
            if PAUSE_PATTERN.fullmatch(token):
                return token
            if inner in STAGE_DIRECTION_MAP.values() or any(inner == v[1:-1] for v in STAGE_DIRECTION_MAP.values()):
                return f"[{inner}]"
            # Return custom bracket tag as-is
            return token

        # Parenthetical (cue) or asterisk *cue*
        inner = token[1:-1].strip().lower()

        # Check if pause directive - leave for pause splitter
        if PAUSE_PATTERN.fullmatch(token):
            return token

        # Check if mapped paralinguistic tag
        if inner in STAGE_DIRECTION_MAP:
            return STAGE_DIRECTION_MAP[inner]

        # Multi-word check
        normalized_inner = re.sub(r'\s+', ' ', inner)
        if normalized_inner in STAGE_DIRECTION_MAP:
            return STAGE_DIRECTION_MAP[normalized_inner]

        # Non-vocal stage direction (e.g. "smiling", "looking at camera") -> strip from spoken text
        stripped_cues.append(token)
        return ""

    normalized = PARENTHETICAL_OR_TAG_PATTERN.sub(replace_tag, text)
    # Clean up double whitespace introduced by stripping
    normalized = re.sub(r'[ \t]+', ' ', normalized).strip()
    # Clean up punctuation formatting artifacts, e.g. " ." -> "."
    normalized = re.sub(r'\s+([,.:;!?])', r'\1', normalized)
    return normalized, stripped_cues


def clean_speaker_name(speaker: str) -> str:
    """Normalizes speaker name by removing parentheticals like (V.O.), (O.S.), and symbols."""
    # Remove Fountain @ escape
    speaker = speaker.lstrip("@").strip()
    # Remove extensions like (V.O.), (O.S.), (CONT'D)
    speaker = re.sub(r'\(.*?\)', '', speaker).strip()
    return speaker.upper()


def is_fountain_character_cue(line: str) -> bool:
    """Determines whether a line qualifies as a Fountain character cue."""
    clean = line.strip()
    if not clean:
        return False
    # Scene headings or transitions are not character cues
    if SCENE_HEADING_PATTERN.match(clean) or TRANSITION_PATTERN.match(clean):
        return False
    # Fountain character cues can start with '@'
    if clean.startswith("@"):
        return True
    # In Fountain, character cue must be uppercase
    # Strip any parenthetical extension e.g. "HOST (V.O.)"
    base = re.sub(r'\(.*?\)', '', clean).strip()
    if not base:
        return False
    # Must contain at least one letter and all letters must be uppercase
    letters = [c for c in base if c.isalpha()]
    if not letters:
        return False
    # Allowed characters: uppercase letters, digits, spaces, hyphens, periods, apostrophes
    return base.isupper() and all(c.isalnum() or c in " -.'’" for c in base)


def split_dialogue_on_pauses(
    speaker: str,
    dialogue: str,
    raw_text: str,
    default_pause: float = 0.4,
    default_beat: float = 0.8,
) -> List[ScriptItem]:
    """Splits a single dialogue line on inline pause directives into multiple ScriptItems.

    For example:
        "Hello world! [pause: 1.5s] Welcome back."
    Becomes:
        Item 1: text="Hello world!", pause_after_sec=1.5
        Item 2: text="Welcome back.", pause_after_sec=default_pause
    """
    items: List[ScriptItem] = []
    current_pos = 0

    for match in PAUSE_PATTERN.finditer(dialogue):
        pause_start, pause_end = match.span()
        segment = dialogue[current_pos:pause_start].strip()

        # Parse pause duration
        sec_str, is_beat = match.groups()
        if is_beat:
            duration = default_beat
        elif sec_str is not None:
            duration = float(sec_str)
        else:
            duration = default_pause

        if segment:
            # Segment before the pause directive
            norm_text, stripped = normalize_paralinguistics(segment)
            if norm_text:
                items.append(
                    ScriptItem(
                        index=0,  # Will be renumbered
                        speaker=speaker,
                        text=norm_text,
                        pause_after_sec=duration,
                        raw_text=segment,
                        metadata={"stage_directions": stripped} if stripped else {},
                    )
                )
        elif items:
            # If no text before pause, update previous item's pause duration
            items[-1].pause_after_sec = duration

        current_pos = pause_end

    # Remaining text after the last pause directive (or entire text if no pause directives)
    remaining = dialogue[current_pos:].strip()
    if remaining:
        norm_text, stripped = normalize_paralinguistics(remaining)
        if norm_text:
            items.append(
                ScriptItem(
                    index=0,
                    speaker=speaker,
                    text=norm_text,
                    pause_after_sec=default_pause,
                    raw_text=remaining,
                    metadata={"stage_directions": stripped} if stripped else {},
                )
            )

    return items


def parse_script(
    script_text: str,
    default_pause: float = 0.4,
    default_beat: float = 0.8,
) -> List[ScriptItem]:
    """Parses a script into a list of structured ScriptItem objects.

    Supports:
    - Colon format: `HOST: Hello world!`
    - Fountain format:
        HOST
        (chuckles)
        Hello world!
    - Standalone pause directives: `(beat)`, `[pause: 1.5s]`
    - Inline pause directives and stage direction normalizations.
    """
    lines = script_text.splitlines()
    items: List[ScriptItem] = []

    current_speaker: Optional[str] = None
    dialogue_buffer: List[str] = []
    parenthetical_buffer: List[str] = []

    def flush_dialogue() -> None:
        nonlocal current_speaker, dialogue_buffer, parenthetical_buffer
        if not current_speaker:
            dialogue_buffer.clear()
            parenthetical_buffer.clear()
            return

        combined_text = " ".join(dialogue_buffer).strip()
        if not combined_text and not parenthetical_buffer:
            return

        # Prepend any pending parenthetical tags to dialogue text if applicable
        prefix_tags: List[str] = []
        for paren in parenthetical_buffer:
            # Check if paren is a pause
            p_val = parse_pause_directive(paren, default_pause, default_beat)
            if p_val is not None:
                if items:
                    items[-1].pause_after_sec = p_val
                continue

            norm_tag, _ = normalize_paralinguistics(paren)
            if norm_tag:
                prefix_tags.append(norm_tag)

        if prefix_tags:
            combined_text = f"{' '.join(prefix_tags)} {combined_text}".strip()

        if combined_text:
            sub_items = split_dialogue_on_pauses(
                speaker=current_speaker,
                dialogue=combined_text,
                raw_text=combined_text,
                default_pause=default_pause,
                default_beat=default_beat,
            )
            items.extend(sub_items)

        dialogue_buffer.clear()
        parenthetical_buffer.clear()

    colon_pattern = re.compile(r'^([A-Za-z0-9_ -]+):\s*(.*)$')
    fountain_meta_keys = {
        "title", "author", "authors", "credit", "source", "draft date",
        "date", "contact", "copyright", "notes", "revision", "scene"
    }

    in_title_page = True

    for line in lines:
        stripped = line.strip()

        # Blank line: in Fountain, ends a character dialogue block
        if not stripped:
            flush_dialogue()
            current_speaker = None
            in_title_page = False
            continue

        # Check for Fountain title page metadata at start of document
        if in_title_page and ":" in stripped:
            prefix = stripped.split(":", 1)[0].strip().lower()
            if prefix in fountain_meta_keys:
                continue

        # Check for standalone pause directive (beat or pause)
        pause_dur = parse_pause_directive(stripped, default_pause, default_beat)
        if pause_dur is not None:
            flush_dialogue()
            if items:
                items[-1].pause_after_sec = pause_dur
            continue

        # Check for colon style line: "SPEAKER: Dialogue..."
        colon_match = colon_pattern.match(stripped)
        if colon_match:
            raw_spk, speech = colon_match.groups()
            # If the prefix is a known title page key, skip
            if raw_spk.strip().lower() in fountain_meta_keys:
                continue

            flush_dialogue()
            current_speaker = clean_speaker_name(raw_spk)

            # Check if line contains inline pause or dialogue
            if speech.strip():
                sub_items = split_dialogue_on_pauses(
                    speaker=current_speaker,
                    dialogue=speech.strip(),
                    raw_text=speech.strip(),
                    default_pause=default_pause,
                    default_beat=default_beat,
                )
                items.extend(sub_items)
            continue

        # Check for Fountain character cue (UPPERCASE name preceded by blank)
        if is_fountain_character_cue(stripped) and not current_speaker:
            flush_dialogue()
            current_speaker = clean_speaker_name(stripped)
            continue

        # Check for Fountain parenthetical: "(chuckles)" or "(beat)"
        if stripped.startswith("(") and stripped.endswith(")"):
            # Check if this parenthetical is a beat/pause
            p_val = parse_pause_directive(stripped, default_pause, default_beat)
            if p_val is not None:
                flush_dialogue()
                if items:
                    items[-1].pause_after_sec = p_val
                continue

            if current_speaker:
                # If there's already dialogue in the buffer, flush it first so
                # this parenthetical applies to the subsequent dialogue
                if dialogue_buffer:
                    flush_dialogue()
                parenthetical_buffer.append(stripped)
                continue

        # Skip scene headings / transitions if not inside dialogue
        if not current_speaker:
            if SCENE_HEADING_PATTERN.match(stripped) or TRANSITION_PATTERN.match(stripped):
                continue
            # Also ignore title block lines (e.g. "Title: ...", "Author: ...")
            if ":" in stripped and any(stripped.lower().startswith(p) for p in ["title:", "author:", "date:", "draft:"]):
                continue

        # Otherwise, if we have a current speaker, it's dialogue!
        if current_speaker:
            dialogue_buffer.append(stripped)

    # Flush any remaining buffer at end of script
    flush_dialogue()

    # Renumber items with 1-based indices
    for idx, item in enumerate(items, start=1):
        item.index = idx

    return items


def parse_script_file(
    file_path: str,
    default_pause: float = 0.4,
    default_beat: float = 0.8,
) -> List[ScriptItem]:
    """Reads and parses a script file from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Script file not found: {file_path}")
    content = path.read_text(encoding="utf-8")
    return parse_script(content, default_pause=default_pause, default_beat=default_beat)


class ScriptParser:
    """Convenience class wrapper for parsing script text and files."""

    @staticmethod
    def parse(
        text: str,
        default_pause: float = 0.4,
        default_beat: float = 0.8,
    ) -> List[ScriptItem]:
        """Parses script text into a list of ScriptItem objects."""
        return parse_script(text, default_pause=default_pause, default_beat=default_beat)

    @staticmethod
    def parse_file(
        file_path: Union[str, Path],
        default_pause: float = 0.4,
        default_beat: float = 0.8,
    ) -> List[ScriptItem]:
        """Parses a script file into a list of ScriptItem objects."""
        return parse_script_file(str(file_path), default_pause=default_pause, default_beat=default_beat)




if __name__ == "__main__":
    test_colon_script = """
    HOST: Hello world! (chuckles) Welcome to the test.
    HOST: Let's see if this works. [pause: 1.5s] Yes it does!
    (beat)
    NARRATOR: (whispers) The test was running smoothly.
    HOST: [laugh] That was awesome!
    """

    test_fountain_script = """
    Title: Test Screenplay
    Author: Allen

    INT. STUDIO - DAY

    HOST
    (chuckles)
    Welcome to our audio script! Today we test the engine.

    (beat)

    NARRATOR
    The host spoke with utmost confidence.

    HOST (V.O.)
    Look at this benchmark! [pause: 2.0s] It's under two seconds.
    (laughs)
    Unbelievable!
    """

    print("--- Testing Colon Script Parsing ---")
    items1 = parse_script(test_colon_script)
    for it in items1:
        print(f"[{it.index}] {it.speaker} (pause={it.pause_after_sec}s): {it.text}")

    print("\n--- Testing Fountain Script Parsing ---")
    items2 = parse_script(test_fountain_script)
    for it in items2:
        print(f"[{it.index}] {it.speaker} (pause={it.pause_after_sec}s): {it.text}")

    # Assertions
    assert len(items1) >= 4, f"Expected at least 4 items for colon script, got {len(items1)}"
    assert items1[0].speaker == "HOST"
    assert "[chuckle]" in items1[0].text
    assert items1[1].pause_after_sec == 1.5
    assert items1[3].speaker == "NARRATOR"
    assert "[whispering]" in items1[3].text
    assert items1[4].speaker == "HOST"
    assert "[laugh]" in items1[4].text

    assert len(items2) >= 4, f"Expected at least 4 items for fountain script, got {len(items2)}"
    assert items2[0].speaker == "HOST"
    assert "[chuckle]" in items2[0].text
    assert items2[0].pause_after_sec == 0.8  # beat following item 0
    assert items2[1].speaker == "NARRATOR"
    assert items2[2].pause_after_sec == 2.0  # split on [pause: 2.0s]
    assert "[laugh]" in items2[4].text

    print("\nAll parser tests passed successfully!")
