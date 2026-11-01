"""
Dateinamen fuer FCPXML-Timelines und ihre Log-Dateien.

Alle Dateien eines Exports teilen sich denselben "Stem" (z.B. "timeline - Mein Video"):
    <stem>.fcpxml, <stem>_log.txt, <stem>_timeline.txt, <stem>_final_timeline.txt, <stem>_metadata.json
Der Stem ist zugleich der Timeline-Name in DaVinci Resolve.
"""

import re
from pathlib import Path

DEFAULT_PREFIX = "timeline - "
MAX_STEM_LENGTH = 120  # haelt auch lange Ausgabepfade unter dem Windows-Limit von 260 Zeichen

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def sanitize_name(text) -> str:
    """Ersetzt unter Windows unzulaessige Zeichen durch '_' und entfernt Leerraum an den Raendern."""
    return _INVALID_CHARS.sub("_", str(text or "")).strip()


def _clean_prefix(prefix) -> str:
    """Bereinigt das Praefix, behaelt aber ein abschliessendes Leerzeichen ("timeline - ")."""
    text = str(prefix or "")
    cleaned = sanitize_name(text)
    if not cleaned:
        return ""
    return cleaned + (" " if text.endswith(" ") else "")


def _finalize_stem(stem: str) -> str:
    stem = stem[:MAX_STEM_LENGTH].rstrip(" .")  # Windows erlaubt keine Namen, die auf Punkt/Leerzeichen enden
    if not stem:
        stem = "timeline"
    if stem.split(".")[0].upper() in _RESERVED:
        stem = f"_{stem}"
    return stem


def _uniquify(stems: list[str]) -> list[str]:
    """Doppelte Stems (Windows: ohne Beachtung der Gross-/Kleinschreibung) bekommen " (2)", " (3)", ..."""
    result: list[str] = []
    used: set[str] = set()
    for stem in stems:
        unique, n = stem, 1
        while unique.lower() in used:
            n += 1
            unique = _finalize_stem(f"{stem[:MAX_STEM_LENGTH - 6]} ({n})")
        used.add(unique.lower())
        result.append(unique)
    return result


def build_timeline_stems(video_paths: list, custom_name: str = "", prefix: str = DEFAULT_PREFIX) -> list[str]:
    """
    Gibt pro Video den Datei-Stem zurueck: <Praefix><Name>.
    Name = eigener Name (leer: Dateiname des Videos). Bei mehreren Videos wird an den eigenen Namen
    " - <Videoname>" angehaengt, damit die Dateien nicht ueberschrieben werden. Doppelte Stems
    innerhalb eines Laufs bekommen " (2)", " (3)", ...
    """
    prefix = _clean_prefix(prefix)
    custom = sanitize_name(custom_name)
    multiple = len(video_paths) > 1

    stems: list[str] = []
    for video in video_paths:
        base = sanitize_name(Path(str(video)).stem)
        if custom:
            name = f"{custom} - {base}" if (multiple and base) else custom
        else:
            name = base
        stems.append(_finalize_stem(f"{prefix}{name}"))
    return _uniquify(stems)


def resolve_stems(names: list, fallbacks: list[str]) -> list[str]:
    """
    Macht aus frei eingegebenen Dateinamen (komplett, ohne zusaetzliches Praefix) die endgueltigen Stems.
    Leere Eingaben fallen auf den jeweiligen Standardnamen zurueck. Die Bereinigung entspricht der
    von build_timeline_stems, doppelte Namen werden eindeutig gemacht.
    """
    stems: list[str] = []
    for name, fallback in zip(names, fallbacks):
        cleaned = sanitize_name(name)
        stems.append(_finalize_stem(cleaned) if cleaned.rstrip(" .") else fallback)
    return _uniquify(stems)
