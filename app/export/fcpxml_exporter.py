"""
FCPXML-Export: baut aus bereits berechneten Highlight-Segmenten eine
Timeline (FCPXML 1.9), OHNE das Video zu schneiden/zu rendern - zum
Importieren in DaVinci Resolve (Datei > Importieren > Timeline).

Jeder Clip wird als asset-clip auf dieselbe Quelldatei referenziert (In-/
Out-Punkte wie im Original), es entsteht also ein reiner Rohschnitt/Pointer
auf das Originalvideo - keine neue Videodatei.

Die Top-N Clips bekommen einen roten Marker in der Clip-Mitte. Resolve
mappt FCPXML "To-Do"-Marker mit completed="0" auf die Farbe Rot.
"""

from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape


def _rational_time(seconds: float, fps: Fraction) -> str:
    """Rundet auf den naechsten Frame und gibt die Zeit als exaktes Rational
    (z.B. "12345/30000s") zurueck - vermeidet Rundungsfehler/Off-by-one-Frame
    Probleme beim Import."""
    return _frames_time(round(seconds * fps), fps)


def _frames_time(frames: int, fps: Fraction) -> str:
    """Gibt eine ganze Frame-Anzahl als exaktes Rational zurueck."""
    value = (Fraction(1, 1) / fps) * frames
    return f"{value.numerator}/{value.denominator}s"


def build_fcpxml(
    video_path: str,
    clips: list[dict],
    fps_num: int,
    fps_den: int,
    width: int,
    height: int,
    source_duration_sec: float,
    project_name: str,
    top_marker_indices: set,
) -> str:
    """
    clips: Liste von {"start": float, "end": float, "label": str,
    "marker_label": str} - start/end in Sekunden im ORIGINAL-Video, in
    finaler Schnitt-Reihenfolge (so wie sie auch aneinandergehaengt wuerden).

    top_marker_indices: Indizes in 'clips', die einen roten Marker in der
    Clip-Mitte bekommen sollen.
    """
    fps = Fraction(fps_num, fps_den)

    asset_uri = Path(video_path).resolve().as_uri()
    asset_duration = _rational_time(source_duration_sec, fps)
    frame_duration_frac = Fraction(1, 1) / fps
    frame_duration = f"{frame_duration_frac.numerator}/{frame_duration_frac.denominator}s"

    asset_frames = round(source_duration_sec * fps)

    # Alle Clip-Positionen in GANZEN FRAMES berechnen. Start und Laenge werden je Clip gerundet und die
    # Timeline-Position ist die Summe der bereits gerundeten Laengen - so schliessen die Clips exakt
    # aneinander (keine Ueberlappung/Luecke von 1 Frame, sonst legt Resolve das Audio auf eine 2. Spur).
    frame_clips = []
    offset_frames = 0
    for idx, clip in enumerate(clips):
        start_frames = max(0, round(clip["start"] * fps))
        dur_frames = max(1, round((clip["end"] - clip["start"]) * fps))
        if asset_frames and start_frames + dur_frames > asset_frames:
            dur_frames = max(1, asset_frames - start_frames)
        frame_clips.append((idx, clip, start_frames, dur_frames, offset_frames))
        offset_frames += dur_frames

    seq_duration = _frames_time(offset_frames, fps)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE fcpxml>",
        '<fcpxml version="1.9">',
        "  <resources>",
        f'    <format id="r1" name="HighlightCutterFormat" frameDuration="{frame_duration}" '
        f'width="{width}" height="{height}"/>',
        f'    <asset id="r2" name="{escape(Path(video_path).name)}" '
        f'src="{escape(asset_uri)}" start="0s" duration="{asset_duration}" '
        f'hasVideo="1" hasAudio="1" format="r1" audioSources="1" audioChannels="2"/>',
        "  </resources>",
        "  <library>",
        '    <event name="Highlight Cutter Export">',
        f'      <project name="{escape(project_name)}">',
        f'        <sequence format="r1" duration="{seq_duration}">',
        "          <spine>",
    ]

    for idx, clip, start_frames, dur_frames, clip_offset_frames in frame_clips:
        offset = _frames_time(clip_offset_frames, fps)
        start = _frames_time(start_frames, fps)
        clip_dur = _frames_time(dur_frames, fps)
        name = escape(clip.get("label", f"Clip {idx + 1}"))

        if idx in top_marker_indices:
            marker_start = _frames_time(start_frames + dur_frames // 2, fps)
            marker_label = escape(clip.get("marker_label", "Top Highlight"))
            lines.append(
                f'            <asset-clip ref="r2" name="{name}" offset="{offset}" '
                f'start="{start}" duration="{clip_dur}">'
            )
            lines.append(
                f'              <marker start="{marker_start}" duration="{frame_duration}" '
                f'value="{marker_label}" completed="0"/>'
            )
            lines.append("            </asset-clip>")
        else:
            lines.append(
                f'            <asset-clip ref="r2" name="{name}" offset="{offset}" '
                f'start="{start}" duration="{clip_dur}"/>'
            )

    lines += [
        "          </spine>",
        "        </sequence>",
        "      </project>",
        "    </event>",
        "  </library>",
        "</fcpxml>",
    ]

    return "\n".join(lines)
