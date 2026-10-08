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
    frames = round(seconds * fps)
    frame_duration = Fraction(1, 1) / fps
    value = frame_duration * frames
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
    audio_channels_per_stream: list[int] | None = None,
) -> str:
    """
    clips: Liste von {"start": float, "end": float, "label": str,
    "marker_label": str} - start/end in Sekunden im ORIGINAL-Video, in
    finaler Schnitt-Reihenfolge (so wie sie auch aneinandergehaengt wuerden).

    top_marker_indices: Indizes in 'clips', die einen roten Marker in der
    Clip-Mitte bekommen sollen.

    audio_channels_per_stream: Kanalzahl JE Audiospur der Quelldatei, in
    Stream-Reihenfolge (z.B. [2, 1, 1, 1, 1, 1] fuer 1 Stereo-Mix + 5 Mono-
    Spuren). JEDER EINZELNE Kanal bekommt eine eigene Lane/Spur (nicht pro
    Stream gebuendelt) - Resolve hat sich als unzuverlaessig erwiesen, wenn
    mehrere Kanaele in einem einzigen audio-channel-source (srcCh="1, 2")
    zusammengefasst werden (Kanaele landeten getrennt/falsch zugeordnet statt
    als Stereopaar). None/leer = 1 Stereo-Spur, 2 Lanes (altes Verhalten).
    """
    fps = Fraction(fps_num, fps_den)

    asset_uri = Path(video_path).resolve().as_uri()
    asset_duration = _rational_time(source_duration_sec, fps)
    frame_duration_frac = Fraction(1, 1) / fps
    frame_duration = f"{frame_duration_frac.numerator}/{frame_duration_frac.denominator}s"

    total_duration = sum(c["end"] - c["start"] for c in clips)
    seq_duration = _rational_time(total_duration, fps)

    audio_channels_per_stream = audio_channels_per_stream or [2]
    total_audio_channels = sum(audio_channels_per_stream) or 2
    audio_source_count = len(audio_channels_per_stream)

    # Jeder einzelne Roh-Kanal bekommt seine eigene Lane (1 Kanal = 1 Spur in
    # Resolve), statt mehrere Kanaele eines Streams in einem srcCh-Listenwert
    # zu buendeln.
    audio_lanes = []
    channel_number = 1
    lane_counter = 0
    for stream_idx, ch_count in enumerate(audio_channels_per_stream):
        for ch_in_stream in range(ch_count):
            role = (
                f"track-{stream_idx + 1}" if ch_count == 1
                else f"track-{stream_idx + 1}-ch{ch_in_stream + 1}"
            )
            audio_lanes.append({
                "src_ch": str(channel_number),
                "lane": 0 if lane_counter == 0 else -lane_counter,
                "role": role,
            })
            channel_number += 1
            lane_counter += 1

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE fcpxml>",
        '<fcpxml version="1.9">',
        "  <resources>",
        f'    <format id="r1" name="HighlightCutterFormat" frameDuration="{frame_duration}" '
        f'width="{width}" height="{height}"/>',
        f'    <asset id="r2" name="{escape(Path(video_path).name)}" '
        f'src="{escape(asset_uri)}" start="0s" duration="{asset_duration}" '
        f'hasVideo="1" hasAudio="1" format="r1" '
        f'audioSources="{audio_source_count}" audioChannels="{total_audio_channels}"/>',
        "  </resources>",
        "  <library>",
        '    <event name="Highlight Cutter Export">',
        f'      <project name="{escape(project_name)}">',
        f'        <sequence format="r1" duration="{seq_duration}">',
        "          <spine>",
    ]

    offset_sec = 0.0
    for idx, clip in enumerate(clips):
        clip_start = clip["start"]
        clip_duration = clip["end"] - clip["start"]

        offset = _rational_time(offset_sec, fps)
        start = _rational_time(clip_start, fps)
        clip_dur = _rational_time(clip_duration, fps)
        name = escape(clip.get("label", f"Clip {idx + 1}"))

        lines.append(
            f'            <asset-clip ref="r2" name="{name}" offset="{offset}" '
            f'start="{start}" duration="{clip_dur}">'
        )

        # Jede Original-Audiospur als eigene Lane referenzieren, damit sie in
        # Resolve als eigene, einzeln bearbeitbare Audiospur landet.
        for lane_info in audio_lanes:
            lane_attr = "" if lane_info["lane"] == 0 else f' lane="{lane_info["lane"]}"'
            lines.append(
                f'              <audio-channel-source srcCh="{lane_info["src_ch"]}" '
                f'role="{lane_info["role"]}"{lane_attr}/>'
            )

        if idx in top_marker_indices:
            marker_time_sec = clip_start + clip_duration / 2
            marker_start = _rational_time(marker_time_sec, fps)
            marker_label = escape(clip.get("marker_label", "Top Highlight"))
            lines.append(
                f'              <marker start="{marker_start}" duration="{frame_duration}" '
                f'value="{marker_label}" completed="0"/>'
            )

        lines.append("            </asset-clip>")

        offset_sec += clip_duration

    lines += [
        "          </spine>",
        "        </sequence>",
        "      </project>",
        "    </event>",
        "  </library>",
        "</fcpxml>",
    ]

    return "\n".join(lines)
