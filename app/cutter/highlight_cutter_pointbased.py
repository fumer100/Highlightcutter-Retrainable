"""
Highlight-Cutter (Punktbasiert) fuer Gameplay-Videos
======================================================
Alternative zum linearen/zustandsbehafteten Sustain-Algorithmus
(highlight_cutter.py). Hier laeuft die Erkennung in zwei klar getrennten
Phasen:

  Phase A - Punkte sammeln
    Alle Trigger-Punkte im gesamten Video werden unabhaengig voneinander
    erfasst: Audio-Peaks (hoher RMS-Percentile) und YOLO-Events. Jeder
    Punkt ist zu diesem Zeitpunkt komplett eigenstaendig, es gibt noch
    keine Reihenfolge-Abhaengigkeit.

  Phase B - Clustering
    Punkte die zeitlich nah beieinander liegen werden zu Clustern
    gruppiert (unabhaengig vom Trigger-Typ). Ein Cluster kann aus einem
    einzelnen Punkt bestehen oder aus vielen (z.B. Multi-Kill-Serie).

  Phase C - Fenster pro Cluster aufspannen
    Jeder Cluster bekommt ein Zeitfenster:
      - Pre-Buffer abhaengig vom Trigger-Typ des ERSTEN Punkts im Cluster
        (audio_pre_buffer_sec vs yolo_pre_buffer_sec)
      - Post-Buffer abhaengig von:
          a) der Lautstaerke IM GESAMTEN Cluster-Fenster (laenger laut
             im fusionierten Fenster -> laengeres Fenster, iterativ neu
             berechnet bis stabil)
          b) Cluster-Groesse (mehr Punkte im Cluster -> mehr Bonus)
          c) Cluster-Dichte (dichter beieinander -> staerkerer Bonus
             pro Punkt als ein lockeres Cluster)
      - Isolierte Einzel-Punkte (Cluster-Groesse 1, kein Nachbar in der
        Naehe) bekommen ein kuerzeres Basis-Fenster und koennen optional
        als Rauschen verworfen werden (min_cluster_confidence)

  Phase D - Merge
    Ueberlappende/nahe finale Fenster werden zusammengefuehrt.

Anschliessend laufen optional dieselben Stufe 2 (interne Luecken
entfernen) und Stufe 3 (Mikro-Trimming) wie im linearen Skript.

NEU: process_video / preload_yolo_events / cut_and_concat nehmen
optional einen progress_callback(phase: str, percent: float) entgegen,
damit z.B. eine tkinter-GUI live einen Fortschrittsbalken anzeigen kann.
Alle bisherigen print()-Ausgaben bleiben unveraendert erhalten.

Benoetigte Installation:
    pip install librosa numpy ultralytics opencv-python

ffmpeg muss installiert und im PATH verfuegbar sein.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
import json
import numpy as np
import librosa
import torch
print("=" * 40)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfügbar?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Aktuelle Grafikkarte: {torch.cuda.get_device_name(0)}")
print("=" * 40)
try:
    from ultralytics import YOLO
    import cv2
except ImportError:
    YOLO = None
    cv2 = None


# ----------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------

@dataclass
class Config:
    # --- Phase A: Trigger-Erkennung ---
    audio_threshold_percentile: float = 90   # Schwelle fuer Audio-Peak-Punkte
    yolo_model_path: str | None = None
    yolo_sample_every_n_frames: int = 5
    yolo_confidence: float = 0.4
    yolo_target_classes: list = field(default_factory=lambda: ["event", "hitmarker"])

    # --- Phase B: Clustering ---
    cluster_max_gap_sec: float = 1.75         # max. Abstand zwischen Punkten im selben Cluster

    # --- Phase C: Fenster pro Cluster ---
    audio_pre_buffer_sec: float = 1.25       # Pre-Buffer wenn Cluster mit Audio-Peak beginnt
    yolo_pre_buffer_sec: float = 1.25        # Pre-Buffer wenn Cluster mit YOLO-Event beginnt

    base_post_buffer_sec: float = 0.725      # Basis-Nachlaufzeit nach letztem Punkt im Cluster

    # Lautstaerke-abhaengiges Wachstum (bezieht sich auf das GESAMTE,
    # bereits fusionierte Cluster-Fenster, iterativ berechnet)
    loud_threshold_percentile: float = 75.0  # ab wann ein Sample im Fenster als "laut" gilt
    loud_growth_per_sec: float = 0.125        # Post-Buffer-Zuwachs pro Sekunde "laute" Zeit im Fenster
    loud_growth_max_sec: float = 1.5         # Kappung des lautstaerke-bedingten Zuwachses

    # Cluster-Groesse-Bonus (mehr Punkte im Cluster = mehr Post-Buffer)
    size_bonus_per_point_sec: float = 0.175   # Bonus-Sekunden pro zusaetzlichem Punkt im Cluster
    size_bonus_max_sec: float = 1.75       # Kappung des Groessen-Bonus

    # Cluster-Dichte-Bonus (je dichter die Punkte, desto staerker der
    # Bonus PRO Punkt - kompensiert/verstaerkt den Groessen-Bonus)
    density_bonus_enabled: bool = True
    density_reference_gap_sec: float = 1.0   # Referenz-Abstand: bei diesem Abstand = Bonus-Faktor 1.0
    density_bonus_max_multiplier: float = 2.0  # max. Multiplikator auf den Groessen-Bonus bei sehr dichten Clustern

    # Rausch-Filter fuer isolierte Einzel-Punkte
    min_cluster_confidence_enabled: bool = True
    isolated_point_post_buffer_sec: float = 1.5  # kuerzeres Fenster fuer Cluster-Groesse 1
    discard_isolated_audio_points: bool = True  # isolierte reine Audio-Punkte komplett verwerfen?

    max_clip_duration_sec: float = 30.0

    # --- Stufe 2: Interne Lücken entfernen (wie im linearen Skript) ---
    enable_internal_gap_removal: bool = True
    internal_silence_threshold_percentile: float = 60.0
    internal_min_gap_sec: float = 2.5
    internal_gap_padding_sec: float = 0.8

    # --- Stufe 3: Mikro-Trimming ---
    enable_micro_trim: bool = True
    micro_trim_search_window_sec: float = 0.2
    micro_trim_silence_percentile: float = 50.0

    # --- Merge ---
    merge_gap_sec: float = 1.0

    # --- Encoding ---
    use_gpu_encoding: bool = True
    crf: int = 14
    nvenc_cq: int = 15


def fmt_time(sec: float) -> str:
    """Formatiert Sekunden als MM:SS.ss (Sek)."""
    m = int(sec) // 60
    s = sec % 60
    return f"{m:02d}:{s:05.2f} ({sec:.1f}s)"


# ----------------------------------------------------------------------
# Phase A: Trigger-Punkte sammeln
# ----------------------------------------------------------------------

def compute_rms(video_path: str):
    print(f"[Audio] Lade Audiospur ...")
    y, sr = librosa.load(video_path, sr=22050, mono=True)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, times


def preload_yolo_events(video_path: str, cfg: Config, progress_callback=None) -> set:
    """
    progress_callback(phase: str, percent: float) wird optional bei
    jedem analysierten Frame mit dem aktuellen Fortschritt (0-100) aufgerufen.
    """
    if cfg.yolo_model_path is None or YOLO is None:
        print("[YOLO] Kein Modell konfiguriert - YOLO uebersprungen.")
        return set()

    print(f"[YOLO] Lade Modell: {cfg.yolo_model_path}")
    model = YOLO(cfg.yolo_model_path).to("cuda")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0
    event_times = set()

    print("[YOLO] Analysiere Video ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % cfg.yolo_sample_every_n_frames == 0:
            results = model.predict(frame, conf=cfg.yolo_confidence, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    if cfg.yolo_target_classes and cls_name not in cfg.yolo_target_classes:
                        continue
                    t = round(frame_idx / fps, 2)
                    event_times.add(t)
                    break

            if progress_callback is not None:
                progress_callback("YOLO-Analyse", min(100.0, frame_idx / total_frames * 100))

        frame_idx += 1

    cap.release()
    print(f"[YOLO] {len(event_times)} Event-Zeitstempel gefunden.")
    if progress_callback is not None:
        progress_callback("YOLO-Analyse", 100.0)
    return event_times


def collect_trigger_points(
    rms: np.ndarray,
    times: np.ndarray,
    cfg: Config,
    yolo_event_times: set,
) -> list[dict]:
    """
    PHASE A: Sammelt ALLE Trigger-Punkte unabhaengig voneinander.
    Jeder Punkt: {"time": float, "type": "audio"|"yolo"}
    """
    peak_threshold = np.percentile(rms, cfg.audio_threshold_percentile)
    print(f"[Phase A] Audio-Peak-Threshold (P{cfg.audio_threshold_percentile}): {peak_threshold:.4f}")

    points = []

    for i in range(len(rms)):
        if rms[i] >= peak_threshold:
            points.append({"time": float(times[i]), "type": "audio", "rms": float(rms[i])})

    for t in yolo_event_times:
        points.append({"time": float(t), "type": "yolo", "rms": None})

    points.sort(key=lambda p: p["time"])

    print(f"[Phase A] {len(points)} Trigger-Punkte gesammelt "
          f"({sum(1 for p in points if p['type']=='audio')} Audio, "
          f"{sum(1 for p in points if p['type']=='yolo')} YOLO).")

    return points


# ----------------------------------------------------------------------
# Phase B: Clustering
# ----------------------------------------------------------------------

def cluster_points(points: list[dict], cfg: Config) -> list[list[dict]]:
    """
    PHASE B: Gruppiert Punkte die zeitlich naeher als cluster_max_gap_sec
    beieinander liegen in gemeinsame Cluster (typ-unabhaengig).
    """
    if not points:
        return []

    clusters = [[points[0]]]

    for p in points[1:]:
        last_cluster = clusters[-1]
        if p["time"] - last_cluster[-1]["time"] <= cfg.cluster_max_gap_sec:
            last_cluster.append(p)
        else:
            clusters.append([p])

    print(f"[Phase B] {len(points)} Punkte zu {len(clusters)} Clustern gruppiert.")
    return clusters


# ----------------------------------------------------------------------
# Phase C: Fenster pro Cluster aufspannen
# ----------------------------------------------------------------------

def compute_cluster_window(
    cluster: list[dict],
    rms: np.ndarray,
    times: np.ndarray,
    cfg: Config,
) -> dict:
    """
    PHASE C: Berechnet das finale (start, end) Fenster fuer einen
    einzelnen Cluster, inkl. aller Boni. Gibt zusaetzlich eine
    Begruendung fuer das Logging zurueck.
    """
    first_point = cluster[0]
    last_point = cluster[-1]
    cluster_size = len(cluster)

    # --- Pre-Buffer abhaengig vom Typ des ERSTEN Punkts ---
    if first_point["type"] == "audio":
        pre_buffer = cfg.audio_pre_buffer_sec
        start_reason = f"Audio-Peak bei {fmt_time(first_point['time'])} (RMS={first_point['rms']:.4f})"
    else:
        pre_buffer = cfg.yolo_pre_buffer_sec
        start_reason = f"YOLO-Event bei {fmt_time(first_point['time'])}"

    raw_start = max(0.0, first_point["time"] - pre_buffer)

    # --- Isolierter Einzel-Punkt: kuerzeres Basis-Fenster ---
    is_isolated = cluster_size == 1

    if is_isolated and cfg.min_cluster_confidence_enabled:
        base_post = cfg.isolated_point_post_buffer_sec
        size_bonus = 0.0
        density_info = "isoliert (kein Cluster-Bonus)"
    else:
        base_post = cfg.base_post_buffer_sec

        # --- Groessen-Bonus ---
        extra_points = cluster_size - 1
        size_bonus = min(extra_points * cfg.size_bonus_per_point_sec, cfg.size_bonus_max_sec)

        # --- Dichte-Bonus (multipliziert den Groessen-Bonus) ---
        if cfg.density_bonus_enabled and extra_points > 0:
            gaps = [cluster[k+1]["time"] - cluster[k]["time"] for k in range(len(cluster)-1)]
            avg_gap = sum(gaps) / len(gaps)
            # je kleiner avg_gap relativ zur Referenz, desto hoeher der Multiplikator
            density_multiplier = min(
                cfg.density_reference_gap_sec / max(avg_gap, 0.1),
                cfg.density_bonus_max_multiplier,
            )
            density_multiplier = max(density_multiplier, 1.0)
            size_bonus = min(size_bonus * density_multiplier, cfg.size_bonus_max_sec)
            density_info = f"avg_gap={avg_gap:.2f}s, Dichte-Multiplikator={density_multiplier:.2f}x"
        else:
            density_info = "kein Dichte-Bonus (nur 1 Punkt oder deaktiviert)"

    raw_end = last_point["time"] + base_post + size_bonus

    # --- Lautstaerke-Bonus: iterativ auf dem FUSIONIERTEN Fenster berechnen ---
    loud_threshold = np.percentile(rms, cfg.loud_threshold_percentile)
    current_end = raw_end
    loud_bonus = 0.0

    for _ in range(5):  # wenige Iterationen reichen zur Stabilisierung
        mask = (times >= raw_start) & (times <= current_end)
        window_times = times[mask]
        window_rms = rms[mask]

        if len(window_times) == 0:
            break

        loud_mask = window_rms >= loud_threshold
        loud_duration = 0.0
        if np.any(loud_mask):
            # zusammenhaengende laute Zeit grob ueber Anteil * Fensterlaenge schaetzen
            loud_ratio = np.sum(loud_mask) / len(loud_mask)
            window_span = window_times[-1] - window_times[0] if len(window_times) > 1 else 0
            loud_duration = loud_ratio * window_span

        new_loud_bonus = min(loud_duration * cfg.loud_growth_per_sec, cfg.loud_growth_max_sec)

        if abs(new_loud_bonus - loud_bonus) < 0.05:
            loud_bonus = new_loud_bonus
            break

        loud_bonus = new_loud_bonus
        current_end = raw_end + loud_bonus

    final_end = min(raw_end + loud_bonus, first_point["time"] + cfg.max_clip_duration_sec)
    final_end = min(final_end, times[-1])

    end_reason = (f"letzter Punkt @ {fmt_time(last_point['time'])} + Basis {base_post:.1f}s "
                  f"+ Groessen-Bonus {size_bonus:.1f}s ({density_info}) "
                  f"+ Lautstaerke-Bonus {loud_bonus:.1f}s (P{cfg.loud_threshold_percentile} im Fenster)")

    return {
        "start": raw_start,
        "end": final_end,
        "cluster_size": cluster_size,
        "is_isolated": is_isolated,
        "start_reason": start_reason,
        "end_reason": end_reason,
        "point_types": [p["type"] for p in cluster],
    }


def build_windows_from_clusters(
    clusters: list[list[dict]],
    rms: np.ndarray,
    times: np.ndarray,
    cfg: Config,
) -> list[dict]:
    print(f"\n[Phase C] Berechne Fenster fuer {len(clusters)} Cluster ...")
    windows = []

    for idx, cluster in enumerate(clusters, start=1):
        w = compute_cluster_window(cluster, rms, times, cfg)

        if w["is_isolated"] and cfg.discard_isolated_audio_points and cluster[0]["type"] == "audio":
            print(f"  [Cluster #{idx}] VERWORFEN (isolierter Audio-Punkt, Rausch-Filter aktiv) "
                  f"@ {fmt_time(cluster[0]['time'])}")
            continue

        print(f"  [Cluster #{idx}] {len(cluster)} Punkt(e) [{', '.join(w['point_types'])}]")
        print(f"      Start : {fmt_time(w['start'])}  ({w['start_reason']})")
        print(f"      Ende  : {fmt_time(w['end'])}  ({w['end_reason']})")

        windows.append(w)

    return windows


# ----------------------------------------------------------------------
# Phase D: Merge ueberlappender Fenster
# ----------------------------------------------------------------------

def merge_windows(windows: list[dict], gap_sec: float) -> list[dict]:
    if not windows:
        return []

    windows = sorted(windows, key=lambda w: w["start"])
    merged = [dict(windows[0])]

    for w in windows[1:]:
        last = merged[-1]
        if w["start"] <= last["end"] + gap_sec:
            if w["end"] > last["end"]:
                last["end"] = w["end"]
                last["end_reason"] = w["end_reason"]
            last["start_reason"] = last["start_reason"] + " [mit nachfolgendem Cluster-Fenster verschmolzen]"
            last["cluster_size"] = last["cluster_size"] + w["cluster_size"]
            last["point_types"] = last["point_types"] + w["point_types"]
        else:
            merged.append(dict(w))

    print(f"[Phase D] {len(windows)} Fenster -> {len(merged)} nach Merge (gap={gap_sec:.1f}s).")
    return merged


# ----------------------------------------------------------------------
# Stufe 2: Interne Lücken entfernen (identisch zum linearen Skript)
# ----------------------------------------------------------------------

def remove_internal_gaps(
    clip: tuple[float, float],
    rms: np.ndarray,
    times: np.ndarray,
    cfg: Config,
    yolo_event_times: set,
) -> list[tuple[float, float]]:
    start, end = clip

    if not cfg.enable_internal_gap_removal:
        return [clip]

    silence_threshold = np.percentile(rms, cfg.internal_silence_threshold_percentile)

    mask = (times >= start) & (times <= end)
    clip_times = times[mask]
    clip_rms = rms[mask]

    if len(clip_times) == 0:
        return [clip]

    yolo_sorted = sorted(t for t in yolo_event_times if start <= t <= end)

    def has_yolo_in_range(a: float, b: float) -> bool:
        for t in yolo_sorted:
            if a <= t <= b:
                return True
        return False

    segments = []
    seg_start = start
    silence_start = None

    for k in range(len(clip_times)):
        t = clip_times[k]
        is_silent = clip_rms[k] < silence_threshold

        if is_silent:
            if silence_start is None:
                silence_start = t
        else:
            if silence_start is not None:
                silence_duration = t - silence_start
                if silence_duration >= cfg.internal_min_gap_sec and not has_yolo_in_range(silence_start, t):
                    seg_end = min(silence_start + cfg.internal_gap_padding_sec, t)
                    next_start = max(t - cfg.internal_gap_padding_sec, seg_end)

                    if seg_end > seg_start:
                        segments.append((seg_start, seg_end))
                        print(f"  [Stufe 2] Interne Luecke entfernt: {fmt_time(silence_start)} -> {fmt_time(t)} "
                              f"({silence_duration:.1f}s Stille) | Segment: {fmt_time(seg_start)} -> {fmt_time(seg_end)}")

                    seg_start = next_start
                silence_start = None

    if seg_start < end:
        segments.append((seg_start, end))

    if len(segments) > 1:
        print(f"  [Stufe 2] Clip in {len(segments)} Sub-Segmente zerlegt (statt 1 durchgehendem Clip)")

    return segments if segments else [clip]


# ----------------------------------------------------------------------
# Stufe 3: Mikro-Trimming (identisch zum linearen Skript)
# ----------------------------------------------------------------------

def micro_trim_edge(
    target_time: float,
    rms: np.ndarray,
    times: np.ndarray,
    cfg: Config,
) -> float:
    if not cfg.enable_micro_trim:
        return target_time

    window = cfg.micro_trim_search_window_sec
    mask = (times >= target_time - window) & (times <= target_time + window)

    candidates_t = times[mask]
    candidates_rms = rms[mask]

    if len(candidates_t) == 0:
        return target_time

    silence_threshold = np.percentile(rms, cfg.micro_trim_silence_percentile)
    quiet_mask = candidates_rms <= silence_threshold

    if np.any(quiet_mask):
        quiet_times = candidates_t[quiet_mask]
        quiet_rms = candidates_rms[quiet_mask]
        best_idx = np.argmin(quiet_rms)
        return float(quiet_times[best_idx])

    best_idx = np.argmin(candidates_rms)
    return float(candidates_t[best_idx])


def apply_micro_trim_to_segments(
    segments: list[tuple[float, float]],
    rms: np.ndarray,
    times: np.ndarray,
    cfg: Config,
) -> list[tuple[float, float]]:
    if not cfg.enable_micro_trim:
        return segments

    trimmed = []
    for start, end in segments:
        new_start = micro_trim_edge(start, rms, times, cfg)
        new_end = micro_trim_edge(end, rms, times, cfg)

        if new_end <= new_start:
            new_start, new_end = start, end

        if abs(new_start - start) > 0.01 or abs(new_end - end) > 0.01:
            print(f"  [Stufe 3] Mikro-Trim: ({fmt_time(start)} -> {fmt_time(end)}) "
                  f"=> ({fmt_time(new_start)} -> {fmt_time(new_end)})")

        trimmed.append((new_start, new_end))

    return trimmed


# ----------------------------------------------------------------------
# ffmpeg: Schneiden & Zusammenfügen
# ----------------------------------------------------------------------

def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

import subprocess
import json

def get_stream_maps(video_path: str) -> list[str]:
    """
    Ermittelt alle tatsächlich vorhandenen Streams (Video + Audio)
    und gibt die passenden -map Argumente zurück.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    video_indices = [s["index"] for s in streams if s["codec_type"] == "video"]
    audio_indices = [s["index"] for s in streams if s["codec_type"] == "audio"]

    maps = []
    for idx in video_indices:
        maps += ["-map", f"0:{idx}"]
    for idx in audio_indices:
        maps += ["-map", f"0:{idx}"]

    return maps

def cut_and_concat(
    video_path: str,
    segments: list[tuple[float, float]],
    output_path: str,
    cfg: Config,
    progress_callback=None,
) -> None:
    if not segments:
        print("Keine Segmente zum Schneiden.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_paths = []

        for i, (start, end) in enumerate(segments):
            clip_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
            duration = end - start

            stream_maps = get_stream_maps(video_path)

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                *stream_maps,
                "-avoid_negative_ts", "make_zero",
            ]

            if cfg.use_gpu_encoding:
                cmd += ["-c:v", "h264_nvenc", "-cq", str(cfg.nvenc_cq), "-preset", "p7"]
            else:
                cmd += ["-c:v", "libx264", "-crf", str(cfg.crf), "-preset", "slow"]

            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", clip_path]

            print(f"[Cut] Segment {i+1}/{len(segments)}: {fmt_time(start)} -> {fmt_time(end)} ({duration:.1f}s)")
            subprocess.run(cmd, check=True)
            subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-select_streams", "a", clip_path], check=True)
            clip_paths.append(clip_path)

            if progress_callback is not None:
                # Cutting zaehlt als 0-80% dieser Phase, Concat danach als 80-100%
                progress_callback("Schneiden", (i + 1) / len(segments) * 80)

        if progress_callback is not None:
            progress_callback("Zusammenfuegen", 85)

        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp.replace(chr(92), '/')}'\n")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        print("[Concat] Fuege Segmente zusammen ...")
        first_clip_maps = get_stream_maps(clip_paths[0])  # erster Zwischen-Clip als Referenz

        subprocess.run([
            "ffmpeg", "-y",
            "-fflags", "+genpts",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            *first_clip_maps,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            output_path,
        ], check=True)

    print(f"[Done] Gespeichert: {output_path}")
    if progress_callback is not None:
        progress_callback("Zusammenfuegen", 100.0)


# ----------------------------------------------------------------------
# Hauptablauf
# ----------------------------------------------------------------------

def process_video(video_path: str, output_path: str, cfg: Config, progress_callback=None) -> dict:
    """
    progress_callback(phase: str, percent: float) wird optional bei
    den wichtigsten Phasen aufgerufen:
      "YOLO-Analyse"   (0-100, pro Frame)
      "Audio-Analyse"  (0/100, da nicht granular messbar)
      "Schneiden"      (0-80, pro Segment)
      "Zusammenfuegen" (85/100)
    """
    import io, sys

    log_buffer = io.StringIO()
    original_stdout = sys.stdout

    class Tee:
        def write(self, msg):
            if original_stdout is not None:
                original_stdout.write(msg)
            log_buffer.write(msg)
        def flush(self):
            if original_stdout is not None:
                original_stdout.flush()

    # Tee nur aktivieren wenn stdout nicht schon von aussen ersetzt wurde
    # (z.B. durch den QueueWriter der GUI), damit kein NoneType-Fehler entsteht
    gui_already_capturing = not isinstance(sys.stdout, io.TextIOBase.__class__) and hasattr(sys.stdout, 'log_queue')
    if not gui_already_capturing:
        sys.stdout = Tee()

    timeline_rows = []
    final_timeline_rows = []
    result = {
    "yolo_event_times": [],
    "intro_metadata": [],
    "output_path": output_path,
    }
    try:
        duration = get_video_duration(video_path)
        print(f"Video: {video_path}")
        print(f"Video-Laenge: {fmt_time(duration)}")

        yolo_event_times = preload_yolo_events(video_path, cfg, progress_callback)
        result["yolo_event_times"] = yolo_event_times
        if progress_callback is not None:
            progress_callback("Audio-Analyse", 0.0)
        rms, times = compute_rms(video_path)
        if progress_callback is not None:
            progress_callback("Audio-Analyse", 100.0)

        # --- Phase A: Punkte sammeln ---
        points = collect_trigger_points(rms, times, cfg, yolo_event_times)

        if not points:
            print("Keine Trigger-Punkte gefunden.")
            sys.stdout = original_stdout
            return result

        # --- Phase B: Clustering ---
        clusters = cluster_points(points, cfg)

        # --- Phase C: Fenster pro Cluster ---
        windows = build_windows_from_clusters(clusters, rms, times, cfg)

        if not windows:
            print("Keine Fenster nach Filterung uebrig.")
            sys.stdout = original_stdout
            return result

        # --- Phase D: Merge ---
        merged_windows = merge_windows(windows, cfg.merge_gap_sec)

        # --- Stufe 2: interne Luecken pro Fenster entfernen ---
        all_segments = []
        segment_origin = []

        for w_idx, w in enumerate(merged_windows, start=1):
            clip = (w["start"], w["end"])
            sub_segments = remove_internal_gaps(clip, rms, times, cfg, yolo_event_times)

            phase_reason = (f"Cluster-Groesse: {w['cluster_size']} Punkt(e) [{', '.join(w['point_types'])}] | "
                             f"Start-Trigger: {w['start_reason']} | "
                             f"End-Berechnung: {w['end_reason']}")

            for sub_idx, seg in enumerate(sub_segments, start=1):
                all_segments.append(seg)
                if len(sub_segments) > 1:
                    origin = (f"Fenster #{w_idx} ({fmt_time(clip[0])}-{fmt_time(clip[1])}), "
                               f"Teil {sub_idx}/{len(sub_segments)} (Stufe 2 gesplittet) || {phase_reason}")
                else:
                    origin = f"Fenster #{w_idx} ({fmt_time(clip[0])}-{fmt_time(clip[1])}) || {phase_reason}"
                segment_origin.append(origin)

        print(f"\n[Stufe 2] Insgesamt {len(all_segments)} Segmente nach Luecken-Entfernung "
              f"(vorher {len(merged_windows)} Fenster).")

        # --- Stufe 3: Mikro-Trimming ---
        pre_trim_segments = list(all_segments)
        all_segments = apply_micro_trim_to_segments(all_segments, rms, times, cfg)

        combined = [
            (s, e, origin, pre_s, pre_e)
            for (s, e), origin, (pre_s, pre_e) in zip(all_segments, segment_origin, pre_trim_segments)
            if e > s
        ]
        combined.sort(key=lambda x: x[0])

        all_segments = [(s, e) for s, e, _, _, _ in combined]

        print(f"\nFinale Segmente ({len(all_segments)}):")
        total = 0
        clip_nr = 0
        for s, e, origin, _, _ in combined:
            clip_nr += 1
            print(f"  #{clip_nr:03d}  {fmt_time(s)} -> {fmt_time(e)}  ({e-s:.1f}s)  [{origin}]")
            total += e - s
            timeline_rows.append((clip_nr, s, e, e - s))
        print(f"Gesamt-Highlight-Laenge: {fmt_time(total)}\n")

        cut_and_concat(video_path, all_segments, output_path, cfg, progress_callback)

        # --- Finale Timeline berechnen (Zeitstempel IM Output-Video) ---
        cumulative = 0.0

        intro_metadata = []

        for clip_nr, (s, e, origin, pre_s, pre_e) in enumerate(combined, start=1):
            seg_duration = e - s

            final_start = cumulative
            final_end = cumulative + seg_duration

            reasons = [f"Quelle: {fmt_time(s)} -> {fmt_time(e)} im Originalvideo"]
            reasons.append(origin)

            if abs(s - pre_s) > 0.01 or abs(e - pre_e) > 0.01:
                reasons.append(
                    f"Stufe 3 Mikro-Trim angepasst (vorher {fmt_time(pre_s)} -> {fmt_time(pre_e)})"
                )

            final_timeline_rows.append(
                (
                    clip_nr,
                    final_start,
                    final_end,
                    seg_duration,
                    " | ".join(reasons),
                )
            )

            cluster_size = 1
            point_types = []

            try:
                if "Cluster-Groesse:" in origin:
                    part = origin.split("Cluster-Groesse:")[1]

                    cluster_size = int(
                        part.split("Punkt")[0].strip()
                    )

                    if "[" in part and "]" in part:
                        point_types = (
                            part.split("[")[1]
                                .split("]")[0]
                                .replace(" ", "")
                                .split(",")
                        )
            except Exception:
                pass

            yolo_count = point_types.count("yolo")
            audio_count = point_types.count("audio")

            score = (
                cluster_size * 1.5
                + yolo_count * 3.0
                + audio_count * 0.25
                + seg_duration * 0.2
            )

            intro_metadata.append(
                {
                    "clip_nr": clip_nr,

                    # Position im fertigen Highlight
                    "start_in_output": round(final_start, 3),
                    "end_in_output": round(final_end, 3),

                    # Position im Originalvideo
                    "source_start": round(s, 3),
                    "source_end": round(e, 3),

                    "duration": round(seg_duration, 3),

                    "cluster_size": cluster_size,
                    "point_types": point_types,

                    "yolo_count": yolo_count,
                    "audio_count": audio_count,

                    "score": round(score, 3),

                    "origin": origin,
                }
            )

            cumulative = final_end
            result["intro_metadata"] = intro_metadata
    finally:
        if not gui_already_capturing:
            sys.stdout = original_stdout

    log_path = output_path.rsplit(".", 1)[0] + "_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    print(f"[Log] Gespeichert: {log_path}")

    table_path = output_path.rsplit(".", 1)[0] + "_timeline.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"{'Clip':<6}{'Start':<18}{'Ende':<18}{'Dauer':<10}\n")
        f.write("-" * 52 + "\n")
        for nr, s, e, d in timeline_rows:
            start_str = fmt_time(s).split(" ")[0]
            end_str = fmt_time(e).split(" ")[0]
            f.write(f"{nr:<6}{start_str:<18}{end_str:<18}{d:.1f}s\n")
    print(f"[Log] Timeline-Tabelle gespeichert: {table_path}")

    final_table_path = output_path.rsplit(".", 1)[0] + "_final_timeline.txt"
    with open(final_table_path, "w", encoding="utf-8") as f:
        f.write("Finale Timeline - Zeitstempel im FERTIGEN Output-Video\n")
        f.write("=" * 90 + "\n\n")
        for nr, fs, fe, d, reason in final_timeline_rows:
            start_str = fmt_time(fs).split(" ")[0]
            end_str = fmt_time(fe).split(" ")[0]
            f.write(f"Clip #{nr:03d}  |  {start_str} -> {end_str}  ({d:.1f}s)\n")

            parts = reason.split(" || ")
            for part in parts:
                sub_parts = part.split(" | ")
                for sp in sub_parts:
                    f.write(f"    - {sp.strip()}\n")
            f.write("\n")
    print(f"[Log] Finale Output-Timeline gespeichert: {final_table_path}")
    json_path = output_path.rsplit(".", 1)[0] + "_metadata.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            intro_metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"[Log] Intro-Metadaten gespeichert: {json_path}")
    return result
# ----------------------------------------------------------------------
# Aufruf
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    config = Config(
        yolo_model_path=r"D:\YOLO_Training\runs\hitmarkerEvents\weights\best.pt",
    )

    # 1. Basis-Verzeichnis für deine fertigen Schnitte
    base_output_dir = r"D:\EDITED CLIPS\Skript Cutted"
    
    # 2. Präziser Zeitstempel inklusive Sekunden und Millisekunden (%f)
    # Erzeugt z. B.: "2026-06-20_12-45-30_102" (Jahr-Monat-Tag_Stunde-Minute-Sekunde_Millisekunde)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    
    # 3. Den neuen, absolut einzigartigen Unterordner definieren und erstellen
    target_folder = os.path.join(base_output_dir, f"Render_{timestamp}")
    os.makedirs(target_folder, exist_ok=True)
    
    # 4. Auch die Videodatei bekommt den Zeitstempel im Namen
    video_filename = f"Highlight_Video_{timestamp}.mp4"
    dynamic_output_path = os.path.join(target_folder, video_filename)

    # Start für das Einzelvideo
    process_video(
        video_path=r"D:\OBS CLIPS\Alle Aufnahmen\2026-06-27 09-06-15.mkv",
        output_path=dynamic_output_path,
        cfg=config,
    )

    # --- Mehrere Videos parallel (falls du das mal nutzt) ---
    # from concurrent.futures import ProcessPoolExecutor
    #
    # videos = [r"D:\OBS CLIPS\video1.mkv", r"D:\OBS CLIPS\video2.mkv"]
    #
    # def worker(path):
    #     # Für parallele Videos nutzen wir den Originalnamen + Zeitstempel
    #     orig_name = os.path.splitext(os.path.basename(path))[0]
    #     video_name = f"{orig_name}_highlight_{timestamp}.mp4"
    #     out = os.path.join(target_folder, video_name)
    #     process_video(path, out, config)
    #
    # with ProcessPoolExecutor(max_workers=3) as executor:
    #     executor.map(worker, videos)