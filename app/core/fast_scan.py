"""
Schnellere Event-Erkennung fuer 'Daten sammeln'.

Fuer das Sammeln von Trainingsdaten werden nur einige hundert Event-Frames gebraucht, nicht jedes Event
des Videos. Statt das ganze Video zu scannen, wird es in gleichmaessig verteilten Abschnitten geprueft
(grob -> fein) und der Scan stoppt, sobald genug Events gefunden sind. Innerhalb eines Abschnitts ist die
Pruefung identisch zum normalen Scan (jeder N-te Frame).

Dekodiert wird wahlweise auf der CPU (OpenCV), auf der GPU (NVDEC ueber den ffmpeg-cuvid-Dekoder)
oder auf beiden gleichzeitig. Faellt die GPU aus, uebernimmt automatisch die CPU.
"""

import collections
import json
import queue
import re
import subprocess
import threading
import time

import numpy as np

PROBE_WIDTH = 640  # Breite der verkleinerten Frames bei GPU/Hybrid (YOLO arbeitet ohnehin mit ~640 px)
BATCH_SIZE = 16
DECODE_MODES = ("cpu", "hybrid", "gpu")
# bicubic liefert auf der GPU die Erkennung, die der bisherigen (CPU, volle Aufloesung) am naechsten kommt:
# an deinem Testvideo ~87 % der Treffer bei gleicher Schwelle (bilinear 75 %, nearest 61 %)
GPU_INTERP = "bicubic"
_PTS_RE = re.compile(r"pts_time:\s*(-?[\d.]+)")


def spread_indices(n: int) -> list[int]:
    """0..n-1 so geordnet, dass jedes Praefix der Liste gleichmaessig ueber den Bereich verteilt ist."""
    if n <= 0:
        return []
    bits = max(1, (n - 1).bit_length())
    order = [int(format(i, f"0{bits}b")[::-1], 2) for i in range(1 << bits)]
    return [i for i in order if i < n]


def make_chunks(duration: float, chunk_sec: float) -> list[list[float]]:
    """Teilt das Video in Abschnitte und ordnet sie so, dass frueh gepruefte Abschnitte gleichmaessig streuen."""
    count = max(1, int(-(-duration // chunk_sec)))
    chunks = [[i * chunk_sec, min(duration, (i + 1) * chunk_sec)] for i in range(count)]
    return [chunks[i] for i in spread_indices(count)]


def count_groups(event_times: list[float], gap_sec: float) -> int:
    """Anzahl getrennter Events (Treffer mit Abstand <= gap_sec zaehlen als ein Event)."""
    groups, last = 0, None
    for t in sorted(event_times):
        if last is None or t - last > gap_sec:
            groups += 1
        last = t
    return groups


# ----------------------------------------------------------------------
# Video-Infos / GPU-Verfuegbarkeit
# ----------------------------------------------------------------------

def probe_video(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height", "-show_entries", "format=duration",
        "-of", "json", str(video_path),
    ]
    out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    stream = out["streams"][0]
    return {
        "codec": stream.get("codec_name", ""),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(out["format"]["duration"]),
    }


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def _out_size(info: dict) -> tuple[int, int]:
    out_w = _even(PROBE_WIDTH if info["width"] > PROBE_WIDTH else info["width"])
    return out_w, _even(info["height"] * out_w / info["width"])


def _gpu_filter(every_n: int, out_w: int, out_h: int, extra: str = "") -> str:
    # Auswahl jedes N-ten Frames und Verkleinern passieren auf der GPU; nur kleine Frames werden heruntergeladen
    return (f"select=not(mod(n\\,{every_n})),scale_cuda={out_w}:{out_h}:interp_algo={GPU_INTERP},"
            f"hwdownload,format=nv12{extra}")


def _gpu_input_args() -> list[str]:
    # -threads 1: sonst scheitert der NVDEC-Start bei vielen CPU-Threads ("decode surfaces")
    return ["-threads", "1", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]


def gpu_decode_check(video_path: str, info: dict | None = None) -> tuple[bool, str]:
    """
    Prueft mit einem 1-Sekunden-Test, ob dieses Video samt Verkleinern auf der GPU verarbeitet werden kann.
    Gibt (True, "ok") oder (False, Grund) zurueck.
    """
    try:
        info = info or probe_video(video_path)
        out_w, out_h = _out_size(info)
    except Exception as e:
        return False, f"Video konnte nicht gelesen werden ({e})"
    cmd = ["ffmpeg", "-hide_banner", "-v", "error"] + _gpu_input_args() + [
        "-t", "1", "-i", str(video_path), "-vf", _gpu_filter(1, out_w, out_h), "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"ffmpeg-Test fehlgeschlagen ({e})"
    if r.returncode != 0 or r.stderr.strip():
        reason = (r.stderr.strip().splitlines() or ["unbekannter Fehler"])[0][:160]
        return False, f"GPU-Dekodierung nicht nutzbar ({info.get('codec', '?')}): {reason}"
    return True, "ok"


# ----------------------------------------------------------------------
# Hilfsklasse: YOLO-Auswertung (immer nur ein Thread ruft das Modell auf)
# ----------------------------------------------------------------------

class _Detector:
    def __init__(self, model, conf, target_classes):
        self.model = model
        self.conf = conf
        self.wanted = set(target_classes) if target_classes else None
        self.names = getattr(model, "names", {}) or {}
        self.events: set[float] = set()

    def run(self, items: list[tuple[float, np.ndarray]]):
        if not items:
            return
        results = self.model.predict([f for _, f in items], conf=self.conf, verbose=False)
        for (t, _), r in zip(items, results):
            for box in r.boxes:
                name = str(self.names.get(int(box.cls[0]), int(box.cls[0])))
                if self.wanted is None or name in self.wanted:
                    self.events.add(round(t, 2))
                    break


# ----------------------------------------------------------------------
# Nur CPU (unveraendertes Verhalten: volle Aufloesung direkt an YOLO)
# ----------------------------------------------------------------------

def scan_windows(
    video_path: str,
    model,
    windows: list[list[float]],
    conf: float,
    target_classes=None,
    every_n: int = 5,
    stop_check=None,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    """Prueft die Zeitfenster der Reihe nach dicht auf der CPU. stop_check(events) -> True beendet den Scan."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geladen werden: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = _Detector(model, conf, target_classes)
    batch: list = []
    windows_done = 0
    scanned_frames = 0

    def flush():
        det.run(batch)
        batch.clear()

    try:
        for a, b in windows:
            start = int(a * fps)
            start -= start % every_n  # gleiche Raster-Frames wie beim normalen Scan
            end = int(b * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            idx = start
            while idx < end:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Vorgang abgebrochen.")
                # nicht benoetigte Frames nur dekodieren (grab), nicht umwandeln (retrieve)
                if idx % every_n == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    batch.append((idx / fps, frame))
                    if len(batch) >= BATCH_SIZE:
                        flush()
                elif not cap.grab():
                    break
                idx += 1
                scanned_frames += 1
            flush()
            windows_done += 1
            if progress_callback is not None:
                progress_callback("Event-Scan", windows_done / len(windows) * 100)
            if stop_check is not None and stop_check(det.events):
                break
    finally:
        cap.release()

    return {"events": sorted(det.events), "windows_done": windows_done, "scanned_sec": scanned_frames / fps,
            "gpu_chunks": 0, "cpu_chunks": windows_done}


# ----------------------------------------------------------------------
# GPU / Hybrid: mehrere Dekodierer fuettern einen YOLO-Verbraucher
# ----------------------------------------------------------------------

def _gpu_chunk(video_path, a, b, out_w, out_h, every_n, emit, should_stop):
    """Dekodiert und verkleinert einen Abschnitt auf der GPU (ffmpeg); emit(items) bekommt Pakete aus (Zeit, Frame)."""
    frame_bytes = out_w * out_h * 3
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", *_gpu_input_args(),
        "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(video_path), "-map", "0:v:0",
        "-vf", _gpu_filter(every_n, out_w, out_h, ",showinfo"),
        "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)
    stamps: list[float] = []
    tail: collections.deque = collections.deque(maxlen=6)
    lock = threading.Lock()

    def drain():
        for raw in proc.stderr:  # showinfo schreibt pro Frame eine Zeile mit der Zeit (relativ zum Abschnittsstart)
            line = raw.decode("utf-8", errors="ignore")
            m = _PTS_RE.search(line)
            with lock:
                if m and "showinfo" in line:
                    stamps.append(float(m.group(1)))
                elif line.strip():
                    tail.append(line.strip())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    def stamp(idx):
        deadline = time.time() + 15
        while time.time() < deadline:
            with lock:
                if len(stamps) > idx:
                    return stamps[idx]
            if proc.poll() is not None:
                reader.join(timeout=1)
                with lock:
                    return stamps[idx] if len(stamps) > idx else None
            time.sleep(0.003)
        return None

    items, n = [], 0
    try:
        while not should_stop():
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            t = stamp(n)
            if t is None:
                raise RuntimeError("ffmpeg lieferte keinen Zeitstempel")
            items.append((a + t, np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3).copy()))
            n += 1
            if len(items) >= BATCH_SIZE:
                emit(items)
                items = []
        if items and not should_stop():
            emit(items)
        stopped = should_stop()
        if stopped and proc.poll() is None:
            proc.kill()  # frueher Stopp: nicht auf das Ende des Abschnitts warten
        proc.wait(timeout=30)
        if not stopped and (proc.returncode != 0 or (n == 0 and (b - a) > 1.0)):
            with lock:
                msg = " | ".join(tail)
            raise RuntimeError(f"ffmpeg-GPU-Dekodierung fehlgeschlagen (Code {proc.returncode}): {msg[:200]}")
    finally:
        if proc.poll() is None:
            proc.kill()
        reader.join(timeout=2)


def _cpu_chunk(cap, fps, a, b, out_w, out_h, every_n, emit, should_stop):
    """Dekodiert einen Abschnitt per OpenCV und verkleinert die Frames auf dieselbe Groesse wie die GPU-Variante."""
    import cv2

    start = int(a * fps)
    start -= start % every_n
    end = int(b * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    items, idx = [], start
    while idx < end and not should_stop():
        if idx % every_n == 0:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                # INTER_LINEAR = genau das Verfahren, mit dem YOLO intern verkleinert (Erkennung bleibt identisch)
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            items.append((idx / fps, frame))
            if len(items) >= BATCH_SIZE:
                emit(items)
                items = []
        elif not cap.grab():
            break
        idx += 1
    if items and not should_stop():
        emit(items)


def scan_parallel(
    video_path: str,
    model,
    windows: list[list[float]],
    conf: float,
    info: dict,
    use_gpu: bool,
    use_cpu: bool,
    target_classes=None,
    every_n: int = 5,
    stop_check=None,
    progress_callback=None,
    cancel_event=None,
    log=print,
) -> dict:
    """
    Verteilt die Abschnitte auf GPU- und/oder CPU-Dekodierer. Ein einzelner Verbraucher fuehrt YOLO aus.
    Faellt der GPU-Dekodierer aus, wird der Abschnitt zurueckgegeben und die CPU uebernimmt.
    """
    import cv2

    out_w, out_h = _out_size(info)

    pending = collections.deque(enumerate(windows))
    pending_lock = threading.Lock()
    stop = threading.Event()
    consumer_alive = threading.Event()
    q: queue.Queue = queue.Queue(maxsize=8)
    state = {"gpu_chunks": 0, "cpu_chunks": 0, "errors": [], "gpu_failed": False}
    state_lock = threading.Lock()

    def take():
        with pending_lock:
            return pending.popleft() if pending else None

    def give_back(item):
        with pending_lock:
            pending.appendleft(item)

    def put(msg):
        while not stop.is_set():
            try:
                q.put(msg, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def worker(kind):
        cap, fps = None, 30.0
        try:
            while not stop.is_set():
                item = take()
                if item is None:
                    break
                cid, (a, b) = item
                emit = lambda items, cid=cid: put(("frames", cid, items))
                try:
                    if kind == "gpu":
                        _gpu_chunk(video_path, a, b, out_w, out_h, every_n, emit, stop.is_set)
                    else:
                        if cap is None:
                            cap = cv2.VideoCapture(str(video_path))
                            if not cap.isOpened():
                                raise RuntimeError(f"Video konnte nicht geladen werden: {video_path}")
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        _cpu_chunk(cap, fps, a, b, out_w, out_h, every_n, emit, stop.is_set)
                except Exception as e:
                    if kind != "gpu":
                        raise
                    # GPU faellt aus: Abschnitt zurueckgeben und in diesem Thread auf CPU weiterarbeiten
                    give_back(item)
                    log(f"[ML] GPU-Dekodierung fehlgeschlagen, CPU uebernimmt: {e}")
                    with state_lock:
                        state["gpu_failed"] = True
                    kind = "cpu"
                    continue
                if not stop.is_set():
                    put(("done", cid, kind, b - a))
        except Exception as e:
            with state_lock:
                state["errors"].append(e)
            stop.set()
        finally:
            if cap is not None:
                cap.release()
            # Exit-Meldung nur zustellen, solange der Verbraucher noch liest
            while consumer_alive.is_set():
                try:
                    q.put(("exit",), timeout=0.2)
                    break
                except queue.Full:
                    continue

    det = _Detector(model, conf, target_classes)
    kinds = (["gpu"] if use_gpu else []) + (["cpu"] if (use_cpu or not use_gpu) else [])
    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in kinds]
    exited, windows_done, scanned_sec = 0, 0, 0.0

    consumer_alive.set()
    for t in threads:
        t.start()

    try:
        while exited < len(threads):
            if cancel_event is not None and cancel_event.is_set():
                stop.set()
                raise RuntimeError("Vorgang abgebrochen.")
            try:
                msg = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg[0] == "exit":
                exited += 1
            elif stop.is_set():
                continue  # nach dem Stopp keine weiteren Frames mehr auswerten
            elif msg[0] == "frames":
                det.run(msg[2])
            elif msg[0] == "done":
                windows_done += 1
                scanned_sec += msg[3]
                with state_lock:
                    state["gpu_chunks" if msg[2] == "gpu" else "cpu_chunks"] += 1
                if progress_callback is not None:
                    progress_callback("Event-Scan", windows_done / len(windows) * 100)
                if stop_check is not None and stop_check(det.events):
                    stop.set()
    finally:
        stop.set()
        consumer_alive.clear()
        for t in threads:
            t.join(timeout=10)

    if state["errors"]:
        raise state["errors"][0]
    return {"events": sorted(det.events), "windows_done": windows_done, "scanned_sec": scanned_sec,
            "gpu_chunks": state["gpu_chunks"], "cpu_chunks": state["cpu_chunks"],
            "gpu_failed": state["gpu_failed"]}


def collect_events(
    video_path: str,
    model_path: str,
    conf: float,
    target_classes=None,
    every_n: int = 5,
    need_groups: int | None = None,
    group_gap_sec: float = 0.5,
    chunk_sec: float = 30.0,
    decode_mode: str = "cpu",
    progress_callback=None,
    cancel_event=None,
    log=print,
) -> dict:
    """
    Sucht Events fuer 'Daten sammeln'. need_groups=None prueft das ganze Video, sonst stoppt der Scan,
    sobald mindestens so viele getrennte Events gefunden wurden.
    decode_mode: "cpu" | "hybrid" (GPU + CPU) | "gpu" (nur GPU). Ist die GPU nicht nutzbar, wird CPU verwendet.
    """
    import cv2
    import torch
    from ultralytics import YOLO

    if decode_mode not in DECODE_MODES:
        decode_mode = "cpu"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration = total_frames / fps if fps > 0 else 0.0
    if duration <= 0:
        raise RuntimeError(f"Videolaenge konnte nicht bestimmt werden: {video_path}")

    gpu_ok, gpu_note = False, ""
    used_mode = "cpu"
    if decode_mode != "cpu":
        info = probe_video(video_path)
        gpu_ok, gpu_note = gpu_decode_check(video_path, info)
        if not gpu_ok:
            log(f"[ML] GPU-Dekodierung nicht verfuegbar, nutze CPU: {gpu_note}")
        else:
            used_mode = decode_mode

    model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    chunks = make_chunks(duration, chunk_sec)
    stop = (lambda ev: count_groups(list(ev), group_gap_sec) >= need_groups) if need_groups else None

    t0 = time.time()
    if used_mode == "cpu":
        result = scan_windows(video_path, model, chunks, conf, target_classes, every_n,
                              stop_check=stop, progress_callback=progress_callback, cancel_event=cancel_event)
    else:
        result = scan_parallel(video_path, model, chunks, conf, info, True, used_mode == "hybrid",
                               target_classes, every_n, stop_check=stop, progress_callback=progress_callback,
                               cancel_event=cancel_event, log=log)
    result.update({
        "duration": duration,
        "chunks_total": len(chunks),
        "seconds": time.time() - t0,
        "stopped_early": result["windows_done"] < len(chunks),
        "decode_mode": used_mode,
        "decode_note": gpu_note if decode_mode != "cpu" and not gpu_ok else "",
    })
    return result
