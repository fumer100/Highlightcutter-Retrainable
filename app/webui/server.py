"""
Lokaler Flask-Server als Ersatz fuer die pywebview-Variante (app/webui/api.py +
main.py), die auf manchen Windows-Systemen wegen eines pythonnet/WinForms-Bugs
abstuerzt ("window.native...: maximum recursion depth exceeded"). Hier laeuft
ein ganz normaler lokaler Webserver, den man im normalen Browser oeffnet.

Start: python app/webui/server.py  ->  http://127.0.0.1:8765 im Browser oeffnen.

Die Business-Logik (Schneiden, Pseudo-Labeling, Training, Dataset-Verwaltung)
ist identisch zu app/webui/api.py, nur die Uebertragung ist jetzt HTTP/JSON +
Server-Sent-Events statt der pywebview-JS-Bruecke.
"""

import dataclasses
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog
import webbrowser
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from app.cutter.highlight_cutter_pointbased import (
    Config, export_fcpxml_timeline, preload_yolo_events, process_video,
)
from app.core.pipeline_controller import (
    COLLECT_EVENT_SHARE, COLLECT_MIN_CONFIDENCE, PipelineController, split_quota, video_duration,
)
from app.core.bootstrap import BOOTSTRAP_FRAME_TARGET, FIRST_MODEL_MIN_LABELED, extract_bootstrap_frames
from app.core.adaptive_thresholds import resolve_ml_params
from app.core.fast_scan import collect_events
from app.core.model_tester import test_model_on_video
from app.dataset.dataset_manager import DatasetManager
from app.extractor.frame_extractor import FrameExtractor
from app.training.trainer import Trainer
from app.training.training_history import append_history, build_training_overview, compute_dataset_snapshot
from config.game_settings import get_game_settings, pop_game_settings, save_game_settings
from config.games import (
    GAMES,
    GAMES_CONFIG_FILE,
    PROJECT_ROOT,
    GamesConfigError,
    reload_games,
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
OUTPUT_DIR = r"D:\EDITED CLIPS\Skript Cutted\Fertig zur Finalisierung"

SETTINGS_SCHEMA = [
    ("ignore_audio", "Nur YOLO-Events (Audio ignorieren)", "bool", "Audio wird bei der Clip-Erkennung gar nicht beachtet: keine Audio-Peaks, kein Lautstaerke-Bonus, keine Stille-Luecken (Stufe 2) und kein Mikro-Trim (Stufe 3). Nur YOLO-Events bestimmen die Clips. Die Audio-Einstellungen darunter greifen dann nicht"),
    ("audio_threshold_percentile", "Audio Peak-Threshold Percentile", "float", "Ab welchem RMS-Percentile ein Audio-Punkt zaehlt (hoch = nur sehr laute Stellen)"),
    ("cluster_max_gap_sec", "Cluster Max. Abstand (s)", "float", "Max. Abstand (s) zwischen Punkten im selben Cluster"),
    ("audio_pre_buffer_sec", "Audio Pre-Buffer (s)", "float", "Puffer VOR einem Audio-Peak-Cluster"),
    ("yolo_pre_buffer_sec", "YOLO Pre-Buffer (s)", "float", "Puffer VOR einem YOLO-Event-Cluster"),
    ("base_post_buffer_sec", "Basis Post-Buffer (s)", "float", "Basis-Nachlaufzeit nach letztem Punkt im Cluster"),
    ("loud_threshold_percentile", "Lautstaerke Threshold Percentile", "float", "Ab wann ein Sample im Fenster als 'laut' gilt (fuer Post-Buffer-Wachstum)"),
    ("loud_growth_per_sec", "Lautstaerke Wachstum pro Sek.", "float", "Post-Buffer-Zuwachs pro Sekunde lauter Zeit im Fenster"),
    ("loud_growth_max_sec", "Lautstaerke Wachstum Max. (s)", "float", "Kappung des lautstaerke-bedingten Post-Buffer-Zuwachses"),
    ("size_bonus_per_point_sec", "Cluster-Groessen-Bonus pro Punkt (s)", "float", "Bonus-Sekunden pro zusaetzlichem Punkt im Cluster"),
    ("size_bonus_max_sec", "Cluster-Groessen-Bonus Max. (s)", "float", "Kappung des Groessen-Bonus"),
    ("density_bonus_enabled", "Dichte-Bonus aktiv", "bool", "Ob dichter beieinanderliegende Punkte einen staerkeren Bonus bekommen"),
    ("density_reference_gap_sec", "Dichte Referenz-Abstand (s)", "float", "Bei diesem Abstand = Bonus-Faktor 1.0"),
    ("density_bonus_max_multiplier", "Dichte Max. Multiplikator", "float", "Maximaler Multiplikator auf den Groessen-Bonus bei sehr dichten Clustern"),
    ("min_cluster_confidence_enabled", "Rausch-Filter aktiv", "bool", "Ob isolierte Einzel-Punkte ein kuerzeres Fenster bekommen"),
    ("isolated_point_post_buffer_sec", "Einzel-Punkt Post-Buffer (s)", "float", "Post-Buffer fuer isolierte Einzel-Punkte (wenn Rausch-Filter aktiv)"),
    ("discard_isolated_audio_points", "Isolierte Audio-Punkte verwerfen", "bool", "Isolierte Audio-Einzel-Punkte komplett verwerfen"),
    ("max_clip_duration_sec", "Max. Clip-Laenge (s)", "float", "Sicherheitslimit: maximale Laenge eines einzelnen Clips"),
    ("enable_internal_gap_removal", "Stufe 2: Luecken entfernen", "bool", "Lange Stille-Abschnitte innerhalb eines Clips herausschneiden"),
    ("internal_silence_threshold_percentile", "Stufe 2: Stille-Threshold Percentile", "float", "Unter diesem Percentile gilt ein Sample als 'still'"),
    ("internal_min_gap_sec", "Stufe 2: Min. Luecken-Laenge (s)", "float", "Mindest-Stilledauer bevor intern geschnitten wird"),
    ("internal_gap_padding_sec", "Stufe 2: Luecken-Padding (s)", "float", "Puffer vor/nach der internen Luecke der erhalten bleibt"),
    ("enable_micro_trim", "Stufe 3: Mikro-Trimming aktiv", "bool", "Schnittkanten auf naechste leise Stelle snappen"),
    ("micro_trim_search_window_sec", "Stufe 3: Trim Suchfenster (s)", "float", "Wie weit links/rechts nach einer leisen Stelle gesucht wird"),
    ("micro_trim_silence_percentile", "Stufe 3: Trim Stille-Percentile", "float", "Unter diesem Percentile gilt eine Stelle als 'leise' zum Snappen"),
    ("merge_gap_sec", "Merge: Max. Abstand (s)", "float", "Cluster-Fenster die naeher beieinander liegen werden zusammengefuehrt"),
    ("use_gpu_encoding", "GPU-Encoding (NVENC)", "bool", "RTX-GPU fuer schnelles Encoding nutzen"),
    ("nvenc_cq", "NVENC Qualitaet (CQ, niedriger=besser)", "int", "Qualitaets-Wert fuer GPU-Encoding (15=sehr gut, 19=gut)"),
    ("crf", "CPU CRF (niedriger=besser)", "int", "Qualitaets-Wert fuer CPU-Encoding (14=sehr gut, 18=gut)"),
    ("yolo_sample_every_n_frames", "YOLO: jeden N-ten Frame pruefen", "int", "Performanz: nur jeden N-ten Frame durch YOLO analysieren"),
    ("yolo_confidence", "YOLO: Mindest-Konfidenz", "float", "Mindest-Erkennungssicherheit fuer YOLO-Treffer (0.0-1.0)"),
    ("yolo_batch_size", "YOLO: Batch-Groesse", "int", "Wie viele Sample-Frames pro GPU-Forward-Pass gebuendelt werden (hoeher = schneller, mehr VRAM)"),
    ("max_parallel_cuts", "Schneiden: parallele Segmente", "int", "Wie viele Segmente gleichzeitig per ffmpeg geschnitten werden (NVENC-Sessions begrenzen!)"),
    ("scoring_mode", "Wertigkeits-Scoring", "choice", "legacy = nur Lautstaerke-Heuristik, llm = LLM entscheidet rein/raus"),
    ("min_worthiness_score", "Mindest-Wertigkeitsscore (0-100)", "float", "Fenster mit LLM-Score darunter werden verworfen (nur bei scoring_mode=llm)"),
    ("llm_provider", "LLM-Anbieter", "choice", "ollama = komplett lokal (empfohlen, kein API-Key), openai = Cloud (braucht OPENAI_API_KEY)"),
    ("llm_model", "LLM-Modell", "text", "Bei Ollama z.B. 'llama3.1:8b' (muss vorher per 'ollama pull' geladen sein), bei OpenAI z.B. 'gpt-4o-mini'"),
    ("ollama_base_url", "Ollama-Server-URL", "text", "Nur relevant bei llm_provider=ollama, Standard: http://localhost:11434/v1"),
    ("llm_batch_size", "LLM: Fenster pro Anfrage", "int", "Wie viele Kandidaten-Fenster pro LLM-Aufruf gebuendelt werden (kleiner = zuverlaessiger bei lokalen Modellen)"),
    ("enable_speech_to_text", "Speech-to-Text aktiv", "bool", "Audiospur transkribieren, damit das LLM Humor/Kommentare mitbewerten kann"),
    ("whisper_model_size", "Whisper-Modellgroesse", "choice", "Groesser = genauer aber langsamer (tiny/base/small/medium/large-v3)"),
    ("whisper_language", "Whisper-Sprache (leer = Auto)", "text", "Sprachcode wie 'de' oder 'en', leer lassen fuer automatische Erkennung"),
    ("ml_auto_train_threshold", "ML: Auto-Training ab Konfidenz", "float", "Bilder, bei denen auch die SCHWAECHSTE Box mindestens so sicher ist, gehen ohne Review direkt ins Training (0.0-1.0). Gilt nicht bei Spielen mit automatischer Stufen-Schwelle (Spiel bearbeiten)"),
    ("ml_review_min_confidence", "ML: Review ab Konfidenz", "float", "Zwischen diesem Wert und der Auto-Training-Schwelle landet ein Bild in der Review Queue, darunter (nichts erkannt) wird es verworfen. Gilt nicht bei automatischer Stufen-Schwelle"),
    ("ml_max_review_per_video", "ML: Max. Review-Bilder pro Video", "int", "Begrenzt die Review Queue pro Video auf die unsichersten Bilder (0 = unbegrenzt). Gilt nicht bei automatischer Stufen-Schwelle"),
    ("ml_collect_frames", "ML: Frames beim Daten sammeln", "int", "Ziel-Anzahl Frames pro Lauf von 'Daten sammeln' (auf alle gewaehlten Videos nach Laenge verteilt). Etwa die Haelfte stammt von Modell-Events, der Rest sind neue Frames. Alles landet in der Review Queue, nichts geht automatisch ins Training"),
    ("ml_fast_scan", "ML: Schnell-Scan beim Daten sammeln", "bool", "An: das Video wird in gleichmaessig verteilten 30-s-Abschnitten geprueft und der Scan stoppt, sobald genug Events fuer die Frame-Anzahl gefunden sind (deutlich schneller). Aus: das ganze Video wird geprueft"),
    ("ml_decode_mode", "ML: Video-Dekodierung beim Daten sammeln", "choice", "cpu = wie bisher (exakt); hybrid = GPU (NVDEC) und CPU gleichzeitig, ca. 2,5x schneller; gpu = nur GPU, die CPU bleibt fuer Spiel/OBS frei, ca. 2,4x schneller. GPU-Modi verkleinern die Frames auf der GPU und finden dadurch ca. 90 % der Events, die der CPU-Modus findet (leicht andere Auswahl). Ist die GPU nicht nutzbar, wird automatisch die CPU genutzt"),
    ("ml_event_group_gap_sec", "ML: Event-Zusammenfassung (s)", "float", "Erkennungen mit kleinerem zeitlichen Abstand gelten als ein Event"),
    ("ml_frames_per_event", "ML: Frames pro Event", "int", "Wie viele Frames pro Event extrahiert werden"),
    ("ml_duplicate_threshold", "ML: Duplikat-Schwelle", "float", "Frames, die einem schon gespeicherten Frame so aehnlich sind (mittlere Grauwert-Abweichung 0-255), werden uebersprungen"),
]

# Rechner-/Dienst-bezogen und daher fuer ALLE Spiele gleich (nicht pro Spiel gespeichert)
GLOBAL_SETTINGS = {
    "use_gpu_encoding", "nvenc_cq", "crf", "max_parallel_cuts",
    "llm_provider", "llm_model", "ollama_base_url", "llm_batch_size", "whisper_model_size",
}

CHOICE_OPTIONS = {
    "scoring_mode": ["legacy", "llm"],
    "llm_provider": ["ollama", "openai"],
    "whisper_model_size": ["tiny", "base", "small", "medium", "large-v3"],
    "ml_decode_mode": ["cpu", "hybrid", "gpu"],
}


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".ts"}
THUMB_DIR = Path(tempfile.gettempdir()) / "highlight_cutter_thumbs"
THUMB_WIDTH = 320


def make_video_thumbnail(video_path: str):
    """Gibt den Pfad zu einem JPEG-Vorschaubild des Videos zurueck (None, falls keins erzeugt werden kann)."""
    try:
        src = Path(video_path)
        if src.suffix.lower() not in VIDEO_EXTENSIONS or not src.is_file():
            return None
        stat = src.stat()
        key = hashlib.sha1(f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()
        thumb = THUMB_DIR / f"{key}.jpg"
        if thumb.is_file() and thumb.stat().st_size > 0:
            return thumb
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        tmp = THUMB_DIR / f"{key}.{threading.get_ident()}.tmp.jpg"
        # erst ein Bild nach 5 s (vermeidet schwarze Intro-Frames), bei sehr kurzen Videos vom Anfang
        for offset in ("5", "0"):
            cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", offset, "-i", str(src),
                   "-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2", "-q:v", "4", str(tmp)]
            subprocess.run(cmd, capture_output=True, timeout=60)
            if tmp.is_file() and tmp.stat().st_size > 0:
                os.replace(tmp, thumb)
                return thumb
        tmp.unlink(missing_ok=True)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


MAX_COLLECT_FRAMES = 5000  # Obergrenze fuer 'Daten sammeln' (Schutz vor Tippfehlern)

_ML_SUM_KEYS = ("events", "groups", "duplicates", "candidates", "auto_train", "auto_val",
                "review", "review_capped", "review_empty", "filled", "discarded",
                "prelabeled", "collect_events", "collect_new", "collect_quota")


def _new_ml_totals() -> dict:
    totals = {k: 0 for k in _ML_SUM_KEYS}
    totals["conf_buckets"] = {}
    return totals


def _add_ml_stats(totals: dict, stats: dict):
    for k in _ML_SUM_KEYS:
        totals[k] += stats.get(k, 0)
    for label, n in stats.get("conf_buckets", {}).items():
        totals["conf_buckets"][label] = totals["conf_buckets"].get(label, 0) + n


def _build_collect_report(totals: dict, cfg: Config, target: int, detect_conf: float, scan_notes=None) -> dict:
    rows = [
        ("Modus", "Daten sammeln", "alles kommt in die Review Queue, nichts geht automatisch ins Training"),
        ("Ziel-Frames", target, f"Einstellung 'ML: Frames beim Daten sammeln'"),
    ]
    for note in (scan_notes or []):
        rows.append(("Scan", "", note))
    rows += [
        ("Erkennungen im Video", totals["events"], f"Event-Erkennung ab Konfidenz {detect_conf:.2f}"),
        ("Zusammengefasste Events", totals["groups"], f"Treffer < {cfg.ml_event_group_gap_sec:g}s Abstand = 1 Event"),
        ("In der Review gelandet", totals["review"], f"{totals['review']} von {target} Ziel-Frames"),
        ("  von Modell-Events", totals["collect_events"], "Frames an Stellen, wo das Modell etwas erkannt hat"),
        ("  neue Frames", totals["collect_new"], "gleichmaessig im Video verteilt, unabhaengig vom Modell"),
        ("  mit Vorlabel", totals["prelabeled"], "Modell schlaegt Boxen vor - bitte pruefen und korrigieren"),
        ("  ohne Vorlabel", totals["review_empty"], "Modell erkennt nichts - bitte selbst Boxen zeichnen (oder leer lassen)"),
        ("Duplikate uebersprungen", totals["duplicates"], "fast identische Frames"),
    ]
    buckets = [{"label": k, "count": v} for k, v in totals["conf_buckets"].items()]
    lines = ["===== ML-Bericht (Daten sammeln) ====="]
    lines += [f"{label:<28}{value:>6}   ({hint})" for label, value, hint in rows]
    lines.append("Konfidenz der Vorlabels (schwaechste Box):")
    lines += [f"  {b['label']}: {b['count']}" for b in buckets]
    return {
        "rows": [{"label": l, "value": v, "hint": h} for l, v, h in rows],
        "buckets": buckets,
        "text": "\n".join(lines),
    }


def _build_ml_report(totals: dict, cfg: Config, params: dict) -> dict:
    auto = totals["auto_train"] + totals["auto_val"]
    auto_on = params["auto_threshold"] <= 1.0
    auto_hint = (
        f"schwaechste Box >= {params['auto_threshold']:.2f} (davon {totals['auto_val']} in val)"
        if auto_on else "in dieser Stufe aus - alles geht durch deine Review"
    )
    review_hint = (
        f"Konfidenz {params['review_min']:.2f}-{params['auto_threshold']:.2f}"
        if auto_on else f"Konfidenz ab {params['review_min']:.2f}"
    )
    stage_hint = (
        f"automatisch nach Datenstand ({params['labeled_images']} gelabelte Bilder mit Boxen)"
        if params["adaptive"] else "feste Werte aus den Einstellungen"
    )
    rows = [
        ("Schwellen-Stufe", params["stage"], stage_hint),
        ("Erkennungen im Video", totals["events"], "YOLO-Treffer auf den geprueften Frames"),
        ("Zusammengefasste Events", totals["groups"], f"Treffer < {cfg.ml_event_group_gap_sec:g}s Abstand = 1 Event"),
        ("Gleichmaessig ergaenzt", totals["filled"],
         "Frames ohne Event, damit ein fruehes Modell genug Material zum Labeln liefert"),
        ("Duplikate uebersprungen", totals["duplicates"], "fast identische Frames"),
        ("Frames vorgelabelt", totals["candidates"], "gespeicherte Kandidaten"),
        ("Auto -> Training", auto, auto_hint),
        ("Zur Review", totals["review"], review_hint),
        ("  davon ohne Vorlabel", totals["review_empty"], "Modell erkennt noch nichts - bitte selbst Boxen zeichnen"),
        ("Review-Limit gekappt", totals["review_capped"],
         f"mehr als {params['max_review']} pro Video, am wenigsten unsichere entfernt"),
        ("Verworfen", totals["discarded"],
         f"nichts >= {params['review_min']:.2f} erkannt" if not params["keep_empty"] else "-"),
    ]
    buckets = [{"label": k, "count": v} for k, v in totals["conf_buckets"].items()]

    lines = ["===== ML-Bericht ====="]
    lines += [f"{label:<28}{value:>6}   ({hint})" for label, value, hint in rows]
    lines.append("Verteilung nach Konfidenz (schwaechste Box):")
    lines += [f"  {b['label']}: {b['count']}" for b in buckets]
    return {
        "rows": [{"label": l, "value": v, "hint": h} for l, v, h in rows],
        "buckets": buckets,
        "text": "\n".join(lines),
    }


class _LogStream:
    """Leitet print()-Ausgaben waehrend eines Hintergrund-Jobs live an die SSE-Clients um."""

    def __init__(self, core: "Core"):
        self.core = core

    def write(self, msg):
        if msg:
            self.core._push("log", msg)
        return len(msg)

    def flush(self):
        pass


class Core:
    def __init__(self):
        self.current_game = next(iter(GAMES), None)
        self.cfg = Config(yolo_model_path=None)
        self._load_game_cfg(self.current_game)
        self._dataset_managers = {}
        self._pipeline_controllers = {}
        self._review_images: list[Path] = []
        self.event_queue: "queue.Queue[dict]" = queue.Queue()
        self.cancel_event = None
        self.last_test_output = None

    # ------------------------------------------------------------
    # Interna
    # ------------------------------------------------------------

    def _push(self, event: str, data=None):
        self.event_queue.put({"event": event, "data": data})

    def _dm_for(self, game_name: str) -> DatasetManager:
        if game_name not in self._dataset_managers:
            self._dataset_managers[game_name] = DatasetManager(game_name)
        return self._dataset_managers[game_name]

    def _dm(self) -> DatasetManager:
        return self._dm_for(self.current_game)

    def _require_model(self):
        """Gibt eine Fehlermeldung zurueck, solange das aktuelle Spiel noch kein Modell hat."""
        if not self._dm().game.model_path.is_file():
            return {
                "ok": False,
                "error": (
                    f"'{self.current_game}' hat noch kein Modell. Bitte zuerst die Frames in der Review Queue "
                    "labeln und das erste Modell trainieren (Button 'Erstes Modell trainieren')."
                ),
            }
        return None

    def _pc(self) -> PipelineController:
        if self.current_game not in self._pipeline_controllers:
            dm = self._dm()
            self._pipeline_controllers[self.current_game] = PipelineController(
                game_name=self.current_game,
                model_path=str(dm.game.model_path),
            )
        return self._pipeline_controllers[self.current_game]

    def _sync_model_path(self):
        game = self._dm().game
        self.cfg.yolo_model_path = str(game.model_path)
        # Pro Spiel: eigene Event-Klassen, sonst der globale Standard (bestehende Spiele)
        self.cfg.yolo_target_classes = (
            list(game.event_classes) if game.event_classes else list(Config().yolo_target_classes)
        )

    def _review_count(self) -> int:
        try:
            return len(list(self._dm().review_images.glob("*.jpg")))
        except Exception:
            return 0

    # ------------------------------------------------------------
    # Allgemeiner Status / Spiele
    # ------------------------------------------------------------

    def get_state(self):
        classes, model_path, review_count = [], "", 0
        has_model, dataset_stats = True, {}
        if self.current_game:
            dm = self._dm()
            classes = dm.game.classes
            model_path = str(dm.game.model_path)
            review_count = self._review_count()
            has_model = dm.game.model_path.is_file()
            dataset_stats = dm.labeled_stats()

        return {
            "games": list(GAMES.keys()),
            "currentGame": self.current_game,
            "classes": classes,
            "modelPath": model_path,
            "hasModel": has_model,
            "datasetStats": dataset_stats,
            "firstModelMinLabeled": FIRST_MODEL_MIN_LABELED,
            "reviewCount": review_count,
            "settingsCustomized": bool(self.current_game and get_game_settings(self.current_game)),
            "settingsSchema": self._settings_schema(),
        }

    def _settings_schema(self):
        return [
            {"attr": a, "label": l, "type": t, "desc": d, "value": getattr(self.cfg, a),
             "options": CHOICE_OPTIONS.get(a), "global": a in GLOBAL_SETTINGS,
             "group": "ml" if a.startswith("ml_") else "cutter"}
            for (a, l, t, d) in SETTINGS_SCHEMA
        ]

    def select_game(self, name: str):
        if name not in GAMES:
            return {"ok": False, "error": f"Unbekanntes Spiel: {name}"}
        if self.cancel_event is not None and name != self.current_game:
            return {"ok": False, "error": "Waehrend eines laufenden Vorgangs kann das Spiel nicht gewechselt werden."}
        self.current_game = name
        self._load_game_cfg(name)
        return {"ok": True, "state": self.get_state()}

    def refresh_games(self):
        try:
            reload_games()
        except (GamesConfigError, FileNotFoundError) as e:
            return {"ok": False, "error": str(e)}

        self._dataset_managers.clear()
        self._pipeline_controllers.clear()
        if self.current_game not in GAMES:
            self.current_game = next(iter(GAMES), None)
        self._load_game_cfg(self.current_game)
        return {"ok": True, "state": self.get_state()}

    # ------------------------------------------------------------
    # Einstellungen (pro Spiel gespeichert, siehe config/game_settings.py)
    # ------------------------------------------------------------

    @staticmethod
    def _coerce_setting(attr: str, typ: str, raw):
        if typ == "bool":
            return bool(raw)
        if typ == "int":
            return int(raw)
        if typ == "float":
            value = float(raw)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("keine endliche Zahl")
            return value
        text = str(raw).strip()  # "text" | "choice"
        return (text or None) if attr == "whisper_language" else text

    def _load_game_cfg(self, name):
        """Setzt die spielbezogenen Einstellungen auf Standard + gespeicherte Werte des Spiels."""
        defaults = Config()
        saved = get_game_settings(name) if name else {}
        for attr, _label, typ, _desc in SETTINGS_SCHEMA:
            if attr in GLOBAL_SETTINGS:
                continue
            value = getattr(defaults, attr)
            if attr in saved:
                try:
                    value = self._coerce_setting(attr, typ, saved[attr])
                except (ValueError, TypeError):
                    pass
            setattr(self.cfg, attr, value)

    def _persist_game_cfg(self, name):
        """Speichert nur die vom Standard abweichenden spielbezogenen Werte."""
        defaults = Config()
        diff = {}
        for attr, _label, _typ, _desc in SETTINGS_SCHEMA:
            if attr in GLOBAL_SETTINGS:
                continue
            if getattr(self.cfg, attr) != getattr(defaults, attr):
                diff[attr] = getattr(self.cfg, attr)
        save_game_settings(name, diff)
        return bool(diff)

    def apply_settings(self, values: dict):
        errors = []
        for attr, _label, typ, _desc in SETTINGS_SCHEMA:
            if attr not in values:
                continue
            try:
                setattr(self.cfg, attr, self._coerce_setting(attr, typ, values[attr]))
            except (ValueError, TypeError):
                errors.append(attr)

        customized = False
        if self.current_game:
            try:
                customized = self._persist_game_cfg(self.current_game)
            except OSError as e:
                return {"ok": False, "errors": [], "error": f"Einstellungen konnten nicht gespeichert werden: {e}"}

        return {"ok": not errors, "errors": errors, "game": self.current_game, "customized": customized}

    def reset_settings(self):
        defaults = Config()
        for attr, _label, _typ, _desc in SETTINGS_SCHEMA:
            setattr(self.cfg, attr, getattr(defaults, attr))
        if self.current_game:
            try:
                save_game_settings(self.current_game, {})
            except OSError as e:
                return {"ok": False, "error": f"Einstellungen konnten nicht zurueckgesetzt werden: {e}"}
        return {"ok": True, "settingsSchema": self._settings_schema(), "customized": False}

    # ------------------------------------------------------------
    # Server-seitiger Datei-Browser (Ersatz fuer native Dialoge)
    # ------------------------------------------------------------

    def pick_native_files(self, title: str, filter_str: str, multiple: bool):
        """Zeigt den echten Windows-Explorer-Dateidialog an."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filetypes = [("Videos", "*.mp4 *.mkv *.avi *.mov"), ("Alle Dateien", "*.*")]
        if filter_str.startswith("YOLO"):
            filetypes = [("YOLO-Modell", "*.pt")]
        elif filter_str.startswith("JSON"):
            filetypes = [("JSON", "*.json")]
        try:
            if multiple:
                paths = filedialog.askopenfilenames(title=title, filetypes=filetypes, parent=root)
            else:
                path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=root)
                paths = (path,) if path else ()
            return list(paths)
        finally:
            root.destroy()

    def extract_classes_from_json(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            classes = self._parse_classes(data)
            return {"ok": True, "classes": classes}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _parse_classes(data):
        if isinstance(data, list):
            classes = data
        elif isinstance(data, dict):
            if "classes" in data:
                classes = data["classes"]
            elif "names" in data:
                names = data["names"]
                classes = list(names.values()) if isinstance(names, dict) else list(names)
            elif "categories" in data:
                classes = [c["name"] for c in data["categories"]]
            else:
                raise ValueError("Kein 'classes', 'names' oder 'categories' Feld gefunden.")
        else:
            raise ValueError("Unerwartetes JSON-Format.")

        classes = [str(c).strip() for c in classes if str(c).strip()]
        if not classes:
            raise ValueError("Keine Klassennamen gefunden.")
        return classes

    # ------------------------------------------------------------
    # Neues Spiel anlegen
    # ------------------------------------------------------------

    def add_game(self, name: str, model_source_path: str, classes: list, video_paths: list | None = None):
        """
        Legt ein Spiel an. Pflicht: ENTWEDER ein vorhandenes YOLO-Modell (.pt) ODER
        mindestens ein Video. Bei Video werden ca. 100 Frames in die Review Queue
        gelegt, die der Nutzer labelt; damit wird anschliessend das erste Modell trainiert.
        """
        name = (name or "").strip()
        classes = [c.strip() for c in (classes or []) if c and c.strip()]
        model_source_path = (model_source_path or "").strip()
        video_paths = [v for v in (video_paths or []) if v]

        if not name:
            return {"ok": False, "error": "Bitte einen Spielnamen eingeben."}
        if name in GAMES:
            return {"ok": False, "error": f"Ein Spiel namens '{name}' existiert bereits."}
        if model_source_path and video_paths:
            return {"ok": False, "error": "Bitte nur EINE Quelle waehlen: entweder ein Modell (.pt) oder Video(s)."}
        if not model_source_path and not video_paths:
            return {"ok": False, "error": "Bitte ein YOLO-Modell (.pt) ODER ein Video zum Spiel auswaehlen."}
        if model_source_path and not Path(model_source_path).is_file():
            return {"ok": False, "error": "Bitte ein gueltiges YOLO-Modell (.pt) auswaehlen."}
        missing = [v for v in video_paths if not Path(v).is_file()]
        if missing:
            return {"ok": False, "error": f"Video nicht gefunden: {missing[0]}"}
        if not classes:
            return {"ok": False, "error": "Bitte mindestens eine Klasse angeben."}
        if len(classes) != len(set(classes)):
            return {"ok": False, "error": "Doppelte Klassennamen gefunden."}
        if video_paths and self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}

        try:
            model_dst = PROJECT_ROOT / "models" / name / "current.pt"
            model_dst.parent.mkdir(parents=True, exist_ok=True)
            if model_source_path:
                shutil.copy2(model_source_path, model_dst)

            with open(GAMES_CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)

            raw.setdefault("games", {})[name] = {
                "model_path": f"models/{name}/current.pt",
                "dataset_path": f"datasets/{name}",
                "review_queue_path": f"review_queue/{name}",
                "classes": classes,
                "event_classes": list(classes),  # neue Spiele: alle eigenen Klassen werden von YOLO erkannt/als Event genutzt
                "adaptive_thresholds": True,  # neue Spiele: ML-Schwellen wachsen mit dem Datenstand
            }

            with open(GAMES_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)

            reload_games()
            DatasetManager(name)  # legt Ordnerstruktur (images/labels/train/val) sofort an
        except (GamesConfigError, OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}

        self.current_game = name
        self._load_game_cfg(name)
        if video_paths:
            self.cancel_event = threading.Event()
            threading.Thread(
                target=self._bootstrap_worker, args=(name, video_paths, self.cancel_event), daemon=True
            ).start()
        return {"ok": True, "state": self.get_state(), "bootstrap": bool(video_paths)}

    # ------------------------------------------------------------
    # Spiel bearbeiten / loeschen
    # ------------------------------------------------------------

    @staticmethod
    def _game_dirs(game) -> dict:
        return {
            "models": game.model_path.parent,
            "datasets": game.dataset_path,
            "review_queue": game.review_queue_path,
        }

    @staticmethod
    def _dir_size(path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0

    def get_game_info(self, name: str):
        if name not in GAMES:
            return {"ok": False, "error": f"Unbekanntes Spiel: {name}"}
        game = GAMES[name]
        dm = self._dm_for(name)
        stats = dm.labeled_stats()
        size_mb = sum(self._dir_size(p) for p in self._game_dirs(game).values()) / (1024 * 1024)
        return {
            "ok": True,
            "name": name,
            "classes": game.classes,
            "hasModel": game.model_path.is_file(),
            "train": stats["train"],
            "val": stats["val"],
            "withBoxes": stats["with_boxes"],
            "review": stats["review"],
            "sizeMb": round(size_mb, 1),
            "isLast": len(GAMES) <= 1,
            # Klassen duerfen nur erweitert werden, sobald Modell oder Labels existieren
            "classesLocked": game.model_path.is_file() or stats["with_boxes"] > 0,
            "adaptiveThresholds": game.adaptive_thresholds,
            "stage": resolve_ml_params(self.cfg, True, stats["with_boxes"]),
        }

    def delete_game(self, name: str, confirm_name: str):
        """
        Entfernt ein Spiel aus der Liste. Modell, Datensatz und Review Queue werden NICHT
        sofort geloescht, sondern nach _deleted_games/<Spiel>_<Zeitstempel>/ verschoben.
        """
        if name not in GAMES:
            return {"ok": False, "error": f"Unbekanntes Spiel: {name}"}
        if (confirm_name or "").strip() != name:
            return {"ok": False, "error": "Der eingegebene Spielname stimmt nicht ueberein."}
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft gerade ein Vorgang. Bitte abwarten oder abbrechen."}
        if len(GAMES) <= 1:
            return {"ok": False, "error": "Das letzte Spiel kann nicht geloescht werden."}

        game = GAMES[name]
        dirs = self._game_dirs(game)

        try:
            with open(GAMES_CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            raw.get("games", {}).pop(name, None)
            with open(GAMES_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
            reload_games()
        except (OSError, ValueError, GamesConfigError) as e:
            return {"ok": False, "error": str(e)}

        trash = PROJECT_ROOT / "_deleted_games" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        moved, problems = [], []
        for label, src in dirs.items():
            try:
                src_resolved = src.resolve()
                if not src_resolved.is_dir():
                    continue
                if PROJECT_ROOT.resolve() not in src_resolved.parents:
                    problems.append(f"{label}: Ordner liegt ausserhalb des Projekts und wurde nicht angefasst")
                    continue
                trash.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_resolved), str(trash / label))
                moved.append(label)
            except OSError as e:
                problems.append(f"{label}: {e}")

        removed_settings = pop_game_settings(name)
        if removed_settings:
            trash.mkdir(parents=True, exist_ok=True)
            (trash / "game_settings.json").write_text(
                json.dumps(removed_settings, indent=2, ensure_ascii=False), encoding="utf-8")
            if "settings" not in moved:
                moved.append("settings")

        self._dataset_managers.pop(name, None)
        self._pipeline_controllers.pop(name, None)
        if self.current_game == name or self.current_game not in GAMES:
            self.current_game = next(iter(GAMES), None)
            self._load_game_cfg(self.current_game)

        return {
            "ok": True,
            "state": self.get_state(),
            "trashDir": str(trash) if moved else None,
            "problems": problems,
        }

    def edit_game(self, name: str, classes: list, adaptive_thresholds: bool | None = None):
        if name not in GAMES:
            return {"ok": False, "error": f"Unbekanntes Spiel: {name}"}
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft gerade ein Vorgang."}

        classes = [c.strip() for c in (classes or []) if c and c.strip()]
        if not classes:
            return {"ok": False, "error": "Bitte mindestens eine Klasse angeben."}
        if len(classes) != len(set(classes)):
            return {"ok": False, "error": "Doppelte Klassennamen gefunden."}

        info = self.get_game_info(name)
        old = list(GAMES[name].classes)
        if info["classesLocked"] and classes[:len(old)] != old:
            return {
                "ok": False,
                "error": (
                    "Dieses Spiel hat schon ein Modell bzw. gelabelte Bilder. Bestehende Klassen duerfen nicht "
                    "umbenannt, umsortiert oder entfernt werden (die Klassen-Nummern in den Labels wuerden "
                    "sonst nicht mehr passen). Neue Klassen koennen nur am Ende ergaenzt werden."
                ),
            }

        try:
            with open(GAMES_CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            entry = raw["games"][name]
            entry["classes"] = classes
            if entry.get("event_classes") is not None:
                # Spiele mit eigenen Event-Klassen (neu angelegt): alle Klassen bleiben Event-Klassen
                entry["event_classes"] = list(classes)
            # Nur schreiben, wenn der Nutzer den Schalter wirklich geaendert hat
            if adaptive_thresholds is not None and bool(adaptive_thresholds) != GAMES[name].adaptive_thresholds:
                entry["adaptive_thresholds"] = bool(adaptive_thresholds)
            with open(GAMES_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
            reload_games()
        except (OSError, ValueError, KeyError, GamesConfigError) as e:
            return {"ok": False, "error": str(e)}

        self._dataset_managers.pop(name, None)
        self._pipeline_controllers.pop(name, None)
        return {"ok": True, "state": self.get_state()}

    def _bootstrap_worker(self, game_name, video_paths, cancel_event):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            dm = self._dm_for(game_name)
            self._push("status", f"Erstelle Start-Frames fuer '{game_name}' ...")
            stats = extract_bootstrap_frames(
                FrameExtractor(dm),
                video_paths,
                target=BOOTSTRAP_FRAME_TARGET,
                duplicate_threshold=self.cfg.ml_duplicate_threshold,
                progress_callback=lambda phase, pct: self._push("progress", {"phase": phase, "percent": pct}),
                cancel_event=cancel_event,
            )
            self._push("reviewCount", self._review_count())
            self._push("stateChanged")
            self._push("status", "Start-Frames fertig")
            self._push("done", {
                "message": (
                    f"{stats['extracted']} Frames liegen in der Review Queue von '{game_name}'. "
                    f"Als Naechstes: Frames labeln, dann das erste Modell trainieren."
                ),
            })
        except Exception as e:
            self._push("cancelled" if cancel_event.is_set() else "error", str(e))
        finally:
            sys.stdout = real_stdout
            self.cancel_event = None
            self._push("busy", False)

    # ------------------------------------------------------------
    # Verarbeitung (Hintergrund-Threads, Fortschritt/Log via SSE-Queue)
    # ------------------------------------------------------------

    def start_processing(self, video_paths: list):
        if not video_paths:
            return {"ok": False, "error": "Keine Videos ausgewaehlt."}
        if (err := self._require_model()):
            return err
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}
        self.cancel_event = threading.Event()
        threading.Thread(target=self._process_worker, args=(list(video_paths), self.cancel_event), daemon=True).start()
        return {"ok": True}

    def _process_worker(self, video_paths, cancel_event):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self._sync_model_path()
            total = len(video_paths)

            for idx, video in enumerate(video_paths, start=1):
                if cancel_event.is_set():
                    raise RuntimeError("Vorgang abgebrochen.")

                self._push("status", f"Bearbeite {idx}/{total}: {os.path.basename(video)}")
                base = os.path.splitext(os.path.basename(video))[0]
                output_path = os.path.join(OUTPUT_DIR, f"{base}_highlight.mp4")

                process_video(
                    video_path=video,
                    output_path=output_path,
                    cfg=self.cfg,
                    progress_callback=lambda phase, pct: self._push("progress", {"phase": phase, "percent": pct}),
                    cancel_event=cancel_event,
                    game_name=self.current_game,
                    game_context=self._dm().game.llm_context,
                )
                self._push("overallProgress", {"idx": idx, "total": total})

            self._push("status", "Fertig")
            self._push("done", {"message": "Alle Videos wurden verarbeitet."})
        except Exception as e:
            self._push("cancelled" if cancel_event.is_set() else "error", str(e))
        finally:
            sys.stdout = real_stdout
            self.cancel_event = None
            self._push("busy", False)

    def run_ml_pipeline(self, video_paths: list, export_fcpxml: bool = False, ml_only: bool = False,
                        collect_frames=None):
        """
        export_fcpxml=True: ML-Pipeline + FCPXML-Timeline, ohne das Video zu schneiden.
        ml_only=True: nur Events erkennen + ML-Pipeline (Daten sammeln), weder schneiden noch FCPXML.
        collect_frames: Ziel-Frames fuer 'Daten sammeln'; wird fuer das aktuelle Spiel gespeichert.
        """
        if not video_paths:
            return {"ok": False, "error": "Keine Videos ausgewaehlt."}
        if (err := self._require_model()):
            return err
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}

        customized = None
        if ml_only and collect_frames is not None:
            try:
                value = int(collect_frames)
            except (TypeError, ValueError):
                return {"ok": False, "error": "Die Frame-Anzahl muss eine ganze Zahl sein."}
            if not 1 <= value <= MAX_COLLECT_FRAMES:
                return {"ok": False, "error": f"Die Frame-Anzahl muss zwischen 1 und {MAX_COLLECT_FRAMES} liegen."}
            self.cfg.ml_collect_frames = value
            try:
                customized = self._persist_game_cfg(self.current_game)
            except OSError as e:
                return {"ok": False, "error": f"Einstellung konnte nicht gespeichert werden: {e}"}
        self.cancel_event = threading.Event()
        threading.Thread(
            target=self._ml_pipeline_worker,
            args=(list(video_paths), self.cancel_event, bool(export_fcpxml), bool(ml_only)),
            daemon=True,
        ).start()
        return {"ok": True, "customized": customized, "collectFrames": self.cfg.ml_collect_frames}

    def _ml_pipeline_worker(self, video_paths, cancel_event, export_fcpxml=False, ml_only=False):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self._sync_model_path()
            total = len(video_paths)
            totals = _new_ml_totals()
            total_clips = 0
            dm = self._dm()
            params = None
            if ml_only:
                # Daten sammeln: festes Frame-Ziel, nach Videolaenge auf die Videos verteilt
                target = max(1, int(self.cfg.ml_collect_frames))
                quotas = split_quota([video_duration(v) for v in video_paths], target)
                scan_notes: list[str] = []
                collect_conf = min(self.cfg.yolo_confidence, COLLECT_MIN_CONFIDENCE)
                print(f"[ML] Daten sammeln: Ziel {target} Frames (je ca. {COLLECT_EVENT_SHARE:.0%} von Events und "
                      f"neue Frames), Event-Erkennung und Vorlabels ab {collect_conf:.2f}, alles in die Review")
            else:
                params = resolve_ml_params(self.cfg, dm.game.adaptive_thresholds, dm.labeled_stats()["with_boxes"])
                auto_txt = f"{params['auto_threshold']:.2f}" if params["auto_threshold"] <= 1.0 else "aus"
                print(f"[ML] Schwellen-Stufe '{params['stage']}' ({params['labeled_images']} gelabelte Bilder): "
                      f"Auto-Training {auto_txt}, Review ab {params['review_min']:.2f}, "
                      f"max. {params['max_review']} Review-Bilder pro Video, "
                      f"Event-Erkennung ab {params['detect_conf']:.2f}")
            for idx, video in enumerate(video_paths, start=1):
                if cancel_event.is_set():
                    raise RuntimeError("Vorgang abgebrochen.")

                self._push("status", f"Analyse {idx}/{total}: {os.path.basename(video)}")
                base = os.path.splitext(os.path.basename(video))[0]
                progress_cb = lambda phase, pct: self._push("progress", {"phase": phase, "percent": pct})
                analysis_kwargs = dict(
                    video_path=video,
                    cfg=self.cfg,
                    progress_callback=progress_cb,
                    cancel_event=cancel_event,
                    game_name=self.current_game,
                    game_context=self._dm().game.llm_context,
                )

                if ml_only:
                    if quotas[idx - 1] <= 0:
                        print(f"[ML] {os.path.basename(video)}: Laenge unbekannt/zu kurz, uebersprungen.")
                    else:
                        quota = quotas[idx - 1]
                        decode_mode = self.cfg.ml_decode_mode
                        if self.cfg.ml_fast_scan or decode_mode != "cpu":
                            # Stichproben-Scan: stoppt, sobald genug Events fuer die Event-Haelfte gefunden sind.
                            # Ohne Schnell-Scan wird das ganze Video geprueft (need=None), ggf. auf der GPU.
                            need = int(quota * COLLECT_EVENT_SHARE * 1.5) + 5 if self.cfg.ml_fast_scan else None
                            scan = collect_events(
                                video, self.cfg.yolo_model_path, collect_conf, self.cfg.yolo_target_classes,
                                every_n=self.cfg.yolo_sample_every_n_frames, need_groups=need,
                                group_gap_sec=self.cfg.ml_event_group_gap_sec, decode_mode=decode_mode,
                                progress_callback=progress_cb, cancel_event=cancel_event,
                            )
                            events = scan["events"]
                            used = {"cpu": "CPU", "hybrid": "GPU + CPU", "gpu": "GPU"}[scan["decode_mode"]]
                            if scan["decode_mode"] != "cpu":
                                used += f" ({scan['gpu_chunks']} Abschnitte GPU, {scan['cpu_chunks']} CPU)"
                            if scan.get("gpu_failed"):
                                used += ", GPU ausgefallen -> CPU"
                            note = (f"{os.path.basename(video)}: {scan['scanned_sec']:.0f}s von {scan['duration']:.0f}s geprueft "
                                    f"({scan['windows_done']}/{scan['chunks_total']} Abschnitte, {scan['seconds']:.0f}s Scan-Zeit, "
                                    f"Dekodierung: {used}"
                                    + (", gestoppt: genug Events" if scan["stopped_early"] else ", ganzes Video") + ")")
                            if scan["decode_note"]:
                                note += f" | {scan['decode_note']}"
                            scan_notes.append(note)
                            print(f"[ML] Scan: {note}")
                        else:
                            detect_cfg = dataclasses.replace(self.cfg, yolo_confidence=collect_conf)
                            events = sorted(preload_yolo_events(video, detect_cfg, progress_cb, cancel_event))
                            scan_notes.append(f"{os.path.basename(video)}: ganzes Video geprueft (Schnell-Scan aus)")
                        stats = self._pc().run_collect_cycle(
                            video_path=video,
                            events=events,
                            quota=quota,
                            event_group_gap_sec=self.cfg.ml_event_group_gap_sec,
                            frames_per_event=self.cfg.ml_frames_per_event,
                            duplicate_threshold=self.cfg.ml_duplicate_threshold,
                            min_confidence=collect_conf,
                            progress_callback=progress_cb,
                            status_callback=lambda text: self._push("status", text),
                            cancel_event=cancel_event,
                        )
                        _add_ml_stats(totals, stats)
                    self._push("overallProgress", {"idx": idx, "total": total})
                    continue

                if export_fcpxml:
                    result = export_fcpxml_timeline(
                        output_path=os.path.join(OUTPUT_DIR, f"{base}_timeline.fcpxml"),
                        **analysis_kwargs,
                    )
                    total_clips += result["clip_count"]
                else:
                    result = process_video(
                        output_path=os.path.join(OUTPUT_DIR, f"{base}_highlight.mp4"),
                        **analysis_kwargs,
                    )

                stats = self._pc().run_full_cycle(
                    video_path=video,
                    events=result["yolo_event_times"],
                    event_group_gap_sec=self.cfg.ml_event_group_gap_sec,
                    frames_per_event=self.cfg.ml_frames_per_event,
                    duplicate_threshold=self.cfg.ml_duplicate_threshold,
                    auto_threshold=params["auto_threshold"],
                    review_min_confidence=params["review_min"],
                    max_review_per_video=params["max_review"],
                    keep_empty=params["keep_empty"],
                    fill_to=params["fill_frames"],
                    progress_callback=progress_cb,
                    status_callback=lambda text: self._push("status", text),
                    cancel_event=cancel_event,
                )
                _add_ml_stats(totals, stats)
                self._push("overallProgress", {"idx": idx, "total": total})

            if ml_only:
                report = _build_collect_report(totals, self.cfg, target, collect_conf, scan_notes)
                message = (
                    f"Daten sammeln fertig: {totals['review']} Frames in der Review Queue "
                    f"({totals['prelabeled']} mit Vorlabel, {totals['review_empty']} ohne). "
                    f"Bitte in der Review pruefen und korrigieren."
                )
            else:
                report = _build_ml_report(totals, self.cfg, params)
                message = (
                    f"ML-Pipeline fertig: {totals['auto_train'] + totals['auto_val']} automatisch ins Training, "
                    f"{totals['review']} zur Review, {totals['discarded']} verworfen "
                    f"(Details im ML-Bericht)."
                )
            print("\n" + report["text"])
            self._push("mlReport", report)
            self._push("reviewCount", self._review_count())
            self._push("status", "ML-Pipeline fertig")
            if export_fcpxml:
                message += f"\n{total_clips} Clips als FCPXML exportiert (KEIN Video geschnitten)"
            self._push("done", {"message": message})
        except Exception as e:
            self._push("cancelled" if cancel_event.is_set() else "error", str(e))
        finally:
            sys.stdout = real_stdout
            self.cancel_event = None
            self._push("busy", False)

    def get_training_overview(self):
        if not self.current_game:
            return {"ok": False, "error": "Kein Spiel ausgewaehlt."}
        return {"ok": True, **build_training_overview(self._dm(), FIRST_MODEL_MIN_LABELED)}

    def run_training(self):
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}

        dm = self._dm()
        first_model = not dm.game.model_path.is_file()
        if first_model:
            stats = dm.labeled_stats()
            if stats["with_boxes"] < FIRST_MODEL_MIN_LABELED or stats["val"] < 1:
                return {
                    "ok": False,
                    "error": (
                        f"Fuer das erste Modell werden mindestens {FIRST_MODEL_MIN_LABELED} gelabelte Bilder "
                        f"mit Boxen benoetigt (aktuell {stats['with_boxes']}, davon {stats['val']} in val). "
                        "Bitte in der Review Queue weitere Bilder labeln und mit 'Accept' bestaetigen."
                    ),
                }

        self.cancel_event = threading.Event()
        threading.Thread(target=self._train_worker, args=(self.cancel_event, first_model), daemon=True).start()
        return {"ok": True}

    def _train_worker(self, cancel_event, first_model=False):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            dm = self._dm()
            game_name = self.current_game
            base_weights = PROJECT_ROOT / "yolo26n.pt"
            trainer = Trainer(
                dm,
                model_path=str(dm.game.model_path),
                base_weights=str(base_weights) if base_weights.is_file() else "yolo26n.pt",
            )
            self._push("status", "Erstes Modell wird trainiert..." if first_model else "Training laeuft...")
            epochs = 60 if first_model else 30
            # Datenstand VOR dem Training merken: genau damit wird trainiert
            snapshot = compute_dataset_snapshot(dm)
            trainer.train(epochs=epochs, img_size=640, cancel_event=cancel_event)
            append_history(dm.game.model_path, snapshot, epochs, first_model)
            self._push("status", "Training fertig")

            # Gecachte Pipeline nutzt sonst weiter das alte (bzw. kein) Modell
            self._pipeline_controllers.pop(game_name, None)
            self._push("stateChanged")

            if first_model:
                model_path = str(dm.game.model_path)
                self._push("firstModelReady", {
                    "game": game_name,
                    "modelPath": model_path,
                    "versionsDir": str(trainer.version_dir),
                })
                self._push("done", {"message": f"Erstes Modell fuer '{game_name}' gespeichert: {model_path}"})
            else:
                self._push("done", {"message": "Retraining abgeschlossen."})
        except Exception as e:
            self._push("cancelled" if cancel_event.is_set() else "error", str(e))
        finally:
            sys.stdout = real_stdout
            self.cancel_event = None
            self._push("busy", False)

    def test_model(self, video_path: str, confidence: float, sample_every_n: int, max_minutes: float):
        if not video_path or not Path(video_path).is_file():
            return {"ok": False, "error": "Bitte ein gueltiges Testvideo auswaehlen."}
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}
        if (err := self._require_model()):
            return err
        try:
            confidence = float(confidence)
            sample_every_n = int(sample_every_n)
            max_minutes = float(max_minutes or 0)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Ungueltige Test-Einstellungen."}
        if not 0 < confidence < 1 or sample_every_n < 1 or max_minutes < 0:
            return {"ok": False, "error": "Konfidenz muss zwischen 0 und 1 liegen, 'jeder N-te Frame' >= 1."}

        self.cancel_event = threading.Event()
        threading.Thread(
            target=self._test_model_worker,
            args=(video_path, confidence, sample_every_n, max_minutes, self.cancel_event),
            daemon=True,
        ).start()
        return {"ok": True}

    def _test_model_worker(self, video_path, confidence, sample_every_n, max_minutes, cancel_event):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            game = self._dm().game
            out_dir = PROJECT_ROOT / "model_tests" / game.name
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(out_dir / f"{Path(video_path).stem}_modelltest_{stamp}.mp4")

            self._push("status", f"Modell-Test: {os.path.basename(video_path)}")
            result = test_model_on_video(
                model_path=str(game.model_path),
                video_path=video_path,
                output_path=output_path,
                confidence=confidence,
                sample_every_n=sample_every_n,
                max_minutes=max_minutes,
                progress_callback=lambda phase, pct: self._push("progress", {"phase": phase, "percent": pct}),
                cancel_event=cancel_event,
            )
            self.last_test_output = output_path
            result["game"] = game.name
            result["modelPath"] = str(game.model_path)
            self._push("status", "Modell-Test fertig")
            self._push("modelTestResult", result)
            self._push("done", {"message": f"Modell-Test fertig: {result['detections']} Erkennungen."})
        except Exception as e:
            self._push("cancelled" if cancel_event.is_set() else "error", str(e))
        finally:
            sys.stdout = real_stdout
            self.cancel_event = None
            self._push("busy", False)

    def open_last_test_output(self, folder: bool = False):
        path = self.last_test_output
        if not path or not os.path.isfile(path):
            return {"ok": False, "error": "Noch kein Test-Video vorhanden."}
        if folder:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            os.startfile(path)
        return {"ok": True}

    def export_fcpxml(self, video_paths: list):
        if not video_paths:
            return {"ok": False, "error": "Keine Videos ausgewaehlt."}
        if (err := self._require_model()):
            return err
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}
        self.cancel_event = threading.Event()
        threading.Thread(target=self._fcpxml_worker, args=(list(video_paths), self.cancel_event), daemon=True).start()
        return {"ok": True}

    def _fcpxml_worker(self, video_paths, cancel_event):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self._sync_model_path()
            total = len(video_paths)
            total_clips = 0
            total_top = 0

            for idx, video in enumerate(video_paths, start=1):
                if cancel_event.is_set():
                    raise RuntimeError("Vorgang abgebrochen.")

                self._push("status", f"FCPXML-Export: {os.path.basename(video)}")
                base = os.path.splitext(os.path.basename(video))[0]
                output_path = os.path.join(OUTPUT_DIR, f"{base}_timeline.fcpxml")

                result = export_fcpxml_timeline(
                    video_path=video,
                    output_path=output_path,
                    cfg=self.cfg,
                    progress_callback=lambda phase, pct: self._push("progress", {"phase": phase, "percent": pct}),
                    cancel_event=cancel_event,
                    game_name=self.current_game,
                    game_context=self._dm().game.llm_context,
                )
                total_clips += result["clip_count"]
                total_top += len(result["top_clip_indices"])
                self._push("overallProgress", {"idx": idx, "total": total})

            self._push("status", "FCPXML-Export fertig")
            self._push("done", {
                "message": (
                    f"{total_clips} Clips in {total} Timeline(s) exportiert "
                    f"({total_top} Top-Marker gesetzt). Es wurde KEIN Video geschnitten."
                ),
            })
        except Exception as e:
            self._push("cancelled" if cancel_event.is_set() else "error", str(e))
        finally:
            sys.stdout = real_stdout
            self.cancel_event = None
            self._push("busy", False)

    def cancel_current_operation(self):
        if self.cancel_event is None:
            return {"ok": False, "error": "Kein Vorgang laeuft."}
        self.cancel_event.set()
        return {"ok": True}

    # ------------------------------------------------------------
    # Review / Labeling
    # ------------------------------------------------------------

    def review_open(self):
        dm = self._dm()
        self._review_images = sorted(dm.review_images.glob("*.jpg"))
        return {
            "classes": dm.game.classes,
            "images": [{"id": p.stem} for p in self._review_images],
        }

    _SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

    def find_review_image_path(self, sample_id: str):
        """Sucht das Bild direkt im Review-Ordner des AKTUELLEN Spiels (kein veralteter Zwischenspeicher)."""
        if not self._SAMPLE_ID_RE.match(sample_id or ""):
            return None
        path = self._dm().review_images / f"{sample_id}.jpg"
        return path if path.is_file() else None

    # ------------------------------------------------------------
    # Dataset-Ansicht (nur lesen): train/val-Bilder samt Labels
    # ------------------------------------------------------------

    _DATASET_SPLITS = ("train", "val")

    @staticmethod
    def _read_label_boxes(label_file: Path) -> list:
        """Liest eine YOLO-Label-Datei; ungueltige Zeilen werden uebersprungen."""
        boxes = []
        try:
            text = label_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return boxes
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            boxes.append({"cls": cls, "x": x, "y": y, "w": w, "h": h})
        return boxes

    def _dataset_dirs(self, split: str):
        dm = self._dm()
        if split == "train":
            return dm.train_images, dm.train_labels
        if split == "val":
            return dm.val_images, dm.val_labels
        return None, None

    def dataset_list(self):
        if not self.current_game:
            return {"ok": False, "error": "Kein Spiel ausgewaehlt."}
        dm = self._dm()
        items = []
        per_class: dict[int, int] = {}
        counts = {"train": 0, "val": 0}
        for split in self._DATASET_SPLITS:
            img_dir, lbl_dir = self._dataset_dirs(split)
            for img in img_dir.glob("*.jpg"):
                boxes = self._read_label_boxes(lbl_dir / f"{img.stem}.txt")
                classes = sorted({b["cls"] for b in boxes})
                for b in boxes:
                    per_class[b["cls"]] = per_class.get(b["cls"], 0) + 1
                items.append({"id": img.stem, "split": split, "n": len(boxes), "cls": classes})
                counts[split] += 1
        items.sort(key=lambda it: it["id"])
        return {
            "ok": True,
            "game": dm.game.name,
            "classes": dm.game.classes,
            "items": items,
            "counts": counts,
            "boxesPerClass": {str(k): v for k, v in sorted(per_class.items())},
        }

    def dataset_image_path(self, split: str, sample_id: str):
        if split not in self._DATASET_SPLITS or not self._SAMPLE_ID_RE.match(sample_id or ""):
            return None
        img_dir, _ = self._dataset_dirs(split)
        path = img_dir / f"{sample_id}.jpg"
        return path if path.is_file() else None

    def dataset_labels(self, split: str, sample_id: str):
        if self.dataset_image_path(split, sample_id) is None:
            return {"ok": False, "error": "Bild nicht gefunden."}
        _, lbl_dir = self._dataset_dirs(split)
        result = {"ok": True, "boxes": self._read_label_boxes(lbl_dir / f"{sample_id}.txt"), "meta": None}

        # Herkunft (Video/Zeitpunkt/Konfidenz) steht in den Metadaten, die beim Review-Ordner liegen bleiben
        meta_file = self._dm().review_metadata / f"{sample_id}.json"
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                result["meta"] = {
                    "video": meta.get("video_name") or None,
                    "timestampSec": meta.get("timestamp_sec") or None,
                    "frame": meta.get("frame_number"),
                    "confidence": meta.get("confidence"),
                    "modelVersion": meta.get("model_version") or None,
                }
            except (OSError, ValueError):
                pass
        return result

    def review_get_labels(self, sample_id: str):
        dm = self._dm()
        label_file = dm.review_labels / f"{sample_id}.txt"
        boxes = []
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                boxes.append({"cls": int(cls), "x": float(x), "y": float(y), "w": float(w), "h": float(h)})
        return {"boxes": boxes}

    def review_save_labels(self, sample_id: str, boxes: list):
        dm = self._dm()
        label_file = dm.review_labels / f"{sample_id}.txt"
        lines = [f"{b['cls']} {b['x']} {b['y']} {b['w']} {b['h']}" for b in boxes]
        label_file.write_text("\n".join(lines))
        return {"ok": True}

    def _mark_meta(self, sample_id: str, accepted: bool):
        dm = self._dm()
        meta_file = dm.review_metadata / f"{sample_id}.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["reviewed"] = True
            data["accepted"] = accepted
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _retry_locked(action, attempts: int = 6, delay: float = 0.25):
        """Windows sperrt Dateien kurz, solange der Browser sie noch laedt: kurz erneut versuchen."""
        for attempt in range(attempts):
            try:
                return action()
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)

    def review_accept(self, sample_id: str):
        dm = self._dm()
        img_path = self.find_review_image_path(sample_id)
        if img_path is None:
            return {"ok": False, "error": "Bild nicht gefunden (evtl. schon bearbeitet)."}

        label_path = dm.review_labels / f"{sample_id}.txt"
        try:
            self._mark_meta(sample_id, True)
            self._retry_locked(lambda: dm.route_promoted_sample(img_path, label_path))
        except OSError as e:
            return {"ok": False, "error": f"Accept fehlgeschlagen: {e}"}

        self._review_images = [p for p in self._review_images if p.stem != sample_id]
        return {"ok": True, "remaining": len(self._review_images)}

    def review_reject(self, sample_id: str):
        dm = self._dm()
        img_path = self.find_review_image_path(sample_id)
        if img_path is None:
            return {"ok": False, "error": "Bild nicht gefunden (evtl. schon bearbeitet)."}

        try:
            self._mark_meta(sample_id, False)
            self._retry_locked(lambda: dm.discard_sample(sample_id))
        except OSError as e:
            return {"ok": False, "error": f"Reject fehlgeschlagen: {e}"}

        self._review_images = [p for p in self._review_images if p.stem != sample_id]
        return {"ok": True, "remaining": len(self._review_images)}


core = Core()
app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/state")
def api_state():
    return jsonify(core.get_state())


@app.route("/api/select_game", methods=["POST"])
def api_select_game():
    return jsonify(core.select_game(request.get_json(force=True)["name"]))


@app.route("/api/refresh_games", methods=["POST"])
def api_refresh_games():
    return jsonify(core.refresh_games())


@app.route("/api/apply_settings", methods=["POST"])
def api_apply_settings():
    return jsonify(core.apply_settings(request.get_json(force=True) or {}))


@app.route("/api/reset_settings", methods=["POST"])
def api_reset_settings():
    return jsonify(core.reset_settings())


@app.route("/api/video_thumb")
def api_video_thumb():
    """Vorschaubild fuer ein ausgewaehltes Video (erzeugt per ffmpeg, danach aus dem Cache)."""
    path = request.args.get("path", "")
    thumb = make_video_thumbnail(path)
    if thumb is None:
        abort(404)
    resp = send_file(thumb, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


@app.route("/api/pick_files", methods=["POST"])
def api_pick_files():
    data = request.get_json(force=True)
    kind = data.get("kind")
    dialogs = {
        "video": ("Videos auswählen", "Videos (*.mp4;*.mkv;*.avi;*.mov)|*.mp4;*.mkv;*.avi;*.mov|Alle Dateien (*.*)|*.*", True),
        "model": ("YOLO-Modell auswählen", "YOLO-Modell (*.pt)|*.pt", False),
        "json": ("Klassen-JSON auswählen", "JSON (*.json)|*.json", False),
    }
    if kind not in dialogs:
        return jsonify({"ok": False, "error": "Unbekannter Dialog-Typ."})

    title, filter_str, multiple = dialogs[kind]
    paths = core.pick_native_files(title, filter_str, multiple)
    return jsonify({"ok": True, "paths": paths})


@app.route("/api/extract_classes_from_json", methods=["POST"])
def api_extract_classes_from_json():
    return jsonify(core.extract_classes_from_json(request.get_json(force=True)["path"]))


@app.route("/api/add_game", methods=["POST"])
def api_add_game():
    data = request.get_json(force=True)
    return jsonify(core.add_game(
        data.get("name"), data.get("modelPath"), data.get("classes"), data.get("videoPaths"),
    ))


@app.route("/api/game_info")
def api_game_info():
    return jsonify(core.get_game_info(request.args.get("name", "")))


@app.route("/api/delete_game", methods=["POST"])
def api_delete_game():
    d = request.get_json(force=True)
    return jsonify(core.delete_game(d.get("name"), d.get("confirmName")))


@app.route("/api/edit_game", methods=["POST"])
def api_edit_game():
    d = request.get_json(force=True)
    return jsonify(core.edit_game(d.get("name"), d.get("classes"), d.get("adaptiveThresholds")))


@app.route("/api/start_processing", methods=["POST"])
def api_start_processing():
    return jsonify(core.start_processing(request.get_json(force=True).get("videos", [])))


@app.route("/api/run_ml_pipeline", methods=["POST"])
def api_run_ml_pipeline():
    data = request.get_json(force=True)
    return jsonify(core.run_ml_pipeline(
        data.get("videos", []), bool(data.get("exportFcpxml", False)), bool(data.get("mlOnly", False)),
        data.get("collectFrames"),
    ))


@app.route("/api/test_model", methods=["POST"])
def api_test_model():
    d = request.get_json(force=True)
    return jsonify(core.test_model(
        d.get("videoPath"), d.get("confidence", 0.4), d.get("sampleEveryN", 10), d.get("maxMinutes", 0)
    ))


@app.route("/api/open_test_output", methods=["POST"])
def api_open_test_output():
    return jsonify(core.open_last_test_output(bool(request.get_json(force=True).get("folder", False))))


@app.route("/api/export_fcpxml", methods=["POST"])
def api_export_fcpxml():
    return jsonify(core.export_fcpxml(request.get_json(force=True).get("videos", [])))


@app.route("/api/training_overview")
def api_training_overview():
    return jsonify(core.get_training_overview())


@app.route("/api/run_training", methods=["POST"])
def api_run_training():
    return jsonify(core.run_training())


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    return jsonify(core.cancel_current_operation())


@app.route("/api/events")
def api_events():
    def gen():
        while True:
            item = core.event_queue.get()
            yield f"data: {json.dumps(item)}\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/review/open", methods=["POST"])
def api_review_open():
    return jsonify(core.review_open())


@app.route("/api/review/image/<sample_id>")
def api_review_image(sample_id):
    p = core.find_review_image_path(sample_id)
    if p is None or not p.exists():
        abort(404)
    return send_file(p)


@app.route("/api/dataset/list")
def api_dataset_list():
    return jsonify(core.dataset_list())


@app.route("/api/dataset/image/<split>/<sample_id>")
def api_dataset_image(split, sample_id):
    p = core.dataset_image_path(split, sample_id)
    if p is None:
        abort(404)
    return send_file(p, max_age=0)


@app.route("/api/dataset/labels/<split>/<sample_id>")
def api_dataset_labels(split, sample_id):
    return jsonify(core.dataset_labels(split, sample_id))


@app.route("/api/review/labels/<sample_id>")
def api_review_labels(sample_id):
    return jsonify(core.review_get_labels(sample_id))


@app.route("/api/review/labels/<sample_id>", methods=["POST"])
def api_review_save_labels(sample_id):
    return jsonify(core.review_save_labels(sample_id, request.get_json(force=True).get("boxes", [])))


@app.errorhandler(Exception)
def _api_error(exc):
    """API-Fehler immer als JSON melden, damit das Frontend sie anzeigen kann (statt HTML-Fehlerseite)."""
    from werkzeug.exceptions import HTTPException

    if not request.path.startswith("/api/"):
        if isinstance(exc, HTTPException):
            return exc
        raise exc
    if isinstance(exc, HTTPException):
        return jsonify({"ok": False, "error": exc.description}), exc.code
    print(f"[API-Fehler] {request.path}: {type(exc).__name__}: {exc}")
    return jsonify({"ok": False, "error": f"Interner Fehler: {exc}"}), 500


@app.route("/api/review/accept/<sample_id>", methods=["POST"])
def api_review_accept(sample_id):
    return jsonify(core.review_accept(sample_id))


@app.route("/api/review/reject/<sample_id>", methods=["POST"])
def api_review_reject(sample_id):
    return jsonify(core.review_reject(sample_id))


def main():
    try:
        from config.games import GAMES as _  # loest Validierung von games.json aus
    except Exception as e:
        print(f"Konfigurationsfehler in config/games.json: {e}")
        raise SystemExit(1)

    host, port = "127.0.0.1", 8765
    url = f"http://{host}:{port}"
    print(f"Highlight Cutter Web-UI laeuft auf {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
