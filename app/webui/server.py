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

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog
import webbrowser
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from app.cutter.highlight_cutter_pointbased import Config, process_video
from app.core.pipeline_controller import PipelineController
from app.dataset.dataset_manager import DatasetManager
from app.training.trainer import Trainer
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
]


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
        self._dataset_managers = {}
        self._pipeline_controllers = {}
        self._review_images: list[Path] = []
        self.event_queue: "queue.Queue[dict]" = queue.Queue()
        self.cancel_event = None

    # ------------------------------------------------------------
    # Interna
    # ------------------------------------------------------------

    def _push(self, event: str, data=None):
        self.event_queue.put({"event": event, "data": data})

    def _dm(self) -> DatasetManager:
        if self.current_game not in self._dataset_managers:
            self._dataset_managers[self.current_game] = DatasetManager(self.current_game)
        return self._dataset_managers[self.current_game]

    def _pc(self) -> PipelineController:
        if self.current_game not in self._pipeline_controllers:
            dm = self._dm()
            self._pipeline_controllers[self.current_game] = PipelineController(
                game_name=self.current_game,
                model_path=str(dm.game.model_path),
            )
        return self._pipeline_controllers[self.current_game]

    def _sync_model_path(self):
        self.cfg.yolo_model_path = str(self._dm().game.model_path)

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
        if self.current_game:
            dm = self._dm()
            classes = dm.game.classes
            model_path = str(dm.game.model_path)
            review_count = self._review_count()

        return {
            "games": list(GAMES.keys()),
            "currentGame": self.current_game,
            "classes": classes,
            "modelPath": model_path,
            "reviewCount": review_count,
            "settingsSchema": [
                {"attr": a, "label": l, "type": t, "desc": d, "value": getattr(self.cfg, a)}
                for (a, l, t, d) in SETTINGS_SCHEMA
            ],
        }

    def select_game(self, name: str):
        if name not in GAMES:
            return {"ok": False, "error": f"Unbekanntes Spiel: {name}"}
        self.current_game = name
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
        return {"ok": True, "state": self.get_state()}

    # ------------------------------------------------------------
    # Einstellungen
    # ------------------------------------------------------------

    def apply_settings(self, values: dict):
        errors = []
        for attr, _label, typ, _desc in SETTINGS_SCHEMA:
            if attr not in values:
                continue
            raw = values[attr]
            try:
                if typ == "bool":
                    setattr(self.cfg, attr, bool(raw))
                elif typ == "int":
                    setattr(self.cfg, attr, int(raw))
                else:
                    setattr(self.cfg, attr, float(raw))
            except (ValueError, TypeError):
                errors.append(attr)

        return {"ok": not errors, "errors": errors}

    def reset_settings(self):
        defaults = Config()
        for attr, _label, _typ, _desc in SETTINGS_SCHEMA:
            setattr(self.cfg, attr, getattr(defaults, attr))
        return {
            "ok": True,
            "settingsSchema": [
                {"attr": a, "label": l, "type": t, "desc": d, "value": getattr(self.cfg, a)}
                for (a, l, t, d) in SETTINGS_SCHEMA
            ],
        }

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

    def add_game(self, name: str, model_source_path: str, classes: list):
        name = (name or "").strip()
        classes = [c.strip() for c in (classes or []) if c and c.strip()]

        if not name:
            return {"ok": False, "error": "Bitte einen Spielnamen eingeben."}
        if name in GAMES:
            return {"ok": False, "error": f"Ein Spiel namens '{name}' existiert bereits."}
        if not model_source_path or not Path(model_source_path).is_file():
            return {"ok": False, "error": "Bitte ein gueltiges YOLO-Modell (.pt) auswaehlen."}
        if not classes:
            return {"ok": False, "error": "Bitte mindestens eine Klasse angeben."}
        if len(classes) != len(set(classes)):
            return {"ok": False, "error": "Doppelte Klassennamen gefunden."}

        try:
            model_dst = PROJECT_ROOT / "models" / name / "current.pt"
            model_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_source_path, model_dst)

            with open(GAMES_CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)

            raw.setdefault("games", {})[name] = {
                "model_path": f"models/{name}/current.pt",
                "dataset_path": f"datasets/{name}",
                "review_queue_path": f"review_queue/{name}",
                "classes": classes,
            }

            with open(GAMES_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)

            reload_games()
            DatasetManager(name)  # legt Ordnerstruktur (images/labels/train/val) sofort an
        except (GamesConfigError, OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}

        self.current_game = name
        return {"ok": True, "state": self.get_state()}

    # ------------------------------------------------------------
    # Verarbeitung (Hintergrund-Threads, Fortschritt/Log via SSE-Queue)
    # ------------------------------------------------------------

    def start_processing(self, video_paths: list):
        if not video_paths:
            return {"ok": False, "error": "Keine Videos ausgewaehlt."}
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

    def run_ml_pipeline(self, video_paths: list):
        if not video_paths:
            return {"ok": False, "error": "Keine Videos ausgewaehlt."}
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}
        self.cancel_event = threading.Event()
        threading.Thread(target=self._ml_pipeline_worker, args=(list(video_paths), self.cancel_event), daemon=True).start()
        return {"ok": True}

    def _ml_pipeline_worker(self, video_paths, cancel_event):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self._sync_model_path()
            total = len(video_paths)
            total_extracted = total_annotated = total_needs_review = 0

            for idx, video in enumerate(video_paths, start=1):
                if cancel_event.is_set():
                    raise RuntimeError("Vorgang abgebrochen.")

                self._push("status", f"ML-Pipeline: {os.path.basename(video)}")
                base = os.path.splitext(os.path.basename(video))[0]
                output_path = os.path.join(OUTPUT_DIR, f"{base}_highlight.mp4")

                result = process_video(
                    video_path=video,
                    output_path=output_path,
                    cfg=self.cfg,
                    progress_callback=lambda phase, pct: self._push("progress", {"phase": phase, "percent": pct}),
                    cancel_event=cancel_event,
                )

                stats = self._pc().run_full_cycle(video_path=video, events=result["yolo_event_times"])
                total_extracted += stats["extracted"]
                total_annotated += stats["annotated"]
                total_needs_review += stats["needs_review"]
                self._push("overallProgress", {"idx": idx, "total": total})

            self._push("reviewCount", self._review_count())
            self._push("status", "ML-Pipeline fertig")
            self._push("done", {
                "message": (
                    f"{total_extracted} Frames extrahiert\n"
                    f"{total_annotated} automatisch annotiert\n"
                    f"{total_needs_review} Frames benoetigen manuelle Korrektur"
                ),
            })
        except Exception as e:
            self._push("cancelled" if cancel_event.is_set() else "error", str(e))
        finally:
            sys.stdout = real_stdout
            self.cancel_event = None
            self._push("busy", False)

    def run_training(self):
        if self.cancel_event is not None:
            return {"ok": False, "error": "Es laeuft bereits ein Vorgang."}
        self.cancel_event = threading.Event()
        threading.Thread(target=self._train_worker, args=(self.cancel_event,), daemon=True).start()
        return {"ok": True}

    def _train_worker(self, cancel_event):
        real_stdout = sys.stdout
        sys.stdout = _LogStream(self)
        self._push("busy", True)
        try:
            dm = self._dm()
            trainer = Trainer(dm, model_path=str(dm.game.model_path))
            self._push("status", "Training laeuft...")
            trainer.train(epochs=30, img_size=640, cancel_event=cancel_event)
            self._push("status", "Training fertig")
            self._push("done", {"message": "Retraining abgeschlossen."})
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

    def find_review_image_path(self, sample_id: str):
        for p in self._review_images:
            if p.stem == sample_id:
                return p
        return None

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

    def review_accept(self, sample_id: str):
        dm = self._dm()
        img_path = self.find_review_image_path(sample_id)
        if img_path is None:
            return {"ok": False, "error": "Bild nicht gefunden."}

        label_path = dm.review_labels / f"{sample_id}.txt"
        self._mark_meta(sample_id, True)
        dm.route_promoted_sample(img_path, label_path)

        self._review_images = [p for p in self._review_images if p.stem != sample_id]
        return {"ok": True, "remaining": len(self._review_images)}

    def review_reject(self, sample_id: str):
        dm = self._dm()
        img_path = self.find_review_image_path(sample_id)
        if img_path is None:
            return {"ok": False, "error": "Bild nicht gefunden."}

        label_path = dm.review_labels / f"{sample_id}.txt"
        meta_path = dm.review_metadata / f"{sample_id}.json"
        self._mark_meta(sample_id, False)

        discard_dir = dm.review / "discarded"
        discard_dir.mkdir(exist_ok=True)
        img_path.rename(discard_dir / img_path.name)
        if label_path.exists():
            label_path.rename(discard_dir / label_path.name)
        if meta_path.exists():
            meta_path.rename(discard_dir / meta_path.name)

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
    return jsonify(core.add_game(data.get("name"), data.get("modelPath"), data.get("classes")))


@app.route("/api/start_processing", methods=["POST"])
def api_start_processing():
    return jsonify(core.start_processing(request.get_json(force=True).get("videos", [])))


@app.route("/api/run_ml_pipeline", methods=["POST"])
def api_run_ml_pipeline():
    return jsonify(core.run_ml_pipeline(request.get_json(force=True).get("videos", [])))


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


@app.route("/api/review/labels/<sample_id>")
def api_review_labels(sample_id):
    return jsonify(core.review_get_labels(sample_id))


@app.route("/api/review/labels/<sample_id>", methods=["POST"])
def api_review_save_labels(sample_id):
    return jsonify(core.review_save_labels(sample_id, request.get_json(force=True).get("boxes", [])))


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
