import os
import io
import sys
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from app.review.review_window import ReviewWindow 
from app.dataset.dataset_manager import DatasetManager
from app.training.trainer import Trainer 
from app.cutter.highlight_cutter_pointbased import Config, process_video
from app.core.pipeline_controller import PipelineController
from config.games import GAMES, reload_games, GamesConfigError
class QueueWriter(io.TextIOBase):
    def __init__(self, log_queue: queue.Queue, real_stdout):
        self.log_queue = log_queue
        self.real_stdout = real_stdout

    def write(self, msg):
        if msg:
            self.log_queue.put(msg)
            if self.real_stdout is not None:
                try:
                    self.real_stdout.write(msg)
                except Exception:
                    pass
        return len(msg)

    def flush(self):
        if self.real_stdout is not None:
            try:
                self.real_stdout.flush()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Einstellungs-Fenster
# ----------------------------------------------------------------------

class SettingsWindow(tk.Toplevel):
    """
    Separates Fenster mit allen Config-Parametern als editierbare Felder.
    Scrollbar fuer alle ~25 Parameter. Aenderungen werden erst beim
    Klick auf "Uebernehmen" aktiv.
    """

    # (name, label, typ, beschreibung)
    PARAMS = [
        # Audio
        ("audio_threshold_percentile",  "Audio Peak-Threshold Percentile",    float, "Ab welchem RMS-Percentile ein Audio-Punkt zaehlt (hoch = nur sehr laute Stellen)"),
        ("cluster_max_gap_sec",         "Cluster Max. Abstand (s)",            float, "Max. Abstand (s) zwischen Punkten im selben Cluster"),
        ("audio_pre_buffer_sec",        "Audio Pre-Buffer (s)",                float, "Puffer VOR einem Audio-Peak-Cluster"),
        ("yolo_pre_buffer_sec",         "YOLO Pre-Buffer (s)",                 float, "Puffer VOR einem YOLO-Event-Cluster"),
        ("base_post_buffer_sec",        "Basis Post-Buffer (s)",               float, "Basis-Nachlaufzeit nach letztem Punkt im Cluster"),
        ("loud_threshold_percentile",   "Lautstaerke Threshold Percentile",    float, "Ab wann ein Sample im Fenster als 'laut' gilt (fuer Post-Buffer-Wachstum)"),
        ("loud_growth_per_sec",         "Lautstaerke Wachstum pro Sek.",       float, "Post-Buffer-Zuwachs pro Sekunde lauter Zeit im Fenster"),
        ("loud_growth_max_sec",         "Lautstaerke Wachstum Max. (s)",       float, "Kappung des lautstaerke-bedingten Post-Buffer-Zuwachses"),
        ("size_bonus_per_point_sec",    "Cluster-Groessen-Bonus pro Punkt (s)",float, "Bonus-Sekunden pro zusaetzlichem Punkt im Cluster"),
        ("size_bonus_max_sec",          "Cluster-Groessen-Bonus Max. (s)",     float, "Kappung des Groessen-Bonus"),
        ("density_bonus_enabled",       "Dichte-Bonus aktiv",                  bool,  "Ob dichter beieinanderliegende Punkte einen staerkeren Bonus bekommen"),
        ("density_reference_gap_sec",   "Dichte Referenz-Abstand (s)",         float, "Bei diesem Abstand = Bonus-Faktor 1.0"),
        ("density_bonus_max_multiplier","Dichte Max. Multiplikator",           float, "Maximaler Multiplikator auf den Groessen-Bonus bei sehr dichten Clustern"),
        ("min_cluster_confidence_enabled","Rausch-Filter aktiv",               bool,  "Ob isolierte Einzel-Punkte ein kuerzeres Fenster bekommen"),
        ("isolated_point_post_buffer_sec","Einzel-Punkt Post-Buffer (s)",      float, "Post-Buffer fuer isolierte Einzel-Punkte (wenn Rausch-Filter aktiv)"),
        ("discard_isolated_audio_points","Isolierte Audio-Punkte verwerfen",   bool,  "Isolierte Audio-Einzel-Punkte komplett verwerfen"),
        ("max_clip_duration_sec",       "Max. Clip-Laenge (s)",                float, "Sicherheitslimit: maximale Laenge eines einzelnen Clips"),
        # Stufe 2
        ("enable_internal_gap_removal", "Stufe 2: Luecken entfernen",          bool,  "Lange Stille-Abschnitte innerhalb eines Clips herausschneiden"),
        ("internal_silence_threshold_percentile","Stufe 2: Stille-Threshold Percentile", float, "Unter diesem Percentile gilt ein Sample als 'still'"),
        ("internal_min_gap_sec",        "Stufe 2: Min. Luecken-Laenge (s)",    float, "Mindest-Stilledauer bevor intern geschnitten wird"),
        ("internal_gap_padding_sec",    "Stufe 2: Luecken-Padding (s)",        float, "Puffer vor/nach der internen Luecke der erhalten bleibt"),
        # Stufe 3
        ("enable_micro_trim",           "Stufe 3: Mikro-Trimming aktiv",       bool,  "Schnittkanten auf naechste leise Stelle snappen"),
        ("micro_trim_search_window_sec","Stufe 3: Trim Suchfenster (s)",       float, "Wie weit links/rechts nach einer leisen Stelle gesucht wird"),
        ("micro_trim_silence_percentile","Stufe 3: Trim Stille-Percentile",    float, "Unter diesem Percentile gilt eine Stelle als 'leise' zum Snappen"),
        # Merge
        ("merge_gap_sec",               "Merge: Max. Abstand (s)",             float, "Cluster-Fenster die naeher beieinander liegen werden zusammengefuehrt"),
        # Encoding
        ("use_gpu_encoding",            "GPU-Encoding (NVENC)",                bool,  "RTX-GPU fuer schnelles Encoding nutzen"),
        ("nvenc_cq",                    "NVENC Qualitaet (CQ, niedriger=besser)",int, "Qualitaets-Wert fuer GPU-Encoding (15=sehr gut, 19=gut)"),
        ("crf",                         "CPU CRF (niedriger=besser)",          int,   "Qualitaets-Wert fuer CPU-Encoding (14=sehr gut, 18=gut)"),
        # YOLO
        ("yolo_sample_every_n_frames",  "YOLO: jeden N-ten Frame pruefen",     int,   "Performanz: nur jeden N-ten Frame durch YOLO analysieren"),
        ("yolo_confidence",             "YOLO: Mindest-Konfidenz",             float, "Mindest-Erkennungssicherheit fuer YOLO-Treffer (0.0-1.0)"),
    ]

    def __init__(self, parent, config: Config, on_apply):
        super().__init__(parent)
        self.title("Einstellungen")
        self.geometry("700x600")
        self.resizable(True, True)
        self.on_apply = on_apply
        self.config_ref = config

        self.grab_set()  # modales Fenster

        # --- Scrollbarer Bereich ---
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mausrad-Scrollen
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        # --- Parameter-Felder aufbauen ---
        self.vars = {}
        for i, (attr, label, typ, desc) in enumerate(self.PARAMS):
            row = tk.Frame(self.scroll_frame)
            row.pack(fill="x", padx=10, pady=3)

            tk.Label(row, text=label, width=38, anchor="w", font=("", 9, "bold")).pack(side="left")

            current_val = getattr(config, attr)

            if typ == bool:
                var = tk.BooleanVar(value=current_val)
                tk.Checkbutton(row, variable=var).pack(side="left")
            else:
                var = tk.StringVar(value=str(current_val))
                tk.Entry(row, textvariable=var, width=10).pack(side="left")

            tk.Label(row, text=desc, fg="gray", font=("", 8), anchor="w").pack(side="left", padx=8)
            self.vars[attr] = (var, typ)

        # --- Buttons ---
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=8)

        tk.Button(btn_frame, text="Uebernehmen", command=self._apply, width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Abbrechen",   command=self.destroy, width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Zuruecksetzen", command=self._reset, width=15).pack(side="left", padx=5)
        

    def _apply(self):
        errors = []
        for attr, (var, typ) in self.vars.items():
            try:
                if typ == bool:
                    setattr(self.config_ref, attr, var.get())
                else:
                    setattr(self.config_ref, attr, typ(var.get()))
            except ValueError:
                errors.append(attr)

        if errors:
            messagebox.showerror("Ungueltige Werte", f"Folgende Felder haben ungueltige Werte:\n" + "\n".join(errors))
            return

        self.on_apply(self.config_ref)
        messagebox.showinfo("Gespeichert", "Einstellungen uebernommen.")
        self.destroy()

    def _reset(self):
        defaults = Config()
        for attr, (var, typ) in self.vars.items():
            val = getattr(defaults, attr)
            if typ == bool:
                var.set(val)
            else:
                var.set(str(val))


# ----------------------------------------------------------------------
# Haupt-GUI
# ----------------------------------------------------------------------

class HighlightCutterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Highlight Cutter")
        self.root.geometry("900x750")

        self.video_files = []
        self.log_queue = queue.Queue()

        # Config-Objekt (wird vom Einstellungs-Fenster veraendert)
        self.cfg = Config(
            yolo_model_path=None
        )

        # Cache: haelt pro Spiel genau einen DatasetManager/PipelineController vor,
        # damit nicht bei jedem Klick neu instanziiert wird
        self._dataset_managers = {}
        self._pipeline_controllers = {}

        model_frame = tk.Frame(root)
        model_frame.pack(pady=(0, 5), fill="x", padx=10)

        tk.Label(model_frame, text="Spiel / YOLO-Modell:").pack(side="left")

        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            state="readonly",
            width=40,
        )
        self.model_dropdown.pack(side="left", padx=5)

        tk.Button(
            model_frame, text="🔄", width=3,
            command=self.refresh_games
        ).pack(side="left", padx=2)

        self._refresh_game_dropdown()
        self._refresh_game_dropdown()

        tk.Label(root, text="Videos").pack(pady=5)

        self.listbox = tk.Listbox(root, width=110, height=10)
        self.listbox.pack(fill="both", expand=False, padx=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Videos auswaehlen", command=self.select_videos
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame, text="Liste leeren", command=self.clear_list
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame, text="Einstellungen", command=self.open_settings
        ).pack(side="left", padx=5)

        self.start_btn = tk.Button(
            btn_frame, text="Start", command=self.start_processing
        )
        self.start_btn.pack(side="left", padx=5)

        tk.Button(
            btn_frame, text="🚀 Process + ML Pipeline",
            command=self.run_ml_pipeline, bg="green", fg="white"
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame, text="🧠 Train Model",
            command=self.run_training, bg="blue", fg="white"
        ).pack(side="left", padx=5)

        self.review_btn = tk.Button(
            btn_frame, text="🧾 Open Review Queue",
            command=self.open_review, bg="orange", fg="black"
        )
        self.review_btn.pack(side="left", padx=5)

        # --- Gesamt-Fortschritt (Video X/Y) ---
        tk.Label(root, text="Gesamt-Fortschritt (Video X von Y)").pack(anchor="w", padx=10)
        self.progress = ttk.Progressbar(root, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(0, 5))

        # --- Sub-Fortschritt (aktuelle Phase im aktuellen Video) ---
        self.sub_phase_label = tk.StringVar(value="Phase: -")
        tk.Label(root, textvariable=self.sub_phase_label).pack(anchor="w", padx=10)
        self.sub_progress = ttk.Progressbar(root, mode="determinate")
        self.sub_progress.pack(fill="x", padx=10, pady=(0, 5))

        self.status = tk.StringVar(value="Bereit")
        tk.Label(root, textvariable=self.status).pack(pady=(5, 0))

        # --- Live-Log-Fenster ---
        tk.Label(root, text="Live-Log").pack(anchor="w", padx=10, pady=(10, 0))

        log_frame = tk.Frame(root)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=18, wrap="none", bg="#111", fg="#ddd")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.root.after(100, self._drain_log_queue)
        
    # ------------------------------------------------------------
    # UI-Aktionen
    # ------------------------------------------------------------

    def select_videos(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Videos", "*.mp4 *.mkv *.avi *.mov")]
        )
        for f in files:
            if f not in self.video_files:
                self.video_files.append(f)
                self.listbox.insert(tk.END, f)

    def clear_list(self):
        self.video_files.clear()
        self.listbox.delete(0, tk.END)

    def open_settings(self):
        SettingsWindow(self.root, self.cfg, on_apply=self._on_settings_apply)

    def _refresh_game_dropdown(self):
        game_names = list(GAMES.keys())
        if not game_names:
            messagebox.showerror(
                "Keine Spiele konfiguriert",
                "config/games.json enthaelt keine Eintraege unter 'games'."
            )
            return

        self.model_dropdown["values"] = game_names
        if self.model_var.get() not in game_names:
            self.model_var.set(game_names[0])

    def refresh_games(self):
        """Laedt config/games.json neu ein, ohne die GUI neu zu starten."""
        try:
            reload_games()
        except (GamesConfigError, FileNotFoundError) as e:
            messagebox.showerror("Fehler in games.json", str(e))
            return

        self._dataset_managers.clear()
        self._pipeline_controllers.clear()
        self._refresh_game_dropdown()
        messagebox.showinfo("Aktualisiert", f"{len(GAMES)} Spiele geladen.")

    def _current_game_name(self):
        return self.model_var.get()

    def _get_dataset_manager(self):
        game_name = self._current_game_name()
        if game_name not in self._dataset_managers:
            self._dataset_managers[game_name] = DatasetManager(game_name)
        return self._dataset_managers[game_name]

    def _get_pipeline_controller(self):
        game_name = self._current_game_name()
        if game_name not in self._pipeline_controllers:
            dm = self._get_dataset_manager()
            self._pipeline_controllers[game_name] = PipelineController(
                game_name=game_name,
                model_path=str(dm.game.model_path),
            )
        return self._pipeline_controllers[game_name]

    def _sync_model_path(self):
        dm = self._get_dataset_manager()
        self.cfg.yolo_model_path = str(dm.game.model_path)
    
    def _on_settings_apply(self, updated_cfg: Config):
        self.cfg = updated_cfg

    def start_processing(self):
        if not self.video_files:
            messagebox.showwarning("Hinweis", "Keine Videos ausgewaehlt.")
            return

        self.start_btn.config(state="disabled")
        self.model_dropdown.config(state="disabled")
        self.log_text.delete("1.0", tk.END)
        self.progress["value"] = 0
        self.sub_progress["value"] = 0

        threading.Thread(target=self.process_all, daemon=True).start()

    # ------------------------------------------------------------
    # Live-Log
    # ------------------------------------------------------------

    def _drain_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._drain_log_queue)

    # ------------------------------------------------------------
    # Progress-Callback
    # ------------------------------------------------------------

    def _make_progress_callback(self):
        def callback(phase: str, percent: float):
            def update():
                self.sub_phase_label.set(f"Phase: {phase} ({percent:.0f}%)")
                self.sub_progress["value"] = percent
            self.root.after(0, update)
        return callback

    # ------------------------------------------------------------
    # Verarbeitung
    # ------------------------------------------------------------

    def process_all(self):
        real_stdout = sys.stdout
        sys.stdout = QueueWriter(self.log_queue, real_stdout)

        try:
            output_dir = r"D:\EDITED CLIPS\Skript Cutted\Fertig zur Finalisierung"
            os.makedirs(output_dir, exist_ok=True)

            # YOLO-Modell aus Dropdown in Config uebernehmen
            self._sync_model_path()

            total = len(self.video_files)
            progress_callback = self._make_progress_callback()

            for idx, video in enumerate(self.video_files, start=1):
                self.status.set(f"Bearbeite {idx}/{total}: {os.path.basename(video)}")
                self.sub_progress["value"] = 0

                base = os.path.splitext(os.path.basename(video))[0]
                output_path = os.path.join(output_dir, f"{base}_highlight.mp4")

                process_video(
                    video_path=video,
                    output_path=output_path,
                    cfg=self.cfg,
                    progress_callback=progress_callback,
                )

                self.progress["value"] = idx / total * 100

            self.status.set("Fertig")
            self.sub_phase_label.set("Phase: -")
            messagebox.showinfo("Fertig", "Alle Videos wurden verarbeitet.")

        except Exception as e:
            messagebox.showerror("Fehler", str(e))

        finally:
            sys.stdout = real_stdout
            self.start_btn.config(state="normal")
            self.model_dropdown.config(state="readonly")

    def open_review(self):
        dm = self._get_dataset_manager()
        ReviewWindow(
            self.root,
            str(dm.review),
            dataset_manager=dm,
            classes=dm.game.classes,
        )

    def run_training(self):
        self.start_btn.config(state="disabled")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        real_stdout = sys.stdout
        sys.stdout = QueueWriter(self.log_queue, real_stdout)
        try:
            dm = self._get_dataset_manager()
            trainer = Trainer(dm, model_path=str(dm.game.model_path))
            self.status.set("Training laeuft...")
            trainer.train(epochs=30, img_size=640)
            self.status.set("Training fertig")
            messagebox.showinfo("Training", "Retraining abgeschlossen.")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
        finally:
            sys.stdout = real_stdout
            self.start_btn.config(state="normal")

    def run_ml_pipeline(self):
        if not self.video_files:
            messagebox.showwarning("Hinweis", "Keine Videos ausgewaehlt.")
            return
        self.start_btn.config(state="disabled")
        threading.Thread(target=self._ml_pipeline_worker, daemon=True).start()
        
    def _ml_pipeline_worker(self):
        real_stdout = sys.stdout
        sys.stdout = QueueWriter(self.log_queue, real_stdout)
        try:
            self._sync_model_path()
            output_dir = r"D:\EDITED CLIPS\Skript Cutted\Fertig zur Finalisierung"
            os.makedirs(output_dir, exist_ok=True)

            total_extracted = total_annotated = total_needs_review = 0

            for video in self.video_files:
                self.status.set(f"ML-Pipeline: {os.path.basename(video)}")

                base = os.path.splitext(os.path.basename(video))[0]
                output_path = os.path.join(output_dir, f"{base}_highlight.mp4")

                result = process_video(
                    video_path=video,
                    output_path=output_path,
                    cfg=self.cfg,
                    progress_callback=self._make_progress_callback(),
                )

                stats = self._get_pipeline_controller().run_full_cycle(
                    video_path=video,
                    events=result["yolo_event_times"],
                )

                total_extracted += stats["extracted"]
                total_annotated += stats["annotated"]
                total_needs_review += stats["needs_review"]

            self.status.set("ML-Pipeline fertig")
            self._update_review_badge(total_needs_review)

            messagebox.showinfo(
                "ML-Pipeline abgeschlossen",
                f"{total_extracted} Frames extrahiert\n"
                f"{total_annotated} automatisch annotiert\n"
                f"{total_needs_review} Frames benoetigen manuelle Korrektur",
            )

        except Exception as e:
            messagebox.showerror("Fehler", str(e))
        finally:
            sys.stdout = real_stdout
            self.start_btn.config(state="normal")

    def _update_review_badge(self, count: int):
        label = "🧾 Open Review Queue" if count == 0 else f"🧾 Review Queue ({count})"
        self.review_btn.config(text=label)
        
   
if __name__ == "__main__":
    try:
        from config.games import GAMES  # loest Validierung aus, falls noch nicht geschehen
    except Exception as e:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Konfigurationsfehler", str(e))
        raise SystemExit(1)

    root = tk.Tk()
    HighlightCutterGUI(root)
    root.mainloop()