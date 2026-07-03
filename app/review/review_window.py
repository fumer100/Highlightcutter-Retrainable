import tkinter as tk
from PIL import Image, ImageTk
import cv2
import json
from pathlib import Path
import math


class ReviewWindow(tk.Toplevel):
    
    HANDLE_SIZE = 6
    
    def __init__(self, parent, review_queue_path: str, dataset_manager=None, on_close=None):
        super().__init__(parent)
        self.title("Active Learning Review Tool - v3")
        self.on_close = on_close
        self.dm = dataset_manager

        self.review_path = Path(review_queue_path)
        self.image_paths = list((self.review_path / "images").glob("*.jpg"))
        self.label_dir = self.review_path / "labels"
        self.meta_dir = self.review_path / "metadata"

        self.index = 0
        self.boxes = []
        self.selected_box = None
        self.dragging = False
        self.resizing = False
        self.drag_offset = (0, 0)
        self.canvas_w = 1000
        self.canvas_h = 600
        self.zoom = 1.0
        
        self.counter_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.counter_var, fg="white", bg="black").pack()
        
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack()

        btn = tk.Frame(self)
        btn.pack()

        tk.Button(btn, text="Accept (A)", command=self.accept).pack(side=tk.LEFT)
        tk.Button(btn, text="Reject (D)", command=self.reject).pack(side=tk.LEFT)
        tk.Button(btn, text="Delete (Del)", command=self.delete_box).pack(side=tk.LEFT)
        tk.Button(btn, text="Next", command=self.next).pack(side=tk.LEFT)
        tk.Button(btn, text="Prev", command=self.prev).pack(side=tk.LEFT)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.bind("a", lambda e: self.accept())
        self.bind("d", lambda e: self.reject())
        self.bind("<Delete>", lambda e: self.delete_box())
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self.grab_set()
        self.load_image()

    def _on_window_close(self):
        if self.on_close:
            self.on_close()
        self.destroy()

    # -------------------------
    # IMAGE
    # -------------------------

    def load_image(self):
        self.counter_var.set(f"Bild {self.index + 1} / {len(self.image_paths)}")
        self.canvas.delete("all")

        if not self.image_paths:
            return

        img_path = self.image_paths[self.index]
        self.current_image_path = img_path

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.orig_h, self.orig_w = img.shape[:2]

        self.img = Image.fromarray(img)
        self.img = self.img.resize((self.canvas_w, self.canvas_h))

        self.tk_img = ImageTk.PhotoImage(self.img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.load_labels()
        self.draw_boxes()

    # -------------------------
    # LABELS
    # -------------------------

    def load_labels(self):
        self.selected_box = None
        self.boxes = []

        label_file = self.label_dir / f"{self.current_image_path.stem}.txt"

        if not label_file.exists():
            return

        with open(label_file, "r") as f:

            for line in f.readlines():

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls, x, y, w, h = parts

                self.boxes.append({
                    "cls": int(cls),
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h)
                })

    # -------------------------
    # DRAW
    # -------------------------

    def draw_boxes(self):
        for i, b in enumerate(self.boxes):
            x = b["x"] * self.canvas_w
            y = b["y"] * self.canvas_h
            w = b["w"] * self.canvas_w
            h = b["h"] * self.canvas_h

            color = "yellow" if self.selected_box == i else "red"

            self.canvas.create_rectangle(
                x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                outline=color, width=2,
                tags=("box", f"box_{i}")
            )

            if self.selected_box == i:
                for hx, hy in [
                    (x - w / 2, y - h / 2), (x + w / 2, y - h / 2),
                    (x - w / 2, y + h / 2), (x + w / 2, y + h / 2),
                ]:
                    self.canvas.create_rectangle(
                        hx - self.HANDLE_SIZE, hy - self.HANDLE_SIZE,
                        hx + self.HANDLE_SIZE, hy + self.HANDLE_SIZE,
                        fill="yellow", outline="black",
                        tags="handle"
                    )
                
    def _redraw_boxes_only(self):
        self.canvas.delete("box")
        self.canvas.delete("handle")
        self.draw_boxes()
    # -------------------------
    # SELECTION LOGIC
    # -------------------------

    def find_box(self, x, y):

        for i, b in enumerate(self.boxes):

            cx = b["x"] * self.canvas_w
            cy = b["y"] * self.canvas_h
            w = b["w"] * self.canvas_w
            h = b["h"] * self.canvas_h

            if (
                cx - w / 2 <= x <= cx + w / 2 and
                cy - h / 2 <= y <= cy + h / 2
            ):
                return i

        return None

    # -------------------------
    # MOUSE EVENTS
    # -------------------------


    def on_click(self, event):
        if self.selected_box is not None and self.selected_box < len(self.boxes):
            corner = self._hit_test_handle(self.selected_box, event.x, event.y)
            if corner:
                self.resizing = corner
                return

        hit = self.find_box(event.x, event.y)
        self.selected_box = hit

        if hit is not None:
            b = self.boxes[hit]
            cx = b["x"] * self.canvas_w
            cy = b["y"] * self.canvas_h
            self.drag_offset = (cx - event.x, cy - event.y)
            self.dragging = True
        else:
            # Leerer Bereich -> neue Box beginnt hier
            self.creating = True
            self.new_box_start = (event.x, event.y)
            self.new_box_id = None

        self.load_image()

    def on_drag(self, event):
        if self.resizing:
            self._resize_selected(event.x, event.y, self.resizing)
            self._redraw_boxes_only()
            return

        if self.dragging and self.selected_box is not None:
            dx, dy = self.drag_offset
            cx = (event.x + dx) / self.canvas_w
            cy = (event.y + dy) / self.canvas_h
            self.boxes[self.selected_box]["x"] = cx
            self.boxes[self.selected_box]["y"] = cy
            self._redraw_boxes_only()
            return

        if self.creating:
            self.canvas.delete("preview_box")
            sx, sy = self.new_box_start
            self.canvas.create_rectangle(
                sx, sy, event.x, event.y,
                outline="lime", width=2, tags="preview_box"
            )

    def on_release(self, event):
        if self.resizing:
            self.resizing = False
            self.save_labels()
            return

        if self.dragging:
            self.dragging = False
            self.save_labels()
            return

        if self.creating:
            self.creating = False
            sx, sy = self.new_box_start
            ex, ey = event.x, event.y

            if abs(ex - sx) > 8 and abs(ey - sy) > 8:
                left, right = sorted((sx, ex))
                top, bottom = sorted((sy, ey))

                self.boxes.append({
                    "cls": 0,
                    "x": (left + right) / 2 / self.canvas_w,
                    "y": (top + bottom) / 2 / self.canvas_h,
                    "w": (right - left) / self.canvas_w,
                    "h": (bottom - top) / self.canvas_h,
                })
                self.selected_box = len(self.boxes) - 1
                self.save_labels()

            self.load_image()

    def _hit_test_handle(self, box_idx, x, y):
        if box_idx is None or box_idx >= len(self.boxes):
            return None
        b = self.boxes[box_idx]
        cx = b["x"] * self.canvas_w
        cy = b["y"] * self.canvas_h
        w = b["w"] * self.canvas_w
        h = b["h"] * self.canvas_h

        corners = {
            "tl": (cx - w / 2, cy - h / 2),
            "tr": (cx + w / 2, cy - h / 2),
            "bl": (cx - w / 2, cy + h / 2),
            "br": (cx + w / 2, cy + h / 2),
        }
        for name, (hx, hy) in corners.items():
            if abs(x - hx) <= self.HANDLE_SIZE and abs(y - hy) <= self.HANDLE_SIZE:
                return name
        return None

    def _resize_selected(self, x, y, corner):
        b = self.boxes[self.selected_box]
        cx = b["x"] * self.canvas_w
        cy = b["y"] * self.canvas_h
        w = b["w"] * self.canvas_w
        h = b["h"] * self.canvas_h

        left, top = cx - w / 2, cy - h / 2
        right, bottom = cx + w / 2, cy + h / 2

        if "l" in corner:
            left = x
        if "r" in corner:
            right = x
        if "t" in corner:
            top = y
        if "b" in corner:
            bottom = y

        if right - left < 10 or bottom - top < 10:
            return

        b["x"] = (left + right) / 2 / self.canvas_w
        b["y"] = (top + bottom) / 2 / self.canvas_w if False else (top + bottom) / 2 / self.canvas_h
        b["w"] = (right - left) / self.canvas_w
        b["h"] = (bottom - top) / self.canvas_h

    # -------------------------
    # DELETE BOX
    # -------------------------

    def delete_box(self):

        if self.selected_box is None:
            return

        del self.boxes[self.selected_box]

        self.selected_box = None

        self.save_labels()
        self.load_image()

    # -------------------------
    # SAVE
    # -------------------------

    def save_labels(self):

        label_file = self.label_dir / f"{self.current_image_path.stem}.txt"

        with open(label_file, "w") as f:

            for b in self.boxes:

                f.write(f"{b['cls']} {b['x']} {b['y']} {b['w']} {b['h']}\n")

    # -------------------------
    # NAV
    # -------------------------

    def next(self):

        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.load_image()

    def prev(self):

        if self.index > 0:
            self.index -= 1
            self.load_image()

    # -------------------------
    # REVIEW
    # -------------------------

    def accept(self):
        self._mark(True)
        if self.dm:
            self._promote_to_train()
        else:
            self.next()

    def reject(self):
        self._mark(False)
        if self.dm:
            self._discard_sample()
        else:
            self.next()

    def _discard_sample(self):
        stem = self.current_image_path.stem
        label_path = self.label_dir / f"{stem}.txt"
        meta_path = self.meta_dir / f"{stem}.json"

        discard_dir = self.review_path / "discarded"
        discard_dir.mkdir(exist_ok=True)

        self.current_image_path.rename(discard_dir / self.current_image_path.name)
        if label_path.exists():
            label_path.rename(discard_dir / label_path.name)
        if meta_path.exists():
            meta_path.rename(discard_dir / meta_path.name)

        del self.image_paths[self.index]
        if self.index >= len(self.image_paths):
            self.index = max(0, len(self.image_paths) - 1)

        if self.image_paths:
            self.load_image()
        else:
            self._show_empty_state()

    def _mark(self, accepted: bool):

        meta_file = self.meta_dir / f"{self.current_image_path.stem}.json"

        if meta_file.exists():

            with open(meta_file, "r") as f:
                data = json.load(f)

            data["reviewed"] = True
            data["accepted"] = accepted

            with open(meta_file, "w") as f:
                json.dump(data, f, indent=4)
                
    def _show_empty_state(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas_w // 2, self.canvas_h // 2,
            text="Review Queue leer 🎉", fill="white", font=("", 20)
    )             
    def _promote_to_train(self):
        stem = self.current_image_path.stem
        label_src = self.label_dir / f"{stem}.txt"

        img_dst = self.dm.train_images / self.current_image_path.name
        label_dst = self.dm.train_labels / f"{stem}.txt"

        self.current_image_path.rename(img_dst)
        if label_src.exists():
            label_src.rename(label_dst)

        del self.image_paths[self.index]

        if self.index >= len(self.image_paths):
            self.index = max(0, len(self.image_paths) - 1)

        if self.image_paths:
            self.load_image()
        else:
            self._show_empty_state()