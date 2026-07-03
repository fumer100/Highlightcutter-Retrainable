import tkinter as tk
from PIL import Image, ImageTk
import cv2
import json
from pathlib import Path
import math


class ReviewWindow:

    def __init__(self, review_queue_path: str):

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

        # Zoom (optional minimal)
        self.zoom = 1.0

        self.root = tk.Tk()
        self.root.title("Active Learning Review Tool - v3")

        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack()

        btn = tk.Frame(self.root)
        btn.pack()

        tk.Button(btn, text="Accept (A)", command=self.accept).pack(side=tk.LEFT)
        tk.Button(btn, text="Reject (D)", command=self.reject).pack(side=tk.LEFT)
        tk.Button(btn, text="Delete (Del)", command=self.delete_box).pack(side=tk.LEFT)
        tk.Button(btn, text="Next", command=self.next).pack(side=tk.LEFT)
        tk.Button(btn, text="Prev", command=self.prev).pack(side=tk.LEFT)

        # Events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.root.bind("a", lambda e: self.accept())
        self.root.bind("d", lambda e: self.reject())
        self.root.bind("<Delete>", lambda e: self.delete_box())

        self.load_image()

        self.root.mainloop()

    # -------------------------
    # IMAGE
    # -------------------------

    def load_image(self):

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

            color = "red"

            if self.selected_box == i:
                color = "yellow"

            self.canvas.create_rectangle(
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2,
                outline=color,
                width=2,
                tags=f"box_{i}"
            )

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

        self.selected_box = self.find_box(event.x, event.y)

        if self.selected_box is not None:

            b = self.boxes[self.selected_box]

            cx = b["x"] * self.canvas_w
            cy = b["y"] * self.canvas_h

            self.drag_offset = (cx - event.x, cy - event.y)
            self.dragging = True

        self.load_image()

    def on_drag(self, event):

        if self.selected_box is None:
            return

        if self.dragging:

            dx, dy = self.drag_offset

            cx = (event.x + dx) / self.canvas_w
            cy = (event.y + dy) / self.canvas_h

            self.boxes[self.selected_box]["x"] = cx
            self.boxes[self.selected_box]["y"] = cy

            self.load_image()

    def on_release(self, event):

        self.dragging = False

        self.save_labels()

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
        self.next()

    def reject(self):

        self._mark(False)
        self.next()

    def _mark(self, accepted: bool):

        meta_file = self.meta_dir / f"{self.current_image_path.stem}.json"

        if meta_file.exists():

            with open(meta_file, "r") as f:
                data = json.load(f)

            data["reviewed"] = True
            data["accepted"] = accepted

            with open(meta_file, "w") as f:
                json.dump(data, f, indent=4)