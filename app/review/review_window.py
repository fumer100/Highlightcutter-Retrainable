import tkinter as tk
from PIL import Image, ImageTk
import cv2
import json
from pathlib import Path


class ReviewWindow:

    """
    Minimal Label-Studio-like Review Tool
    für Bounding Box Korrektur + Dataset Cleaning
    """

    def __init__(self, review_queue_path: str):

        self.review_path = Path(review_queue_path)
        self.images = list((self.review_path / "images").glob("*.jpg"))

        self.index = 0

        self.root = tk.Tk()
        self.root.title("Active Learning Review Tool")

        self.canvas = tk.Canvas(self.root, width=900, height=500)
        self.canvas.pack()

        self.label = tk.Label(self.root, text="")
        self.label.pack()

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack()

        tk.Button(self.btn_frame, text="Accept (A)", command=self.accept).pack(side=tk.LEFT)
        tk.Button(self.btn_frame, text="Reject (D)", command=self.reject).pack(side=tk.LEFT)
        tk.Button(self.btn_frame, text="Next (→)", command=self.next_image).pack(side=tk.LEFT)
        tk.Button(self.btn_frame, text="Prev (←)", command=self.prev_image).pack(side=tk.LEFT)

        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("a", lambda e: self.accept())
        self.root.bind("d", lambda e: self.reject())

        self.current_image = None
        self.tk_image = None

        self.load_image()

        self.root.mainloop()


    def load_image(self):

        if not self.images:
            return

        img_path = self.images[self.index]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = img.resize((900, 500))

        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.label.config(text=f"{self.index+1}/{len(self.images)}")

        self.current_image = img_path


    def next_image(self):

        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_image()


    def prev_image(self):

        if self.index > 0:
            self.index -= 1
            self.load_image()


    def accept(self):

        self._mark_reviewed(accepted=True)
        self.next_image()


    def reject(self):

        self._mark_reviewed(accepted=False)
        self.next_image()


    def _mark_reviewed(self, accepted: bool):

        img_path = self.current_image

        meta_path = self.review_path / "metadata" / f"{img_path.stem}.json"

        if meta_path.exists():

            with open(meta_path, "r") as f:
                data = json.load(f)

            data["reviewed"] = True
            data["accepted"] = accepted

            with open(meta_path, "w") as f:
                json.dump(data, f, indent=4)