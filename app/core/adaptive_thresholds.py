AUTO_OFF = 1.01  # Konfidenz kann nie > 1 sein -> Auto-Training faktisch aus

# (obere Grenze der gelabelten Bilder mit Boxen, Stufenname, Auto-Training ab, Review ab,
#  Review-Limit pro Video, Frames ohne Erkennung zur Review behalten + mit gleichmaessigen Frames auffuellen)
STAGES = [
    (100, "Start", AUTO_OFF, 0.10, 50, True),
    (500, "Aufbau", 0.90, 0.20, 40, False),
    (2000, "Reif", 0.85, 0.25, 30, False),
    (float("inf"), "Stabil", 0.80, 0.30, 20, False),
]


def resolve_ml_params(cfg, adaptive: bool, labeled_images: int) -> dict:
    """
    Waehlt die ML-Schwellen. adaptive=False: feste Werte aus den Einstellungen (cfg.ml_*).
    adaptive=True: Stufe nach Anzahl gelabelter Bilder mit Boxen (train + val).

    detect_conf: Konfidenz fuer die Event-Erkennung im ML-Lauf. Ein fruehes Modell ist selten
    ueber der normalen YOLO-Schwelle sicher, sonst wuerden gar keine Events gefunden.
    fill_frames: Zielanzahl Kandidaten-Frames; fehlende werden gleichmaessig im Video ergaenzt.
    keep_empty: Frames ohne Erkennung kommen ungelabelt in die Review statt verworfen zu werden.
    """
    if not adaptive:
        return {
            "adaptive": False,
            "stage": "Manuell",
            "auto_threshold": cfg.ml_auto_train_threshold,
            "review_min": cfg.ml_review_min_confidence,
            "max_review": cfg.ml_max_review_per_video,
            "detect_conf": cfg.yolo_confidence,
            "fill_frames": 0,
            "keep_empty": False,
            "labeled_images": labeled_images,
        }

    for upper, name, auto, review_min, max_review, early in STAGES:
        if labeled_images < upper:
            return {
                "adaptive": True,
                "stage": name,
                "auto_threshold": auto,
                "review_min": review_min,
                "max_review": max_review,
                "detect_conf": min(cfg.yolo_confidence, review_min),
                "fill_frames": max_review if early else 0,
                "keep_empty": early,
                "labeled_images": labeled_images,
            }
    raise AssertionError("unreachable")
