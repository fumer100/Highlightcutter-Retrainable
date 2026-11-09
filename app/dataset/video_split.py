"""
Einmalige Umstellung eines bestehenden Datenstands auf eine saubere Aufteilung PRO VIDEO.

Bis jetzt wurden Bilder zufaellig auf train/val verteilt, sodass fast identische Frames desselben
Videos in beiden Mengen lagen (Datenleck: val wirkt besser als das Modell wirklich ist).
Dieses Werkzeug ordnet jedes Video komplett EINEM Split zu:
  golden - festes Vergleichs-Set (nie trainiert), mit Abdeckung auch seltener Klassen
  val    - Validierung waehrend des Trainings
  train  - der Rest

Aufruf (Standard ist ein Probelauf, der nur den Plan zeigt und nichts veraendert):
    python -m app.dataset.video_split "The Finals"
    python -m app.dataset.video_split "The Finals" --apply

Vor --apply die Web-UI/GUI schliessen (kein Training / keine Review laufen lassen).
Jede Verschiebung wird in split_migration_<Zeitstempel>.json im Dataset-Ordner protokolliert.

Hinweis: Alte automatisch erzeugte Labels lassen sich im Altbestand nicht von menschlich geprueften
unterscheiden. Das Golden Set sollte daher einmal im Dataset-Viewer stichprobenartig geprueft werden.
"""

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field

from app.dataset.dataset_manager import MIN_VIDEOS_FOR_VIDEO_SPLIT, DatasetManager

GOLDEN_RATIO = 0.12  # Ziel-Anteil der Bilder im Golden Set
VAL_RATIO = 0.10  # Ziel-Anteil der Bilder in val
GOLDEN_MAX_RATIO = 0.25  # harte Obergrenzen, damit Klassen-Abdeckung nicht das Training leert
VAL_MAX_RATIO = 0.20
CLASS_SHARE = 0.12  # je Klasse sollen mindestens so viele Boxen im Golden Set liegen (val: halb so viele)


@dataclass
class VideoInfo:
    images: list = field(default_factory=list)  # (sample_id, aktueller Split)
    classes: Counter = field(default_factory=Counter)

    @property
    def count(self) -> int:
        return len(self.images)


def collect_videos(dm: DatasetManager) -> tuple[dict[str, VideoInfo], int]:
    """Gruppiert alle Bilder in train/val/golden nach Quellvideo. Gibt (Videos, Bilder ohne bekanntes Video) zurueck."""
    videos: dict[str, VideoInfo] = {}
    unknown = 0
    for split in ("train", "val", "golden"):
        img_dir, lbl_dir = dm.split_dirs(split)
        for img in img_dir.glob("*.jpg"):
            key = dm.video_key(img.stem)
            if key is None:
                unknown += 1
                continue
            info = videos.setdefault(key, VideoInfo())
            info.images.append((img.stem, split))
            label = lbl_dir / f"{img.stem}.txt"
            if label.is_file():
                for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if parts:
                        try:
                            info.classes[int(float(parts[0]))] += 1
                        except ValueError:
                            pass
    return videos, unknown


def _class_targets(videos: dict[str, VideoInfo], pool: list[str], share: float) -> dict[int, int]:
    """Soll-Boxen pro Klasse in einer Pruefmenge. Klassen, die nur in einem Video vorkommen, sind nicht teilbar."""
    totals: Counter = Counter()
    containing: Counter = Counter()
    for key in pool:
        for cls, n in videos[key].classes.items():
            totals[cls] += n
            containing[cls] += 1
    return {
        cls: max(1, min(math.ceil(share * total), total // 2))
        for cls, total in totals.items()
        if containing[cls] >= 2
    }


def _pick_videos(videos, pool, target_images, max_images, class_targets) -> list[str]:
    """Waehlt ganze Videos: zuerst fehlende (seltene) Klassen abdecken, dann die Zielgroesse treffen."""
    need = dict(class_targets)
    left = set(pool)
    chosen: list[str] = []
    images = 0
    while left:
        fits = [k for k in left if images + videos[k].count <= max_images]
        if not fits:
            break
        gap = target_images - images

        def gain(k):
            return sum(
                min(n, need[c]) / class_targets[c] for c, n in videos[k].classes.items() if need.get(c, 0) > 0
            )

        best = max(fits, key=lambda k: (round(gain(k), 6), -abs(videos[k].count - gap) if gap > 0 else -videos[k].count))
        if gap <= 0 and gain(best) <= 0:
            break
        chosen.append(best)
        left.remove(best)
        images += videos[best].count
        for c, n in videos[best].classes.items():
            if c in need:
                need[c] = max(0, need[c] - n)
    return chosen


def plan_split(videos: dict[str, VideoInfo], golden_ratio=GOLDEN_RATIO, val_ratio=VAL_RATIO) -> dict[str, str]:
    """Video -> 'golden' | 'val' | 'train'."""
    total = sum(v.count for v in videos.values())
    pool = list(videos)

    golden = _pick_videos(videos, pool, golden_ratio * total, GOLDEN_MAX_RATIO * total,
                          _class_targets(videos, pool, CLASS_SHARE))
    pool = [k for k in pool if k not in golden]
    val = _pick_videos(videos, pool, val_ratio * total, VAL_MAX_RATIO * total,
                       _class_targets(videos, pool, CLASS_SHARE / 2))

    plan = {k: "train" for k in videos}
    plan.update({k: "golden" for k in golden})
    plan.update({k: "val" for k in val})
    return plan


def summarize(dm: DatasetManager, videos: dict[str, VideoInfo], plan: dict[str, str]) -> str:
    names = dm.game.classes
    lines = []
    for split in ("train", "val", "golden"):
        keys = [k for k, s in plan.items() if s == split]
        classes: Counter = Counter()
        for k in keys:
            classes.update(videos[k].classes)
        detail = ", ".join(f"{names[c] if c < len(names) else c}: {n}" for c, n in sorted(classes.items()))
        lines.append(f"{split:<7} {len(keys):>3} Videos, {sum(videos[k].count for k in keys):>5} Bilder | {detail}")
    return "\n".join(lines)


def apply_plan(dm: DatasetManager, videos: dict[str, VideoInfo], plan: dict[str, str]) -> list[dict]:
    """Verschiebt die Bilder (samt Labels) in den geplanten Split und aktualisiert video_splits.json."""
    moves = []
    for key, target in plan.items():
        dst_img_dir, dst_lbl_dir = dm.split_dirs(target)
        for sample_id, current in videos[key].images:
            if current == target:
                continue
            src_img_dir, src_lbl_dir = dm.split_dirs(current)
            stem = dm._free_stem(sample_id, [(dst_img_dir, ".jpg"), (dst_lbl_dir, ".txt")])
            (src_img_dir / f"{sample_id}.jpg").rename(dst_img_dir / f"{stem}.jpg")
            label = src_lbl_dir / f"{sample_id}.txt"
            if label.exists():
                label.rename(dst_lbl_dir / f"{stem}.txt")
            moves.append({"sample_id": sample_id, "new_id": stem, "from": current, "to": target})

    splits = dm.load_video_splits()
    splits.update(plan)
    dm.save_video_splits(splits)

    log = dm.dataset / f"split_migration_{time.strftime('%Y%m%d_%H%M%S')}.json"
    log.write_text(json.dumps(moves, indent=2, ensure_ascii=False), encoding="utf-8")
    if any(m["to"] == "golden" or m["from"] == "golden" for m in moves):
        # Das bisherige Modell wurde mit Bildern trainiert, die jetzt im Golden Set liegen:
        # Das naechste Training wird deshalb ohne Vergleich uebernommen und setzt die Baseline neu.
        dm.baseline_reset_flag.write_text("golden set changed by video_split migration\n", encoding="utf-8")
    return moves


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("game", help="Spielname wie in config/games.json, z.B. 'The Finals'")
    parser.add_argument("--apply", action="store_true", help="Plan tatsaechlich ausfuehren (sonst nur anzeigen)")
    parser.add_argument("--golden-ratio", type=float, default=GOLDEN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args(argv)

    dm = DatasetManager(args.game)
    videos, unknown = collect_videos(dm)
    if len(videos) < MIN_VIDEOS_FOR_VIDEO_SPLIT:
        print(f"Nur {len(videos)} Videos bekannt (mindestens {MIN_VIDEOS_FOR_VIDEO_SPLIT} noetig) - "
              "keine saubere Aufteilung moeglich, nichts geaendert.")
        return 1

    plan = plan_split(videos, args.golden_ratio, args.val_ratio)
    print(f"{len(videos)} Videos, {sum(v.count for v in videos.values())} Bilder"
          + (f" ({unknown} Bilder ohne bekanntes Video bleiben unveraendert)" if unknown else ""))
    print("Geplante Aufteilung:")
    print(summarize(dm, videos, plan))

    moves_needed = sum(1 for k, t in plan.items() for _, cur in videos[k].images if cur != t)
    print(f"{moves_needed} Bilder wuerden verschoben.")
    if not args.apply:
        print("Probelauf - nichts veraendert. Mit --apply ausfuehren.")
        return 0

    moves = apply_plan(dm, videos, plan)
    print(f"Fertig: {len(moves)} Bilder verschoben (Protokoll: split_migration_*.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
