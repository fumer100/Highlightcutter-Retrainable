from pathlib import Path

from app.dataset.dataset_manager import DatasetManager
from app.extractor.frame_extractor import FrameExtractor
from app.pseudo_label.auto_annotator import AutoAnnotator

CONFIDENCE_BUCKETS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


def group_events(times: list[float], gap_sec: float) -> list[list[float]]:
    """Fasst aufeinanderfolgende Event-Zeitstempel (Abstand <= gap_sec) zu einem Event zusammen."""
    groups: list[list[float]] = []
    for t in sorted(set(times)):
        if groups and t - groups[-1][-1] <= gap_sec:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def pick_group_timestamps(group: list[float], frames_per_event: int) -> list[float]:
    """Waehlt gleichmaessig verteilte Zeitstempel einer Event-Gruppe (1 = mittlerer)."""
    n = max(1, min(frames_per_event, len(group)))
    if n == 1:
        return [group[len(group) // 2]]
    idx = sorted({round(i * (len(group) - 1) / (n - 1)) for i in range(n)})
    return [group[i] for i in idx]


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.2f}-{min(hi, 1.0):.2f}"


AUTO_OFF = 1.01  # Konfidenz kann nie > 1 sein -> nichts wird automatisch ins Training uebernommen
COLLECT_MIN_CONFIDENCE = 0.10  # Daten-sammeln: auch sehr unsichere Vorschlaege des Modells als Vorlabel zeigen
COLLECT_EVENT_SHARE = 0.5  # Anteil der Sammel-Frames, der von Modell-Events stammt (Rest: neue Frames)
_UNIFORM_OFFSETS = (0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875)


def video_duration(video_path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frames / fps if fps > 0 else 0.0


def split_quota(weights: list[float], total: int) -> list[int]:
    """Verteilt `total` ganzzahlig proportional auf `weights` (Summe ist exakt `total`)."""
    s = sum(weights)
    if s <= 0 or total <= 0:
        return [0] * len(weights)
    raw = [total * w / s for w in weights]
    base = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[: total - sum(base)]:
        base[i] += 1
    return base


def thin_evenly(items: list[float], limit: int) -> list[float]:
    """Reduziert eine sortierte Liste gleichmaessig auf hoechstens `limit` Eintraege."""
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def uniform_candidates(duration: float, count: int, avoid: list[float], min_gap: float = 1.0) -> list[float]:
    """
    Gleichmaessig verteilte Reserve-Zeitstempel in Durchgaengen (grob -> fein), damit nach dem
    Aussortieren von Duplikaten trotzdem genug unterschiedliche Frames uebrig bleiben.
    """
    if duration <= 0 or count <= 0:
        return []
    out = []
    for offset in _UNIFORM_OFFSETS:
        for k in range(count):
            t = duration * (k + offset) / count
            if all(abs(t - a) >= min_gap for a in avoid):
                out.append(t)
    return out


class PipelineController:

    def __init__(self, game_name: str, model_path: str):
        self.dm = DatasetManager(game_name)
        self.extractor = FrameExtractor(self.dm)
        self.annotator = AutoAnnotator(self.dm, model_path=model_path)

    def run_full_cycle(
        self,
        video_path: str,
        events: list[float],
        model_version: str = "v1",
        event_group_gap_sec: float = 0.5,
        frames_per_event: int = 1,
        duplicate_threshold: float = 3.0,
        auto_threshold: float = 0.85,
        review_min_confidence: float = 0.25,
        max_review_per_video: int = 30,
        keep_empty: bool = False,
        fill_to: int = 0,
        progress_callback=None,
        status_callback=None,
        cancel_event=None,
    ) -> dict:
        video_path = Path(video_path)
        name = video_path.name

        def step(n: int, text: str):
            print(f"[ML {n}/4] {text}")
            if status_callback is not None:
                status_callback(f"ML-Pipeline {n}/4: {text} ({name})")

        def check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Vorgang abgebrochen.")

        stats = {
            "video": name,
            "events": len(events),
            "groups": 0,
            "duplicates": 0,
            "candidates": 0,
            "auto_train": 0,
            "auto_val": 0,
            "review": 0,
            "review_capped": 0,
            "review_empty": 0,
            "filled": 0,
            "discarded": 0,
            "conf_buckets": {bucket_label(lo, hi): 0 for lo, hi in CONFIDENCE_BUCKETS},
        }

        if not events and not fill_to:
            print(f"[ML] Keine Events fuer {name}, ueberspringe Extraktion.")
            return self._finalize(stats)

        # 1/4: Events zusammenfassen
        check_cancel()
        groups = group_events(events, event_group_gap_sec)
        timestamps = [t for g in groups for t in pick_group_timestamps(g, frames_per_event)]
        stats["groups"] = len(groups)

        # Fruehes Modell findet kaum Events: mit gleichmaessig verteilten Frames auffuellen
        if fill_to > len(timestamps):
            extra = self._uniform_timestamps(str(video_path), fill_to - len(timestamps), timestamps)
            stats["filled"] = len(extra)
            timestamps += extra
        step(1, f"{len(events)} Erkennungen -> {len(groups)} Events, {stats['filled']} gleichmaessig ergaenzt, "
                f"{len(timestamps)} Kandidaten-Frames")

        # 2/4: Frames extrahieren (ohne Duplikate)
        step(2, "Frames extrahieren")
        sample_ids, duplicates = self.extractor.extract_frames(
            video_path=str(video_path),
            timestamps=timestamps,
            model_version=model_version,
            duplicate_threshold=duplicate_threshold,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        stats["duplicates"] = duplicates
        stats["candidates"] = len(sample_ids)
        print(f"      {len(sample_ids)} Frames gespeichert, {duplicates} Duplikate uebersprungen")

        # 3/4: Vorlabeln und sortieren
        step(3, "Vorlabeln (" + (f"Auto-Training ab {auto_threshold:.2f}, " if auto_threshold <= 1.0 else "Auto-Training aus, ")
                + f"Review ab {review_min_confidence:.2f}, "
                + ("leere Frames kommen ungelabelt zur Review)" if keep_empty else "darunter verwerfen)"))
        review_samples = []
        for i, sample_id in enumerate(sample_ids, start=1):
            check_cancel()
            result = self.annotator.annotate_sample(
                sample_id=sample_id,
                model_version=model_version,
                auto_threshold=auto_threshold,
                min_confidence=review_min_confidence,
                keep_empty=keep_empty,
            )
            if result is not None:
                outcome = result["outcome"]
                if outcome == "discarded":
                    stats["discarded"] += 1
                elif outcome == "review" and result["label_count"] == 0:
                    stats["review_empty"] += 1
                    review_samples.append(result)
                else:
                    self._count_bucket(stats, result["confidence"])
                    if outcome == "auto":
                        stats["auto_val" if result["split"] == "val" else "auto_train"] += 1
                    else:
                        review_samples.append(result)
            if progress_callback is not None:
                progress_callback("Vorlabeln", i / len(sample_ids) * 100)

        # 4/4: Review-Queue pro Video begrenzen (unsicherste zuerst behalten)
        step(4, "Review-Queue begrenzen")
        if max_review_per_video > 0 and len(review_samples) > max_review_per_video:
            mid = (review_min_confidence + auto_threshold) / 2
            review_samples.sort(key=lambda r: abs(r["confidence"] - mid))
            for r in review_samples[max_review_per_video:]:
                self._delete_sample(r["sample_id"])
                stats["review_capped"] += 1
                if r["label_count"] == 0:
                    stats["review_empty"] -= 1
            review_samples = review_samples[:max_review_per_video]
        stats["review"] = len(review_samples)

        return self._finalize(stats)

    def run_collect_cycle(
        self,
        video_path: str,
        events: list[float],
        quota: int,
        model_version: str = "collect",
        event_group_gap_sec: float = 0.5,
        frames_per_event: int = 1,
        duplicate_threshold: float = 3.0,
        min_confidence: float = COLLECT_MIN_CONFIDENCE,
        progress_callback=None,
        status_callback=None,
        cancel_event=None,
    ) -> dict:
        """
        Daten-sammeln-Modus: legt ca. `quota` Frames des Videos in die Review Queue. Die Haelfte stammt
        von Modell-Events, der Rest sind neue, gleichmaessig verteilte Frames. Nichts wird automatisch
        ins Training uebernommen und nichts verworfen: Das Modell schlaegt Boxen vor (auch unsichere),
        wo es nichts findet, bleibt das Bild ungelabelt. So lassen sich auch Fehler eines schwachen
        Modells korrigieren.
        """
        video_path = Path(video_path)
        name = video_path.name

        def step(n: int, text: str):
            print(f"[Sammeln {n}/3] {text}")
            if status_callback is not None:
                status_callback(f"Daten sammeln {n}/3: {text} ({name})")

        def check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Vorgang abgebrochen.")

        stats = {
            "video": name, "events": len(events), "groups": 0, "duplicates": 0, "candidates": 0,
            "auto_train": 0, "auto_val": 0, "review": 0, "review_capped": 0, "review_empty": 0,
            "filled": 0, "discarded": 0, "prelabeled": 0, "collect_events": 0, "collect_new": 0,
            "collect_quota": quota,
            "conf_buckets": {bucket_label(lo, hi): 0 for lo, hi in CONFIDENCE_BUCKETS},
        }
        if quota <= 0:
            return self._finalize(stats)

        # 1/3: Kandidaten waehlen (Event-Frames zuerst, danach neue Frames als Reserve-Liste)
        check_cancel()
        groups = group_events(events, event_group_gap_sec)
        stats["groups"] = len(groups)
        event_ts = sorted(t for g in groups for t in pick_group_timestamps(g, frames_per_event))
        event_ts = thin_evenly(event_ts, int(quota * COLLECT_EVENT_SHARE))
        new_ts = uniform_candidates(video_duration(str(video_path)), quota, event_ts)
        step(1, f"{len(events)} Erkennungen -> {len(groups)} Events; Ziel {quota} Frames "
                f"(bis {len(event_ts)} von Events, Rest neue Frames)")

        # 2/3: extrahieren (hoert beim Ziel auf, ueberspringt Duplikate)
        step(2, "Frames extrahieren")
        kept_ts: list[float] = []
        sample_ids, duplicates = self.extractor.extract_frames(
            video_path=str(video_path),
            timestamps=event_ts + new_ts,
            model_version=model_version,
            duplicate_threshold=duplicate_threshold,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            max_keep=quota,
            keep_order=True,
            kept_timestamps=kept_ts,
        )
        event_set = set(event_ts)
        stats["collect_events"] = sum(1 for t in kept_ts if t in event_set)
        stats["collect_new"] = len(kept_ts) - stats["collect_events"]
        stats["duplicates"] = duplicates
        stats["candidates"] = len(sample_ids)
        print(f"      {len(sample_ids)} Frames gespeichert ({stats['collect_events']} von Events, "
              f"{stats['collect_new']} neu), {duplicates} Duplikate uebersprungen")

        # 3/3: Vorlabeln - alles bleibt in der Review Queue
        step(3, f"Vorlabeln (Vorschlaege ab {min_confidence:.2f}, nichts wird automatisch uebernommen)")
        for i, sample_id in enumerate(sample_ids, start=1):
            check_cancel()
            result = self.annotator.annotate_sample(
                sample_id=sample_id,
                model_version=model_version,
                auto_threshold=AUTO_OFF,
                min_confidence=min_confidence,
                keep_empty=True,
            )
            if result is not None:
                stats["review"] += 1
                if result["label_count"] == 0:
                    stats["review_empty"] += 1
                else:
                    stats["prelabeled"] += 1
                    self._count_bucket(stats, result["confidence"])
            if progress_callback is not None:
                progress_callback("Vorlabeln", i / len(sample_ids) * 100)

        return self._finalize(stats)

    @staticmethod
    def _uniform_timestamps(video_path: str, count: int, existing: list[float], min_gap: float = 1.0) -> list[float]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        duration = frames / fps if fps > 0 else 0.0
        if duration <= 0 or count <= 0:
            return []

        slots = [duration * (k + 0.5) / count for k in range(count)]
        return [t for t in slots if all(abs(t - e) >= min_gap for e in existing)]

    def _count_bucket(self, stats: dict, conf: float):
        for lo, hi in CONFIDENCE_BUCKETS:
            if lo <= conf < hi:
                stats["conf_buckets"][bucket_label(lo, hi)] += 1
                return

    def _delete_sample(self, sample_id: str):
        for p in (
            self.dm.review_images / f"{sample_id}.jpg",
            self.dm.review_labels / f"{sample_id}.txt",
            self.dm.review_metadata / f"{sample_id}.json",
        ):
            p.unlink(missing_ok=True)

    def _finalize(self, stats: dict) -> dict:
        # Rueckwaertskompatible Schluessel
        stats["extracted"] = stats["candidates"]
        stats["annotated"] = stats["auto_train"] + stats["auto_val"] + stats["review"]
        stats["needs_review"] = stats["review"]
        return stats
