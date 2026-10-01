# Roadmap: LLM-basierter "Wertigkeitsscore" statt Audio-Lautstärke-Heuristik

Ziel: Die aktuelle Audio-Lautstärke-basierte Entscheidung ("ist dieser Moment
laut genug / groß genug, um ins Highlight zu kommen?") durch eine LLM-gestützte
Bewertung ersetzen, die YOLO-Events (Hitmarker, Kills, spielspezifische Trigger),
Cluster-Eigenschaften und optional Audio als **Kontext** nutzt, um einen
**Wertigkeitsscore** (z.B. 0–100) pro Kandidaten-Clip zu vergeben. Ein
konfigurierbarer **Mindest-Wertigkeitsscore** entscheidet, ob der Clip ins
finale Video übernommen wird.

Dieses Dokument ist die Entscheidungs- und Umsetzungsgrundlage, nicht der Code
selbst. Es zeigt: was heute schon da ist, wo der neue Schritt technisch andockt,
welche Entscheidungen du treffen musst, und einen Phasenplan bis zur
produktiven Nutzung.

---

## 1. Ist-Zustand (was heute die Entscheidung trifft)

Datei: `app/cutter/highlight_cutter_pointbased.py`

Der aktuelle Ablauf ist bereits mehrstufig:

```
Phase A  Punkte sammeln       -> Audio-Peaks (RMS-Percentile) + YOLO-Events
Phase B  Clustering            -> zeitlich nahe Punkte -> Cluster
Phase C  Fenster pro Cluster   -> Pre-/Post-Buffer, u.a. lautstärke-abhängig
Phase D  Merge                 -> überlappende Fenster verschmelzen
Stufe 2  Interne Lücken raus   -> stille Abschnitte im Clip rausschneiden
Stufe 3  Mikro-Trimming        -> Schnittkanten auf leise Stelle snappen
```

Die **Lautstärke** wirkt aktuell an drei Stellen:
1. `loud_threshold_percentile` / `loud_growth_per_sec` / `loud_growth_max_sec`
   → verlängert das Post-Buffer-Fenster, wenn im Cluster-Fenster viel "laut"
   ist (Phase C).
2. `min_cluster_confidence_enabled` / `discard_isolated_audio_points` →
   isolierte **reine Audio**-Einzelpunkte werden als Rauschen verworfen
   (Noise-Filter, Phase C).
3. `internal_silence_threshold_percentile` / `micro_trim_silence_percentile`
   → Lücken-Entfernung und Mikro-Trimming (Stufe 2/3, **bleiben unverändert**,
   das ist reines Audio-Schnitt-Feintuning, keine "ist das relevant"-Frage).

**Wichtig für die Migration:** Nur Punkt 1 und 2 sind eigentlich
"Relevanz-Entscheidungen" (lohnt sich der Moment?). Punkt 3 ist reines
Audio-Schnitthandwerk und sollte **erhalten bleiben**, unabhängig vom neuen
Scoring. Das ist die erste wichtige Trennung, die du im Kopf behalten musst:

> **Scoring entscheidet WAS reinkommt. Stufe 2/3 entscheiden WIE sauber
> geschnitten wird.** Beides bleibt getrennt.

---

## 2. Zielbild

Nach Phase D (fertig gemergte Fenster = Kandidaten-Clips) kommt ein neuer
Schritt:

```
Phase D  Merge
   |
   v
Phase E  Wertigkeits-Scoring (NEU)
   - pro finalem Fenster: strukturierte Beschreibung bauen
     (YOLO-Events im Fenster, Cluster-Größe/-Dichte, Audio-Kennzahlen,
      Spiel-Kontext/Glossar)
   - LLM (oder lokal trainierter Scorer) bewertet: Score 0-100 + Begründung
   - Fenster mit Score < min_worthiness_score werden verworfen
   |
   v
Stufe 2  Interne Lücken raus   (unverändert)
Stufe 3  Mikro-Trimming         (unverändert)
```

Die **Zeitgrenzen** (Start/Ende) der Fenster kommen weiterhin aus der
bestehenden deterministischen Logik (Audio+YOLO-Punkte, Cluster, Buffer).
Das LLM entscheidet **nur** über "rein oder raus" (und optional später über
einen Bonus/Malus auf die Post-Buffer-Länge, siehe Abschnitt 7).

### Warum nicht das LLM auch die Zeitgrenzen bestimmen lassen?

Bewusst **nicht empfohlen** für v1:
- LLMs sind schlecht darin, aus einer Text-Beschreibung präzise
  Sekunden-Zeitstempel zu "erfinden", ohne die Rohdaten (Audio-Kurve,
  Frame-für-Frame-Bild) wirklich zu sehen.
- Ein Fehler hier bedeutet abgeschnittene/falsch getrimmte Clips - viel
  riskanter als ein falsch bewerteter, aber korrekt geschnittener Clip.
- Die bestehende Timing-Logik ist bereits fein einstellbar (Buffer,
  Merge-Gap, Mikro-Trim) und funktioniert nachweislich gut.

→ **Trennung von Concern**: Timing = deterministischer Algorithmus (bleibt),
Relevanz/Wertigkeit = LLM (neu).

---

## 3. Kernentscheidungen, die DU treffen musst

### 3.1 Wo läuft das LLM?

| Option | Vorteile | Nachteile |
|---|---|---|
| **Cloud-API** (OpenAI, Anthropic, Gemini, ...) | Beste Qualität, kein lokales Setup, einfache Integration | Kosten pro Aufruf, Internet nötig, Daten verlassen den PC |
| **Lokales LLM** (Ollama, LM Studio, z.B. Llama 3.1 8B / Qwen2.5) | Kostenlos pro Aufruf, offline, volle Datenkontrolle | Braucht VRAM (läuft neben YOLO/Training!), Qualität schwankt, muss selbst gehostet werden |
| **Hybrid**: Cloud-LLM nur zum Bootstrap/Labeln, danach lokal trainierter Scorer für den Produktivbetrieb | Günstig & schnell im Dauerbetrieb, kein API-Zwang zur Laufzeit | Mehr Aufwand: eigenes Trainings-/Feature-Pipeline nötig (siehe 3.5) |

**Empfehlung:** Mit Cloud-API starten (schnell validierbar, kein VRAM-Konflikt
mit YOLO/Training auf deiner GPU), und **Hybrid als Zielbild** für den
Dauerbetrieb vormerken (siehe Abschnitt 7).

**Zu entscheiden:** Welcher Anbieter/Modell? Budget pro Monat? Ist Cloud
grundsätzlich ok (Datenschutz: es werden nur Metadaten/Zahlen geschickt, keine
Videos/Bilder - siehe 3.3)?

### 3.2 Was genau bekommt das LLM als Input?

Empfehlung: **rein textuell/strukturiert**, kein Bild/Video/Audio an das LLM
schicken (schnell, günstig, deterministisch genug). Beispiel-Payload pro
Kandidaten-Fenster:

```json
{
  "game": "THE FINALS",
  "window_duration_sec": 12.4,
  "cluster_size": 4,
  "cluster_density_sec": 0.6,
  "events": [
    {"type": "hitmarker", "t": 2.1},
    {"type": "hitmarker", "t": 2.4},
    {"type": "event", "t": 3.0},
    {"type": "hitmarker", "t": 9.8}
  ],
  "audio": {
    "loud_ratio_in_window": 0.42,
    "peak_count": 6
  },
  "context_before": "3.2s Stille davor, kein Trigger in den letzten 8s",
  "context_after": "Naechster Cluster erst in 14s"
}
```

**Zu entscheiden:**
- Reicht das (Events + Audio-Kennzahlen + Spielname), oder soll später auch
  Kontext wie Sprachchat/Kommentar-Transkript (Whisper Speech-to-Text) oder
  OCR vom Scoreboard mit rein? → **nicht für v1**, aber als Erweiterungspunkt
  im Schema vormerken (siehe Abschnitt 8).
- Wie viel "Nachbarschafts-Kontext" (vorherige/nächste Cluster) soll das LLM
  sehen, um z.B. "das war Teil einer laengeren Serie" zu erkennen?

### 3.3 Datenschutz / Sicherheit

- Es werden **keine Bilder, kein Ton, keine Dateipfade/Dateinamen** an ein
  Cloud-LLM geschickt - nur abstrakte Zahlen/Klassennamen. Vor dem Versand
  Dateinamen/Pfade explizit rausfiltern (auch aus Log-Ausgaben, die versehentlich
  mitgeschickt werden könnten).
- API-Keys **nur** über Umgebungsvariable oder lokale, von Git ausgeschlossene
  Config-Datei - niemals im Code oder in `games.json` einchecken.
- Rate-Limits/Kosten-Obergrenze einbauen (z.B. max. X Aufrufe/Video), damit ein
  Bug nicht versehentlich ein großes API-Kontingent verbrennt.

### 3.4 Score-Skala & Schwellenwert

**Zu entscheiden:**
- Skala: `0-100` (empfohlen, intuitiv für einen Slider in den Einstellungen)
  vs. `0.0-1.0` vs. Kategorien ("niedrig/mittel/hoch").
- Schwelle **absolut** (z.B. "alles unter 55 raus", einfach, vorhersehbar)
  vs. **relativ/percentile** (z.B. "die besten 60% aller Cluster in diesem
  Video", passt sich an die "Energie" des jeweiligen Videos an, aber weniger
  vorhersehbar). **Empfehlung: absolut für v1**, percentile als spätere
  Option (`scoring_mode: "absolute" | "percentile"`).
- Soll der Score zusätzlich (wie frueher die Lautstaerke) die
  Post-Buffer-Laenge beeinflussen ("Bonus-Sekunden bei sehr hohem Score"),
  oder **nur** ein reiner Ja/Nein-Filter sein? **Empfehlung:** in v1 nur
  Filter (einfacher, robuster), Bonus-Buffer erst in v2 wenn das Scoring sich
  bewaehrt hat.

### 3.5 Bootstrap vs. Dauerbetrieb (wichtig für "produktiv einsetzbar")

Das Projekt hat bereits eine Active-Learning-Infrastruktur für YOLO
(`DatasetManager`, `AutoAnnotator`, `ReviewWindow`/Web-Review-Tool, `Trainer`).
Dasselbe Muster funktioniert für das Scoring:

1. **Bootstrap-Phase**: Cloud-LLM bewertet jedes Fenster live waehrend der
   Verarbeitung, Ergebnis (Score + Begründung) wird zusammen mit dem Clip in
   der Review-Oberflaeche angezeigt. Du bestaetigst/korrigierst den Score
   manuell (wie heute Accept/Reject bei YOLO-Labels).
2. **Trainings-Datensatz** entsteht daraus: Feature-Vektor (Events, Cluster-Infos,
   Audio-Kennzahlen, Spiel) -> von dir bestaetigter/korrigierter Score.
3. **Lokaler Scorer** (z.B. Gradient Boosting / kleines MLP auf den
   strukturierten Features, **kein** Sprachmodell noetig) wird periodisch
   nachtrainiert - genau wie `Trainer.train()` heute das YOLO-Modell
   nachtrainiert.
4. **Produktivbetrieb**: lokaler Scorer läuft ohne API-Kosten/Latenz/Internet.
   Cloud-LLM bleibt im Hintergrund verfuegbar für neue Spiele/Edge-Cases
   (Cold-Start fuer ein neues Spiel, wo noch keine Trainingsdaten existieren).

**Zu entscheiden:** Willst du gleich in Schritt 4 investieren, oder erstmal
dauerhaft mit Cloud-LLM im Hot-Path leben (einfacher, aber laufende Kosten +
Internetabhaengigkeit pro Videoverarbeitung)? Für "professionell und
produktiv einsetzbar" ist **Schritt 4 langfristig klar zu empfehlen**, aber
Schritt 1-3 liefern schon einen nutzbaren Zwischenstand.

### 3.6 Pro-Spiel-Kontext (Glossar)

Jedes Spiel hat unterschiedliche YOLO-Klassen (`config/games.json` ->
`classes`). Das LLM braucht pro Spiel eine kurze Erklaerung, was diese
Klassen bedeuten und wie wichtig sie sind (z.B. "hitmarker" bei The Finals ist
haeufig aber schwach, "event" ist selten aber stark; bei Battlefield 6 ist
"Kill/Assist" das einzige Signal).

**Zu entscheiden:** Erweiterung von `games.json` um ein Feld wie:
```json
"llm_context": {
  "event": "Seltenes, starkes Signal - meist ein besonderer Kill oder Teamwipe.",
  "hitmarker": "Haeufiges, schwaches Signal - normaler Treffer im Kampf."
}
```
Diese Texte musst DU pro Spiel formulieren (Fachwissen ueber das Spiel noetig)
- das kann kein Modell fuer dich erraten.

### 3.7 Fehlerverhalten / Fallback

**Zu entscheiden, aber Empfehlung ist klar:** Wenn die LLM-Anfrage fehlschlaegt
(Timeout, Rate-Limit, ungueltiges JSON), **nicht** den ganzen Lauf abbrechen,
sondern:
- Fenster mit einem neutralen Default-Score behandeln (z.B. "alte Heuristik
  als Fallback fuer dieses eine Fenster"), UND
- deutlich im Log/Timeline-Report markieren, dass hier ein Fallback griff.

---

## 4. Wo im Code das andockt (technischer Fahrplan)

Kein Code in diesem Dokument - nur die Ansatzpunkte, damit die Umsetzung
zielgerichtet ist:

1. **`app/cutter/highlight_cutter_pointbased.py`**
   - Neue Funktion `score_windows(merged_windows, rms, times, yolo_event_times, cfg, game_context)`
     zwischen Phase D und Stufe 2 einhaengen (in `process_video`).
   - `Config` um Felder erweitern: `enable_llm_scoring`, `min_worthiness_score`,
     `llm_provider`, `llm_model`, `llm_batch_size`, `scoring_mode`.
   - Die bisherigen Noise-Filter-Felder (`discard_isolated_audio_points`,
     `min_cluster_confidence_enabled`) bleiben als **Fallback/Legacy-Modus**
     erhalten (Abwaertskompatibilitaet, siehe 6).

2. **Neues Modul, z.B. `app/scoring/worthiness_scorer.py`**
   - Baut die strukturierte Payload pro Fenster.
   - Kapselt den LLM-Client (austauschbar: Cloud-API-Client vs. lokaler
     Scorer) hinter einem gemeinsamen Interface, z.B.
     `score(window_features: dict) -> {"score": float, "reason": str}`.
   - Batching mehrerer Fenster in einen Request (Kosten/Latenz sparen).
   - Caching: Ergebnis pro Video/Fenster in einer Datei neben der Ausgabe
     speichern (wie schon `_metadata.json`), damit ein erneuter Lauf nicht
     nochmal bezahlt/gewartet werden muss.

3. **`config/games.py` / `config/games.json`**
   - Neues Feld `llm_context` (Klassen-Glossar) pro Spiel, analog zu `classes`.
   - Validierung dafuer in `_validate_game_entry` ergaenzen.

4. **`app/webui/server.py` + `frontend/`**
   - Neue Settings-Schema-Eintraege: Scoring an/aus, Mindest-Score-Slider,
     Anbieter/Modell-Auswahl.
   - Timeline/Log-Ausgabe (`_final_timeline.txt`, `intro_metadata`) um
     `worthiness_score` + `reason` pro Clip erweitern - **wichtig fuer
     Nachvollziehbarkeit** (professionelle Anforderung: du willst sehen,
     *warum* ein Clip drin oder draussen ist).
   - Review-Tool optional erweitern: Score + Begruendung neben dem Bild
     anzeigen, mit der Moeglichkeit den Score manuell zu korrigieren (das ist
     die Datenquelle fuer Schritt 3.5).

5. **Neuer, optionaler Trainings-Baustein** (erst wenn 3.5/Schritt 3-4 ansteht)
   - Analog zu `app/training/trainer.py`, aber fuer den lokalen Scorer statt
     YOLO. Eigenes leichtgewichtiges Modell (kein Deep Learning noetig -
     Gradient Boosting reicht fuer strukturierte Tabellen-Features meist aus).

---

## 5. Phasenplan (Meilensteine)

| Phase | Inhalt | Ergebnis |
|---|---|---|
| **M0** | Entscheidungen aus Abschnitt 3 treffen (Anbieter, Skala, Schwelle, Payload-Schema, Spiel-Glossar-Texte schreiben) | Klarer Rahmen, keine Code-Aenderung noetig |
| **M1** | `WorthinessScorer`-Modul bauen (Payload-Bau + LLM-Client + JSON-Schema-Validierung + Retry/Fallback) | Isoliert testbar, noch nicht in die Pipeline eingehaengt |
| **M2** | In `process_video` einhaengen, `Config`-Felder + Web-UI-Settings ergaenzen | Scoring laeuft, filtert Fenster, aber noch ohne Review-Integration |
| **M3** | Score + Begruendung in Timeline/Metadata-Dateien ausgeben | Nachvollziehbarkeit/Transparenz gegeben |
| **M4** | Shadow-Mode: Scoring **parallel** zur alten Heuristik laufen lassen, beide Ergebnisse loggen/vergleichen, OHNE dass Shadow-Ergebnis den Schnitt beeinflusst | Belastbare Datenbasis, um Schwellenwert realistisch zu kalibrieren, bevor "scharf" geschaltet wird |
| **M5** | Umschalten: Scoring bestimmt tatsaechlich, was reinkommt (alte Heuristik nur noch als Fallback bei Fehlern) | Produktiv nutzbar |
| **M6** | Review-Tool: Score anzeigen + manuell korrigierbar machen, Korrekturen persistieren | Datengrundlage fuer M7 |
| **M7** (optional, langfristig) | Lokalen Scorer aus gesammelten Korrekturen trainieren, LLM nur noch als Fallback/Cold-Start fuer neue Spiele | Kein laufender API-Zwang mehr im Alltag |

---

## 6. Migrations-/Kompatibilitaetsstrategie

- Neuer `Config`-Schalter `scoring_mode: "legacy" | "llm" | "hybrid"`
  (Default zunaechst `"legacy"`, damit bestehende Videos/Workflows nicht
  ueberrascht werden).
- Alte Felder (`loud_threshold_percentile` etc.) bleiben bestehen und wirken
  weiterhin auf die **Fenstergroesse** (Timing), unabhaengig vom
  `scoring_mode` - sie sind ja fuer etwas anderes zustaendig (siehe Abschnitt 1/2).
- Reines Rollback: `scoring_mode = "legacy"` schaltet auf 100% altes Verhalten
  zurueck, ohne Codeaenderung.

---

## 7. Erweiterungsideen fuer spaeter (bewusst nicht v1)

- **Bonus-Buffer durch hohen Score** (analog zur heutigen Lautstaerke-Logik):
  sehr hoch bewertete Cluster bekommen mehr Post-Buffer. Erst sinnvoll, wenn
  Score zuverlaessig kalibriert ist (nach M4/M5).
- **Multimodaler Kontext**: Whisper-Transkript von Sprachchat/Kommentar als
  zusaetzliches Signal ("das klang nach einer Reaktion/Ueberraschung").
- **Scoreboard-/HUD-OCR**: zusaetzliche Spielzustands-Infos (z.B. Rundenstand,
  verbleibende Zeit) als Kontextsignal - kann "clutch"-Momente erkennbar machen.
- **Percentile-/relative Schwelle** pro Video statt absoluter Schwelle.
- **Multi-Spiel-Glossar-Editor** direkt im "Spiel hinzufuegen"-Formular der
  Web-UI (statt manuellem JSON-Edit).

---

## 8. Offene Fragen, die ich von dir brauche, bevor Code entsteht

1. Cloud-API oder lokales LLM? Falls Cloud: welcher Anbieter (Budget-Rahmen)?
2. Score-Skala: 0-100 ok? Absolute Schwelle ok, oder direkt percentile?
3. Soll Scoring in v1 **nur filtern** oder auch die Puffer-Laenge beeinflussen?
4. Willst du Shadow-Mode (M4) wirklich vor dem Scharfschalten, oder direkt
   "scharf" testen (schneller, aber riskanter fuer die ersten Videos)?
5. Pro Spiel: kannst du mir (oder dem LLM-Setup) kurze Texte liefern, was
   jede YOLO-Klasse bedeutet und wie wichtig sie ungefaehr ist?
6. Zielst du von Anfang an auf M7 (lokaler Scorer, kein Dauer-API-Zwang), oder
   reicht dir vorerst Cloud-LLM im Dauerbetrieb?

Sobald diese Punkte geklaert sind, kann M1 (das eigentliche Scoring-Modul)
direkt losgehen.
