"use strict";

// ------------------------------------------------------------
// Globaler State
// ------------------------------------------------------------

const state = {
  games: [],
  currentGame: null,
  classes: [],
  videos: [],
  reviewImages: [],
  reviewIndex: 0,
  reviewBoxes: [],
  selectedBox: null,
  selectedClass: 0,
};

const PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"];
const HANDLE_SIZE = 6;

function classColor(cls) {
  return PALETTE[cls % PALETTE.length];
}

// ------------------------------------------------------------
// Kleiner Fetch-Helfer
// ------------------------------------------------------------

async function apiGet(path) {
  const res = await fetch(path);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return res.json();
}

// ------------------------------------------------------------
// Init
// ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", init);

async function init() {
  const s = await apiGet("/api/state");
  applyState(s);
  bindEvents();
  connectEvents();
  if (!s.games.length) {
    showToast("Keine Spiele konfiguriert. Bitte ein Spiel hinzufügen.", "warn");
  }
}

function applyState(s) {
  state.games = s.games;
  state.currentGame = s.currentGame;
  state.classes = s.classes;
  state.hasModel = s.hasModel !== false;
  state.datasetStats = s.datasetStats || {};
  state.firstModelMinLabeled = s.firstModelMinLabeled || 20;
  renderGameDropdown();
  renderSettings(s.settingsSchema);
  updateReviewBadge(s.reviewCount);
  renderBootstrapBanner();
  updateActionButtons();
}

async function refreshState() {
  applyState(await apiGet("/api/state"));
}

function renderBootstrapBanner() {
  const banner = document.getElementById("bootstrapBanner");
  const needsModel = state.currentGame && !state.hasModel;
  banner.classList.toggle("hidden", !needsModel);
  document.getElementById("trainBtn").textContent =
    needsModel ? "🧠 Erstes Modell trainieren" : "🧠 Train Model";
  if (!needsModel) return;

  const st = state.datasetStats;
  const open = st.review ?? 0;
  const labeled = st.with_boxes ?? 0;
  const min = state.firstModelMinLabeled;
  const step1 = document.getElementById("bootstrapStep1");
  const step2 = document.getElementById("bootstrapStep2");
  step1.textContent = `Frames labeln: Review Queue öffnen, Boxen zeichnen und mit „Accept“ bestätigen ` +
    `(offen: ${open}, bestätigt mit Boxen: ${labeled} / mind. ${min}).`;
  step1.classList.toggle("done", labeled >= min);
  step2.textContent = `Erstes Modell trainieren: Button „Erstes Modell trainieren“ (dauert je nach GPU einige Minuten).`;
}

function updateActionButtons() {
  const needsModel = state.currentGame && !state.hasModel;
  const modelButtons = ["mlPipelineFcpxmlBtn", "exportFcpxmlBtn", "testModelBtn"];
  modelButtons.forEach((id) => {
    document.getElementById(id).disabled = state.busy || needsModel;
  });
  document.getElementById("trainBtn").disabled = !!state.busy;
  document.getElementById("cancelBtn").disabled = !state.busy;
}

// ------------------------------------------------------------
// Server-Sent-Events: Fortschritt / Log / Status vom Backend
// ------------------------------------------------------------

function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (e) => {
    const { event, data } = JSON.parse(e.data);
    switch (event) {
      case "log": onLog(data); break;
      case "progress": onProgress(data.phase, data.percent); break;
      case "overallProgress": onOverallProgress(data.idx, data.total); break;
      case "status": onStatus(data); break;
      case "busy": onBusy(data); break;
      case "done": onDone(data); break;
      case "error": onError(data); break;
      case "cancelled": onCancelled(data); break;
      case "reviewCount": onReviewCount(data); break;
      case "mlReport": onMlReport(data); break;
      case "stateChanged": refreshState(); break;
      case "modelTestResult": onModelTestResult(data); break;
      case "firstModelReady": onFirstModelReady(data); break;
    }
  };
  es.onerror = () => {
    // Browser versucht automatisch neu zu verbinden.
  };
}

// ------------------------------------------------------------
// Event-Wiring
// ------------------------------------------------------------

function bindEvents() {
  document.getElementById("gameDropdown").addEventListener("change", async (e) => {
    const res = await apiPost("/api/select_game", { name: e.target.value });
    if (res.ok) applyState(res.state);
    else showToast(res.error, "error");
  });

  document.getElementById("refreshGamesBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/refresh_games");
    if (res.ok) {
      applyState(res.state);
      showToast(`${res.state.games.length} Spiele geladen.`, "success");
    } else {
      showToast(res.error, "error");
    }
  });

  document.getElementById("addGameBtn").addEventListener("click", openAddGameModal);
  document.getElementById("editGameBtn").addEventListener("click", openEditGameModal);
  document.getElementById("cancelEditGameBtn").addEventListener("click", closeEditGameModal);
  document.getElementById("saveEditGameBtn").addEventListener("click", saveEditGame);
  document.getElementById("deleteGameBtn").addEventListener("click", openDeleteGameModal);
  document.getElementById("cancelDeleteGameBtn").addEventListener("click", closeDeleteGameModal);
  document.getElementById("deleteGameConfirmInput").addEventListener("input", (e) => {
    document.getElementById("confirmDeleteGameBtn").disabled = e.target.value.trim() !== state.deleteTarget;
  });
  document.getElementById("confirmDeleteGameBtn").addEventListener("click", confirmDeleteGame);
  document.getElementById("cancelGameBtn").addEventListener("click", closeAddGameModal);

  document.getElementById("browseModelBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/pick_files", { kind: "model" });
    if (res.ok && res.paths.length) document.getElementById("newGameModelPath").value = res.paths[0];
  });

  document.getElementById("browseGameVideosBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/pick_files", { kind: "video" });
    if (res.ok && res.paths.length) {
      state.newGameVideos = res.paths;
      document.getElementById("newGameVideoPaths").value = res.paths.join("; ");
    }
  });

  document.querySelectorAll('input[name="gameSource"]').forEach((r) =>
    r.addEventListener("change", updateGameSourceRows));

  document.getElementById("firstModelOkBtn").addEventListener("click", () => {
    document.getElementById("firstModelModal").classList.add("hidden");
  });

  document.getElementById("loadClassesBtn").addEventListener("click", async () => {
    const picked = await apiPost("/api/pick_files", { kind: "json" });
    if (!picked.ok || !picked.paths.length) return;
    const res = await apiPost("/api/extract_classes_from_json", { path: picked.paths[0] });
    if (res.ok) document.getElementById("newGameClasses").value = res.classes.join(", ");
    else showToast(res.error, "error");
  });

  document.getElementById("newGameName").addEventListener("input", updateAddGameInfo);

  document.getElementById("saveGameBtn").addEventListener("click", async () => {
    const name = document.getElementById("newGameName").value.trim();
    const source = document.querySelector('input[name="gameSource"]:checked').value;
    const modelPath = source === "model" ? document.getElementById("newGameModelPath").value.trim() : "";
    const videoPaths = source === "video" ? state.newGameVideos : [];
    const classes = document.getElementById("newGameClasses").value
      .split(",").map((c) => c.trim()).filter(Boolean);

    if (source === "model" && !modelPath) { showToast("Bitte ein YOLO-Modell (.pt) auswählen.", "warn"); return; }
    if (source === "video" && !videoPaths.length) { showToast("Bitte mindestens ein Video auswählen.", "warn"); return; }

    const res = await apiPost("/api/add_game", { name, modelPath, videoPaths, classes });
    if (res.ok) {
      applyState(res.state);
      closeAddGameModal();
      showToast(
        res.bootstrap
          ? `Spiel '${name}' angelegt. Start-Frames werden erzeugt ...`
          : `Spiel '${name}' wurde hinzugefügt.`,
        "success");
    } else {
      showToast(res.error, "error");
    }
  });

  document.getElementById("selectVideosBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/pick_files", { kind: "video" });
    if (res.ok) res.paths.forEach(addVideo);
  });

  document.getElementById("clearVideosBtn").addEventListener("click", () => {
    state.videos = [];
    renderVideoList();
  });

  document.getElementById("settingsToggle").addEventListener("click", () => {
    document.getElementById("settingsBody").classList.toggle("collapsed");
    document.querySelector("#settingsToggle .chevron").classList.toggle("collapsed");
  });

  document.getElementById("applySettingsBtn").addEventListener("click", async () => {
    const values = collectSettingsValues();
    const res = await apiPost("/api/apply_settings", values);
    if (res.ok) showToast("Einstellungen übernommen.", "success");
    else showToast("Ungültige Felder: " + res.errors.join(", "), "error");
  });

  document.getElementById("resetSettingsBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/reset_settings");
    renderSettings(res.settingsSchema);
  });

  // Schneiden-Buttons (aktuell ausgeblendet, siehe index.html)
  // document.getElementById("startBtn").addEventListener("click", async () => {
  //   if (!state.videos.length) { showToast("Keine Videos ausgewählt.", "warn"); return; }
  //   setBusy(true);
  //   const res = await apiPost("/api/start_processing", { videos: state.videos });
  //   if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
  // });
  //
  // document.getElementById("mlPipelineBtn").addEventListener("click", async () => {
  //   if (!state.videos.length) { showToast("Keine Videos ausgewählt.", "warn"); return; }
  //   setBusy(true);
  //   const res = await apiPost("/api/run_ml_pipeline", { videos: state.videos });
  //   if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
  // });

  document.getElementById("mlPipelineFcpxmlBtn").addEventListener("click", async () => {
    if (!state.videos.length) { showToast("Keine Videos ausgewählt.", "warn"); return; }
    setBusy(true);
    const res = await apiPost("/api/run_ml_pipeline", { videos: state.videos, exportFcpxml: true });
    if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
  });

  document.getElementById("exportFcpxmlBtn").addEventListener("click", async () => {
    if (!state.videos.length) { showToast("Keine Videos ausgewählt.", "warn"); return; }
    setBusy(true);
    const res = await apiPost("/api/export_fcpxml", { videos: state.videos });
    if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
  });

  document.getElementById("testModelBtn").addEventListener("click", openTestModelModal);
  document.getElementById("closeTestBtn").addEventListener("click", closeTestModelModal);
  document.getElementById("closeTestResultBtn").addEventListener("click", closeTestModelModal);
  document.getElementById("testAgainBtn").addEventListener("click", showTestForm);
  document.getElementById("browseTestVideoBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/pick_files", { kind: "video" });
    if (res.ok && res.paths.length) document.getElementById("testVideoPath").value = res.paths[0];
  });
  document.getElementById("runTestBtn").addEventListener("click", async () => {
    const videoPath = document.getElementById("testVideoPath").value.trim();
    if (!videoPath) { showToast("Bitte ein Testvideo auswählen.", "warn"); return; }
    setBusy(true);
    const res = await apiPost("/api/test_model", {
      videoPath,
      confidence: parseFloat(document.getElementById("testConfidence").value),
      sampleEveryN: parseInt(document.getElementById("testEveryN").value, 10),
      maxMinutes: parseFloat(document.getElementById("testMaxMinutes").value) || 0,
    });
    if (res.ok) closeTestModelModal();
    else { setBusy(false); showToast(res.error, "error"); }
  });
  document.getElementById("openTestVideoBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/open_test_output", { folder: false });
    if (!res.ok) showToast(res.error, "error");
  });
  document.getElementById("openTestFolderBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/open_test_output", { folder: true });
    if (!res.ok) showToast(res.error, "error");
  });

  document.getElementById("trainBtn").addEventListener("click", openTrainOverview);
  document.getElementById("trainOverviewCancelBtn").addEventListener("click", closeTrainOverview);
  document.getElementById("trainOverviewStartBtn").addEventListener("click", async () => {
    closeTrainOverview();
    setBusy(true);
    const res = await apiPost("/api/run_training");
    if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
  });

  document.getElementById("cancelBtn").addEventListener("click", async () => {
    const res = await apiPost("/api/cancel");
    if (res.ok) {
      onStatus("Abbruch wird ausgefuehrt...");
      showToast("Abbruch angefordert.", "warn");
    } else {
      showToast(res.error, "warn");
    }
  });

  document.getElementById("reviewBtn").addEventListener("click", openReview);
  document.getElementById("backToMainBtn").addEventListener("click", showMainView);

  document.getElementById("acceptBtn").addEventListener("click", acceptCurrent);
  document.getElementById("rejectBtn").addEventListener("click", rejectCurrent);
  document.getElementById("deleteBoxBtn").addEventListener("click", deleteSelectedBox);
  document.getElementById("prevBtn").addEventListener("click", prevImage);
  document.getElementById("nextBtn").addEventListener("click", nextImage);

  document.addEventListener("keydown", onKeyDown);
  bindCanvasEvents();
}

function onKeyDown(e) {
  if (document.getElementById("reviewView").classList.contains("hidden")) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;

  const key = e.key.toLowerCase();
  if (key === "a") acceptCurrent();
  else if (key === "d") rejectCurrent();
  else if (e.key === "Delete") deleteSelectedBox();
  else if (/^[1-9]$/.test(e.key)) {
    const idx = parseInt(e.key, 10) - 1;
    if (idx < state.classes.length) selectClass(idx);
  }
}

// ------------------------------------------------------------
// Spiele / Dropdown
// ------------------------------------------------------------

function renderGameDropdown() {
  const sel = document.getElementById("gameDropdown");
  sel.innerHTML = "";
  state.games.forEach((g) => {
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    if (g === state.currentGame) opt.selected = true;
    sel.appendChild(opt);
  });
}

function updateReviewBadge(count) {
  document.getElementById("reviewBtn").textContent =
    count === 0 ? "🧾 Review Queue (0)" : `🧾 Review Queue (${count})`;
}

// ------------------------------------------------------------
// Add-Game-Modal
// ------------------------------------------------------------

async function openEditGameModal() {
  if (!state.currentGame) return;
  const info = await apiGet(`/api/game_info?name=${encodeURIComponent(state.currentGame)}`);
  if (!info.ok) { showToast(info.error, "error"); return; }
  state.editTarget = info.name;
  document.getElementById("editGameTitle").textContent = `Spiel bearbeiten – ${info.name}`;
  document.getElementById("editGameInfo").textContent =
    `Modell: ${info.hasModel ? "vorhanden" : "noch keins"}\n` +
    `Bilder: ${info.train} train, ${info.val} val, ${info.review} in der Review Queue`;
  document.getElementById("editGameClasses").value = info.classes.join(", ");
  document.getElementById("editGameHint").textContent = info.classesLocked
    ? "Es gibt schon ein Modell bzw. gelabelte Bilder: Bestehende Klassen können nicht umbenannt, umsortiert " +
      "oder entfernt werden. Neue Klassen nur am Ende ergänzen (z.B. „..., neue_klasse“)."
    : "Noch keine Labels und kein Modell: Klassen können frei geändert werden.";
  document.getElementById("editGameModal").classList.remove("hidden");
}

function closeEditGameModal() {
  document.getElementById("editGameModal").classList.add("hidden");
}

async function saveEditGame() {
  const classes = document.getElementById("editGameClasses").value
    .split(",").map((c) => c.trim()).filter(Boolean);
  const res = await apiPost("/api/edit_game", { name: state.editTarget, classes });
  if (res.ok) {
    applyState(res.state);
    closeEditGameModal();
    showToast(`Spiel '${state.editTarget}' wurde aktualisiert.`, "success");
  } else {
    showToast(res.error, "error");
  }
}

async function openDeleteGameModal() {
  if (!state.currentGame) return;
  const info = await apiGet(`/api/game_info?name=${encodeURIComponent(state.currentGame)}`);
  if (!info.ok) { showToast(info.error, "error"); return; }
  if (info.isLast) { showToast("Das letzte Spiel kann nicht gelöscht werden.", "warn"); return; }

  state.deleteTarget = info.name;
  document.getElementById("deleteGameInfo").textContent =
    `Spiel: ${info.name}\n` +
    `Modell: ${info.hasModel ? "vorhanden" : "keins"}\n` +
    `Datensatz: ${info.train} Trainings- und ${info.val} Validierungsbilder\n` +
    `Review Queue: ${info.review} Bilder\n` +
    `Speicherplatz: ca. ${info.sizeMb} MB\n\n` +
    `Das Spiel verschwindet aus der Liste. Modell, Datensatz und Review Queue werden in den Ordner ` +
    `„_deleted_games“ im Projekt verschoben (nicht sofort unwiderruflich gelöscht).`;
  document.getElementById("deleteGameConfirmLabel").textContent =
    `Zur Bestätigung den Spielnamen eingeben: ${info.name}`;
  const input = document.getElementById("deleteGameConfirmInput");
  input.value = "";
  document.getElementById("confirmDeleteGameBtn").disabled = true;
  document.getElementById("deleteGameModal").classList.remove("hidden");
  input.focus();
}

function closeDeleteGameModal() {
  document.getElementById("deleteGameModal").classList.add("hidden");
}

async function confirmDeleteGame() {
  const name = state.deleteTarget;
  const res = await apiPost("/api/delete_game", {
    name,
    confirmName: document.getElementById("deleteGameConfirmInput").value.trim(),
  });
  if (!res.ok) { showToast(res.error, "error"); return; }

  closeDeleteGameModal();
  applyState(res.state);
  showToast(
    `Spiel '${name}' gelöscht.` + (res.trashDir ? ` Dateien liegen in: ${res.trashDir}` : ""),
    "success");
  if (res.problems && res.problems.length) showToast(res.problems.join("; "), "warn");
}

function openAddGameModal() {
  document.getElementById("newGameName").value = "";
  document.getElementById("newGameModelPath").value = "";
  document.getElementById("newGameVideoPaths").value = "";
  document.getElementById("newGameClasses").value = "";
  state.newGameVideos = [];
  document.querySelector('input[name="gameSource"][value="model"]').checked = true;
  updateGameSourceRows();
  updateAddGameInfo();
  document.getElementById("addGameModal").classList.remove("hidden");
}

function closeAddGameModal() {
  document.getElementById("addGameModal").classList.add("hidden");
}

function updateGameSourceRows() {
  const source = document.querySelector('input[name="gameSource"]:checked').value;
  document.getElementById("sourceModelRow").classList.toggle("hidden", source !== "model");
  document.getElementById("sourceVideoRow").classList.toggle("hidden", source !== "video");
}

async function openTrainOverview() {
  const o = await apiGet("/api/training_overview");
  if (!o.ok) { showToast(o.error, "error"); return; }

  const now = o.now, last = o.last;
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const delta = (a, b) => {
    if (b === null || b === undefined) return "–";
    const d = a - b;
    return d > 0 ? `+${d}` : String(d);
  };
  const row = (label, nowVal, lastVal) =>
    `<tr><td>${esc(label)}</td><td class="num">${lastVal ?? "–"}</td><td class="num">${nowVal}</td>` +
    `<td class="num">${delta(nowVal, lastVal)}</td></tr>`;

  document.getElementById("trainOverviewTitle").textContent =
    o.firstModel ? `Erstes Modell trainieren – ${o.game}` : `Modell neu trainieren – ${o.game}`;
  document.getElementById("trainOverviewSummary").textContent = last
    ? `Letztes Training: ${o.lastTimestamp} (insgesamt ${o.trainingCount} erfasst)\n` +
      `Seit dem letzten Training: ${o.added} neue und ${o.removed} entfernte Bilder.`
    : o.firstModel
      ? "Das ist das erste Training für dieses Spiel."
      : "Für dieses Spiel ist noch kein früherer Trainingsstand erfasst – ab jetzt wird jeder Stand gespeichert, " +
        "der nächste Vergleich ist dann möglich.";

  let rows = row("Trainingsbilder", now.train, last?.train) + row("Validierungsbilder", now.val, last?.val) +
    row("Bilder gesamt", now.total, last?.total) + row("davon mit Boxen", now.with_boxes, last?.with_boxes) +
    row("Boxen gesamt", now.boxes, last?.boxes);
  const names = [...new Set([...Object.keys(now.per_class), ...Object.keys(last?.per_class ?? {})])];
  names.forEach((n) => { rows += row(`Boxen „${n}“`, now.per_class[n] ?? 0, last ? (last.per_class[n] ?? 0) : null); });
  document.getElementById("trainOverviewRows").innerHTML = rows;

  const hints = [];
  if (o.reviewPending) hints.push(`${o.reviewPending} Bilder warten noch in der Review Queue und werden NICHT mittrainiert.`);
  if (last && o.added === 0) hints.push("Keine neuen Daten seit dem letzten Training – ein erneutes Training bringt vermutlich nichts.");
  if (o.problem) hints.push(o.problem);
  const hintEl = document.getElementById("trainOverviewHint");
  hintEl.textContent = hints.join("\n");
  hintEl.classList.toggle("hidden", !hints.length);

  document.getElementById("trainOverviewStartBtn").disabled = !!o.problem;
  document.getElementById("trainOverviewModal").classList.remove("hidden");
}

function closeTrainOverview() {
  document.getElementById("trainOverviewModal").classList.add("hidden");
}

function openTestModelModal() {
  document.getElementById("testModelGameInfo").textContent =
    `Spiel: ${state.currentGame}\nModell: ${state.modelPath || "-"}`;
  const confInput = document.querySelector('#settingsBody [data-attr="yolo_confidence"]');
  if (confInput && confInput.value) document.getElementById("testConfidence").value = confInput.value;
  showTestForm();
  document.getElementById("testModelModal").classList.remove("hidden");
}

function closeTestModelModal() {
  document.getElementById("testModelModal").classList.add("hidden");
}

function showTestForm() {
  document.getElementById("testModelForm").classList.remove("hidden");
  document.getElementById("testModelResult").classList.add("hidden");
}

function onModelTestResult(r) {
  const pct = (a, b) => (b ? ((a / b) * 100).toFixed(0) : "0");
  document.getElementById("testResultSummary").textContent =
    `Spiel: ${r.game}\n` +
    `Geprüfte Frames: ${r.framesChecked} (jeder ${r.sampleEveryN}., Mindest-Konfidenz ${r.confidence})\n` +
    `Frames mit Erkennung: ${r.framesWithDetection} (${pct(r.framesWithDetection, r.framesChecked)} %)\n` +
    `Erkennungen gesamt: ${r.detections}, Ø Konfidenz: ${r.avgConfidence.toFixed(2)}\n` +
    (r.firstTimes.length ? `Erste Erkennungen bei: ${r.firstTimes.map((t) => t + "s").join(", ")}\n` : "") +
    `Test-Video: ${r.outputPath}`;

  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  document.getElementById("testResultClasses").innerHTML = r.classes.length
    ? r.classes.map((c) =>
        `<tr><td>${esc(c.name)}</td><td class="num">${c.count}</td>` +
        `<td class="hint">Ø ${c.avg.toFixed(2)} (min ${c.min.toFixed(2)}, max ${c.max.toFixed(2)})</td></tr>`).join("")
    : `<tr><td colspan="3" class="hint">Keine Erkennungen. Konfidenz senken oder anderes Video testen.</td></tr>`;
  document.getElementById("testResultBuckets").innerHTML = r.buckets.map((b) =>
    `<tr><td>${esc(b.label)}</td><td class="num">${b.count}</td><td></td></tr>`).join("");

  document.getElementById("testModelForm").classList.add("hidden");
  document.getElementById("testModelResult").classList.remove("hidden");
  document.getElementById("testModelModal").classList.remove("hidden");
}

function onFirstModelReady({ game, modelPath, versionsDir }) {
  document.getElementById("firstModelInfo").textContent =
    `Das erste Modell für '${game}' wurde trainiert und gespeichert.\n\n` +
    `Modell:  ${modelPath}\n` +
    `Trainingsläufe (Kurven, Metriken, weitere Gewichte):  ${versionsDir}\n\n` +
    `Ab jetzt läuft alles wie gewohnt: Video auswählen, ML Pipeline + FCPXML starten, ` +
    `Review Queue labeln und bei Bedarf neu trainieren.`;
  document.getElementById("firstModelModal").classList.remove("hidden");
}

function updateAddGameInfo() {
  const name = document.getElementById("newGameName").value.trim() || "<Spielname>";
  document.getElementById("newGamePathsInfo").textContent =
    "Wird angelegt unter:\n" +
    `  Modell:        models/${name}/current.pt\n` +
    `  Dataset:       datasets/${name}\n` +
    `  Review-Queue:  review_queue/${name}`;
}

// ------------------------------------------------------------
// Video-Liste
// ------------------------------------------------------------

function addVideo(path) {
  if (!state.videos.includes(path)) {
    state.videos.push(path);
    renderVideoList();
  }
}

function renderVideoList() {
  const ul = document.getElementById("videoList");
  ul.innerHTML = "";
  state.videos.forEach((v, i) => {
    const li = document.createElement("li");
    const span = document.createElement("span");
    span.textContent = v;
    li.appendChild(span);

    const rm = document.createElement("button");
    rm.textContent = "✕";
    rm.className = "icon-btn small";
    rm.addEventListener("click", () => {
      state.videos.splice(i, 1);
      renderVideoList();
    });
    li.appendChild(rm);
    ul.appendChild(li);
  });
}

// ------------------------------------------------------------
// Einstellungen
// ------------------------------------------------------------

function renderSettings(schema) {
  const body = document.getElementById("settingsBody");
  body.innerHTML = "";
  schema.forEach((item) => {
    const row = document.createElement("div");
    row.className = "setting-row";

    const label = document.createElement("label");
    label.textContent = item.label;
    row.appendChild(label);

    let input;
    if (item.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!item.value;
    } else if (item.type === "choice") {
      input = document.createElement("select");
      (item.options || []).forEach((opt) => {
        const optionEl = document.createElement("option");
        optionEl.value = opt;
        optionEl.textContent = opt;
        if (opt === item.value) optionEl.selected = true;
        input.appendChild(optionEl);
      });
    } else if (item.type === "text") {
      input = document.createElement("input");
      input.type = "text";
      input.value = item.value ?? "";
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = item.type === "int" ? "1" : "0.01";
      input.value = item.value;
    }
    input.dataset.attr = item.attr;
    input.dataset.type = item.type;
    row.appendChild(input);

    const desc = document.createElement("small");
    desc.textContent = item.desc;
    row.appendChild(desc);

    body.appendChild(row);
  });
}

function collectSettingsValues() {
  const values = {};
  document.querySelectorAll("#settingsBody [data-attr]").forEach((input) => {
    if (input.dataset.type === "bool") values[input.dataset.attr] = input.checked;
    else if (input.dataset.type === "choice" || input.dataset.type === "text") values[input.dataset.attr] = input.value;
    else values[input.dataset.attr] = parseFloat(input.value);
  });
  return values;
}

// ------------------------------------------------------------
// Verarbeitung: Callbacks aus den Server-Sent-Events
// ------------------------------------------------------------

function onLog(msg) {
  const el = document.getElementById("logConsole");
  el.textContent += msg;
  el.scrollTop = el.scrollHeight;
}

function onProgress(phase, pct) {
  document.getElementById("phaseLabel").textContent = `Phase: ${phase} (${pct.toFixed(0)}%)`;
  document.getElementById("subProgress").style.width = `${pct}%`;
}

function onOverallProgress(idx, total) {
  const pct = total ? (idx / total) * 100 : 0;
  document.getElementById("overallProgress").style.width = `${pct}%`;
}

function onStatus(text) {
  document.getElementById("statusLine").textContent = text;
}

function onBusy(busy) {
  setBusy(busy);
}

function onDone(result) {
  setBusy(false);
  showToast(result.message || "Fertig.", "success");
}

function onError(msg) {
  setBusy(false);
  showToast("Fehler: " + msg, "error");
}

function onCancelled(msg) {
  setBusy(false);
  onStatus("Abgebrochen");
  showToast(msg || "Vorgang abgebrochen.", "warn");
}

function onReviewCount(count) {
  updateReviewBadge(count);
}

function onMlReport(report) {
  const fill = (id, rowsHtml) => { document.getElementById(id).innerHTML = rowsHtml; };
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  fill("mlReportRows", report.rows.map((r) =>
    `<tr><td>${esc(r.label)}</td><td class="num">${r.value}</td><td class="hint">${esc(r.hint)}</td></tr>`).join(""));
  fill("mlReportBuckets", report.buckets.map((b) =>
    `<tr><td>${esc(b.label)}</td><td class="num">${b.count}</td><td></td></tr>`).join(""));
  document.getElementById("mlReportPanel").classList.remove("hidden");
}

function setBusy(busy) {
  state.busy = busy;
  document.getElementById("busyOverlay").classList.toggle("hidden", !busy);
  updateActionButtons();
}

// ------------------------------------------------------------
// Toasts
// ------------------------------------------------------------

function showToast(msg, type = "info") {
  const container = document.getElementById("toastContainer");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  container.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 300);
  }, 4500);
}

// ------------------------------------------------------------
// View-Umschaltung
// ------------------------------------------------------------

function showReviewView() {
  document.getElementById("mainView").classList.add("hidden");
  document.getElementById("reviewView").classList.remove("hidden");
}

function showMainView() {
  document.getElementById("reviewView").classList.add("hidden");
  document.getElementById("mainView").classList.remove("hidden");
  refreshState();
}

// ------------------------------------------------------------
// Review / Labeling-Tool
// ------------------------------------------------------------

const canvas = document.getElementById("reviewCanvas");
const ctx = canvas.getContext("2d");
let currentImg = new Image();
let mouseState = { mode: null };

async function openReview() {
  const res = await apiPost("/api/review/open");
  state.reviewImages = res.images;
  state.classes = res.classes;
  state.reviewIndex = 0;
  state.selectedClass = 0;
  renderClassButtons();
  showReviewView();
  loadReviewImage();
}

function renderClassButtons() {
  const container = document.getElementById("classButtons");
  container.innerHTML = "";
  state.classes.forEach((name, i) => {
    const btn = document.createElement("button");
    btn.className = "class-btn";
    btn.style.background = classColor(i);
    btn.textContent = `${i + 1}: ${name}`;
    btn.addEventListener("click", () => selectClass(i));
    container.appendChild(btn);
  });
  updateClassButtonStyles();
}

function selectClass(i) {
  state.selectedClass = i;
  updateClassButtonStyles();
  if (state.selectedBox !== null && state.reviewBoxes[state.selectedBox]) {
    state.reviewBoxes[state.selectedBox].cls = i;
    redraw();
    saveLabels();
  }
}

function updateClassButtonStyles() {
  document.querySelectorAll(".class-btn").forEach((btn, i) => {
    btn.classList.toggle("active", i === state.selectedClass);
  });
}

function loadReviewImage() {
  const entry = state.reviewImages[state.reviewIndex];
  document.getElementById("reviewCounter").textContent =
    `Bild ${state.reviewImages.length ? state.reviewIndex + 1 : 0} / ${state.reviewImages.length}`;

  if (!entry) {
    state.reviewBoxes = [];
    state.selectedBox = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#fff";
    ctx.font = "24px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Review Queue leer 🎉", canvas.width / 2, canvas.height / 2);
    return;
  }

  currentImg = new Image();
  currentImg.onload = async () => {
    const res = await apiGet(`/api/review/labels/${entry.id}`);
    state.reviewBoxes = res.boxes;
    state.selectedBox = null;
    redraw();
  };
  currentImg.src = `/api/review/image/${entry.id}?t=${Date.now()}`;
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (currentImg.complete && currentImg.naturalWidth) {
    ctx.drawImage(currentImg, 0, 0, canvas.width, canvas.height);
  }
  state.reviewBoxes.forEach((b, i) => drawBox(b, i));
}

function drawBox(b, i) {
  const x = b.x * canvas.width;
  const y = b.y * canvas.height;
  const w = b.w * canvas.width;
  const h = b.h * canvas.height;
  const color = classColor(b.cls);

  ctx.strokeStyle = color;
  ctx.lineWidth = state.selectedBox === i ? 3 : 2;
  ctx.strokeRect(x - w / 2, y - h / 2, w, h);

  ctx.fillStyle = color;
  ctx.font = "bold 13px sans-serif";
  ctx.textAlign = "left";
  const name = state.classes[b.cls] ?? `cls${b.cls}`;
  ctx.fillText(name, x - w / 2 + 4, y - h / 2 - 6);

  if (state.selectedBox === i) {
    const corners = [
      [x - w / 2, y - h / 2], [x + w / 2, y - h / 2],
      [x - w / 2, y + h / 2], [x + w / 2, y + h / 2],
    ];
    ctx.fillStyle = "yellow";
    ctx.strokeStyle = "black";
    corners.forEach(([hx, hy]) => {
      ctx.fillRect(hx - HANDLE_SIZE, hy - HANDLE_SIZE, HANDLE_SIZE * 2, HANDLE_SIZE * 2);
      ctx.strokeRect(hx - HANDLE_SIZE, hy - HANDLE_SIZE, HANDLE_SIZE * 2, HANDLE_SIZE * 2);
    });
  }
}

function canvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
}

function findBox(x, y) {
  for (let i = 0; i < state.reviewBoxes.length; i++) {
    const b = state.reviewBoxes[i];
    const cx = b.x * canvas.width, cy = b.y * canvas.height;
    const w = b.w * canvas.width, h = b.h * canvas.height;
    if (x >= cx - w / 2 && x <= cx + w / 2 && y >= cy - h / 2 && y <= cy + h / 2) return i;
  }
  return null;
}

function hitTestHandle(idx, x, y) {
  if (idx === null || idx >= state.reviewBoxes.length) return null;
  const b = state.reviewBoxes[idx];
  const cx = b.x * canvas.width, cy = b.y * canvas.height;
  const w = b.w * canvas.width, h = b.h * canvas.height;
  const corners = {
    tl: [cx - w / 2, cy - h / 2], tr: [cx + w / 2, cy - h / 2],
    bl: [cx - w / 2, cy + h / 2], br: [cx + w / 2, cy + h / 2],
  };
  for (const [name, [hx, hy]] of Object.entries(corners)) {
    if (Math.abs(x - hx) <= HANDLE_SIZE + 4 && Math.abs(y - hy) <= HANDLE_SIZE + 4) return name;
  }
  return null;
}

function resizeSelected(x, y, corner) {
  const b = state.reviewBoxes[state.selectedBox];
  const cx = b.x * canvas.width, cy = b.y * canvas.height;
  const w = b.w * canvas.width, h = b.h * canvas.height;
  let left = cx - w / 2, top = cy - h / 2, right = cx + w / 2, bottom = cy + h / 2;

  if (corner.includes("l")) left = x;
  if (corner.includes("r")) right = x;
  if (corner.includes("t")) top = y;
  if (corner.includes("b")) bottom = y;

  if (right - left < 10 || bottom - top < 10) return;

  b.x = (left + right) / 2 / canvas.width;
  b.y = (top + bottom) / 2 / canvas.height;
  b.w = (right - left) / canvas.width;
  b.h = (bottom - top) / canvas.height;
}

function bindCanvasEvents() {
  canvas.addEventListener("mousedown", (e) => {
    const { x, y } = canvasPos(e);

    if (state.selectedBox !== null) {
      const corner = hitTestHandle(state.selectedBox, x, y);
      if (corner) { mouseState = { mode: "resize", corner }; return; }
    }

    const hit = findBox(x, y);
    state.selectedBox = hit;

    if (hit !== null) {
      const b = state.reviewBoxes[hit];
      const cx = b.x * canvas.width, cy = b.y * canvas.height;
      mouseState = { mode: "drag", dragOffset: [cx - x, cy - y] };
    } else {
      mouseState = { mode: "create", startX: x, startY: y };
    }
    redraw();
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!mouseState.mode) return;
    const { x, y } = canvasPos(e);

    if (mouseState.mode === "resize") {
      resizeSelected(x, y, mouseState.corner);
      redraw();
    } else if (mouseState.mode === "drag" && state.selectedBox !== null) {
      const [dx, dy] = mouseState.dragOffset;
      state.reviewBoxes[state.selectedBox].x = (x + dx) / canvas.width;
      state.reviewBoxes[state.selectedBox].y = (y + dy) / canvas.height;
      redraw();
    } else if (mouseState.mode === "create") {
      redraw();
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      ctx.strokeRect(mouseState.startX, mouseState.startY, x - mouseState.startX, y - mouseState.startY);
    }
  });

  canvas.addEventListener("mouseup", async (e) => {
    const { x, y } = canvasPos(e);

    if (mouseState.mode === "resize" || mouseState.mode === "drag") {
      mouseState = {};
      await saveLabels();
      return;
    }

    if (mouseState.mode === "create") {
      const sx = mouseState.startX, sy = mouseState.startY;
      if (Math.abs(x - sx) > 8 && Math.abs(y - sy) > 8) {
        const left = Math.min(sx, x), right = Math.max(sx, x);
        const top = Math.min(sy, y), bottom = Math.max(sy, y);
        state.reviewBoxes.push({
          cls: state.selectedClass,
          x: (left + right) / 2 / canvas.width,
          y: (top + bottom) / 2 / canvas.height,
          w: (right - left) / canvas.width,
          h: (bottom - top) / canvas.height,
        });
        state.selectedBox = state.reviewBoxes.length - 1;
        await saveLabels();
      }
      mouseState = {};
      redraw();
    }
  });
}

async function saveLabels() {
  const entry = state.reviewImages[state.reviewIndex];
  if (!entry) return;
  await apiPost(`/api/review/labels/${entry.id}`, { boxes: state.reviewBoxes });
}

function deleteSelectedBox() {
  if (state.selectedBox === null) return;
  state.reviewBoxes.splice(state.selectedBox, 1);
  state.selectedBox = null;
  redraw();
  saveLabels();
}

async function acceptCurrent() {
  const entry = state.reviewImages[state.reviewIndex];
  if (!entry) return;
  const res = await apiPost(`/api/review/accept/${entry.id}`);
  if (res.ok) removeCurrentFromList();
  else showToast(res.error, "error");
}

async function rejectCurrent() {
  const entry = state.reviewImages[state.reviewIndex];
  if (!entry) return;
  const res = await apiPost(`/api/review/reject/${entry.id}`);
  if (res.ok) removeCurrentFromList();
  else showToast(res.error, "error");
}

function removeCurrentFromList() {
  state.reviewImages.splice(state.reviewIndex, 1);
  if (state.reviewIndex >= state.reviewImages.length) {
    state.reviewIndex = Math.max(0, state.reviewImages.length - 1);
  }
  loadReviewImage();
  updateReviewBadge(state.reviewImages.length);
}

function nextImage() {
  if (state.reviewIndex < state.reviewImages.length - 1) {
    state.reviewIndex++;
    loadReviewImage();
  }
}

function prevImage() {
  if (state.reviewIndex > 0) {
    state.reviewIndex--;
    loadReviewImage();
  }
}
