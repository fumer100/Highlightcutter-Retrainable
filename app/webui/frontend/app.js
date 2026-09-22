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
// Init
// ------------------------------------------------------------

function whenReady(fn) {
  if (window.pywebview && window.pywebview.api) fn();
  else window.addEventListener("pywebviewready", fn);
}

whenReady(init);

async function init() {
  const s = await pywebview.api.get_state();
  applyState(s);
  bindEvents();
  if (!s.games.length) {
    showToast("Keine Spiele konfiguriert. Bitte ein Spiel hinzufügen.", "warn");
  }
}

function applyState(s) {
  state.games = s.games;
  state.currentGame = s.currentGame;
  state.classes = s.classes;
  renderGameDropdown();
  renderSettings(s.settingsSchema);
  updateReviewBadge(s.reviewCount);
}

// ------------------------------------------------------------
// Event-Wiring
// ------------------------------------------------------------

function bindEvents() {
  document.getElementById("gameDropdown").addEventListener("change", async (e) => {
    const res = await pywebview.api.select_game(e.target.value);
    if (res.ok) applyState(res.state);
    else showToast(res.error, "error");
  });

  document.getElementById("refreshGamesBtn").addEventListener("click", async () => {
    const res = await pywebview.api.refresh_games();
    if (res.ok) {
      applyState(res.state);
      showToast(`${res.state.games.length} Spiele geladen.`, "success");
    } else {
      showToast(res.error, "error");
    }
  });

  document.getElementById("addGameBtn").addEventListener("click", openAddGameModal);
  document.getElementById("cancelGameBtn").addEventListener("click", closeAddGameModal);

  document.getElementById("browseModelBtn").addEventListener("click", async () => {
    const path = await pywebview.api.browse_model_file();
    if (path) document.getElementById("newGameModelPath").value = path;
  });

  document.getElementById("loadClassesBtn").addEventListener("click", async () => {
    const path = await pywebview.api.browse_json_file();
    if (!path) return;
    const res = await pywebview.api.extract_classes_from_json(path);
    if (res.ok) document.getElementById("newGameClasses").value = res.classes.join(", ");
    else showToast(res.error, "error");
  });

  document.getElementById("newGameName").addEventListener("input", updateAddGameInfo);

  document.getElementById("saveGameBtn").addEventListener("click", async () => {
    const name = document.getElementById("newGameName").value.trim();
    const modelPath = document.getElementById("newGameModelPath").value.trim();
    const classes = document.getElementById("newGameClasses").value
      .split(",").map((c) => c.trim()).filter(Boolean);

    const res = await pywebview.api.add_game(name, modelPath, classes);
    if (res.ok) {
      applyState(res.state);
      closeAddGameModal();
      showToast(`Spiel '${name}' wurde hinzugefügt.`, "success");
    } else {
      showToast(res.error, "error");
    }
  });

  document.getElementById("selectVideosBtn").addEventListener("click", async () => {
    const paths = await pywebview.api.select_videos();
    paths.forEach(addVideo);
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
    const res = await pywebview.api.apply_settings(values);
    if (res.ok) showToast("Einstellungen übernommen.", "success");
    else showToast("Ungültige Felder: " + res.errors.join(", "), "error");
  });

  document.getElementById("resetSettingsBtn").addEventListener("click", async () => {
    const res = await pywebview.api.reset_settings();
    renderSettings(res.settingsSchema);
  });

  document.getElementById("startBtn").addEventListener("click", async () => {
    if (!state.videos.length) { showToast("Keine Videos ausgewählt.", "warn"); return; }
    setBusy(true);
    const res = await pywebview.api.start_processing(state.videos);
    if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
  });

  document.getElementById("mlPipelineBtn").addEventListener("click", async () => {
    if (!state.videos.length) { showToast("Keine Videos ausgewählt.", "warn"); return; }
    setBusy(true);
    const res = await pywebview.api.run_ml_pipeline(state.videos);
    if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
  });

  document.getElementById("trainBtn").addEventListener("click", async () => {
    setBusy(true);
    const res = await pywebview.api.run_training();
    if (!res.ok) { setBusy(false); showToast(res.error, "error"); }
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

function openAddGameModal() {
  document.getElementById("newGameName").value = "";
  document.getElementById("newGameModelPath").value = "";
  document.getElementById("newGameClasses").value = "";
  updateAddGameInfo();
  document.getElementById("addGameModal").classList.remove("hidden");
}

function closeAddGameModal() {
  document.getElementById("addGameModal").classList.add("hidden");
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
    else values[input.dataset.attr] = parseFloat(input.value);
  });
  return values;
}

// ------------------------------------------------------------
// Verarbeitung: Callbacks aus Python (window.evaluate_js)
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

function onReviewCount(count) {
  updateReviewBadge(count);
}

function setBusy(busy) {
  document.getElementById("busyOverlay").classList.toggle("hidden", !busy);
  ["startBtn", "mlPipelineBtn", "trainBtn"].forEach((id) => {
    document.getElementById(id).disabled = busy;
  });
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
}

// ------------------------------------------------------------
// Review / Labeling-Tool
// ------------------------------------------------------------

const canvas = document.getElementById("reviewCanvas");
const ctx = canvas.getContext("2d");
let currentImg = new Image();
let mouseState = { mode: null };

async function openReview() {
  const res = await pywebview.api.review_open();
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
    const res = await pywebview.api.review_get_labels(entry.id);
    state.reviewBoxes = res.boxes;
    state.selectedBox = null;
    redraw();
  };
  currentImg.src = entry.url;
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
  await pywebview.api.review_save_labels(entry.id, state.reviewBoxes);
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
  const res = await pywebview.api.review_accept(entry.id);
  if (res.ok) removeCurrentFromList();
  else showToast(res.error, "error");
}

async function rejectCurrent() {
  const entry = state.reviewImages[state.reviewIndex];
  if (!entry) return;
  const res = await pywebview.api.review_reject(entry.id);
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
