/* ==========================================================================
 * Scenes tab — frontend logic (STO-SCN-146, EPI-SCN-SCENE-MANAGER).
 *
 * A sibling tab to Rank in the Studio app. Self-contained IIFE: shares NO
 * globals with app.js (Rank), only the DOM + the read-only /api/scenes API.
 *
 * This file is the SHELL: tab switching, the scene selector header, the
 * config area + view switcher (Metadata 153 / Spine 147 / Subsets 148), and
 * the New-Scene button (149). The individual views render their own bodies;
 * placeholders here name the story that fills each one.
 * ========================================================================== */
"use strict";

(function scenesTab() {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const S = {
    scenes: [],
    scene: null,     // selected scene name
    view: "meta",    // active config view: meta | spine | subsets
  };

  const el = {
    tabBar: $("#tab-bar"),
    tabRank: $("#tab-rank"),
    tabScenes: $("#tab-scenes"),
    strip: $("#scenes-strip"),
    scLeft: $("#sc-left"),
    scRight: $("#sc-right"),
    newBtn: $("#new-scene-btn"),
    viewBar: $("#scenes-viewbar"),
    view: $("#scenes-view"),
  };

  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
    return r.json();
  }

  // ---- tab switching (Rank | Scenes) -----------------------------------
  let scenesLoaded = false;
  function selectTab(tab) {
    el.tabBar.querySelectorAll("button").forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === tab));
    const isScenes = tab === "scenes";
    el.tabRank.hidden = isScenes;
    el.tabScenes.hidden = !isScenes;
    if (isScenes && !scenesLoaded) {
      scenesLoaded = true;
      loadScenes();
    }
  }

  // ---- scene selector header -------------------------------------------
  async function loadScenes() {
    try {
      S.scenes = await api("/api/all-scenes");   // [{name, thumb}] — ALL scene dirs (incl. pre-render)
    } catch (e) {
      el.strip.innerHTML = `<div class="scenes-empty">Failed to load scenes: ${e.message}</div>`;
      return;
    }
    renderStrip();
    if (S.scenes.length && !S.scene) selectScene(S.scenes[0].name);
  }

  function renderStrip() {
    if (!S.scenes.length) {
      el.strip.innerHTML = `<div class="scenes-empty">No scenes yet. Use <b>+ New Scene</b>.</div>`;
      return;
    }
    el.strip.innerHTML = S.scenes.map((sc) => `
      <div class="scene-card" data-scene="${sc.name}" title="${sc.name}">
        ${sc.thumb ? `<img src="${sc.thumb}" loading="lazy" alt="${sc.name}">`
                   : `<div class="noimg">&#9633;</div>`}
        <div class="nm">${sc.name}</div>
      </div>`).join("");
    el.strip.querySelectorAll(".scene-card").forEach((card) =>
      card.addEventListener("click", () => selectScene(card.dataset.scene)));
    highlight();
  }

  function highlight() {
    el.strip.querySelectorAll(".scene-card").forEach((c) =>
      c.classList.toggle("selected", c.dataset.scene === S.scene));
    const sel = el.strip.querySelector(".scene-card.selected");
    if (sel) sel.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
  }

  function selectScene(name) {
    S.scene = name;
    highlight();
    renderView();
  }

  // ---- view switcher (Metadata | Spine | Subsets) ----------------------
  function selectView(view) {
    S.view = view;
    el.viewBar.querySelectorAll("button").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === view));
    renderView();
  }

  function renderView() {
    if (!S.scene) {
      el.view.innerHTML = `<em>Select a scene above.</em>`;
      return;
    }
    // Each view fills its own body. Until its story lands, show a placeholder
    // naming the story. window.scenesViews.<view> lets later stories register
    // a real renderer without touching this shell.
    const reg = (window.scenesViews || {})[S.view];
    if (typeof reg === "function") { reg(el.view, S.scene); return; }
    const where = { meta: "STO-SCN-153", spine: "STO-SCN-147", subsets: "STO-SCN-148" }[S.view];
    el.view.innerHTML =
      `<div class="view-placeholder">` +
      `<h3>${S.scene}</h3>` +
      `<p>The <b>${S.view}</b> view mounts here (<code>${where}</code>).</p>` +
      `</div>`;
  }

  // ---- New Scene → ingest + canonicalize (STO-SCN-149) -----------------
  function onNewScene() {
    S.scene = null; highlight();
    el.view.innerHTML = `
      <div class="newscene">
        <h3>New Scene</h3>
        <label>Name <input id="ns-name" type="text" placeholder="e.g. back patio" autofocus></label>
        <label>Source path <input id="ns-src" type="text"
          placeholder="server-side path: a video, an image, or a folder"></label>
        <div class="ns-row">
          <label>Mode
            <select id="ns-mode"><option value="copy">copy (keep source)</option>
              <option value="move">move (no copy)</option></select></label>
          <label>Video fps <input id="ns-fps" type="number" value="2" min="0.1" step="0.1" style="width:64px"></label>
        </div>
        <div class="ns-actions">
          <button id="ns-create">Create + Ingest</button>
          <span id="ns-status" class="ns-status"></span>
        </div>
        <div class="ns-prog"><div id="ns-bar"></div></div>
        <p class="ns-hint">Server-side path (we operate locally → MOVE/COPY on the host). Video →
          frames @ fps; images/folder → canonicalized to content-hash. Browser upload: later.</p>
      </div>`;
    el.view.querySelector("#ns-create").addEventListener("click", createScene);
    el.view.querySelector("#ns-name").addEventListener("keydown", (e) => {
      if (e.key === "Enter") el.view.querySelector("#ns-src").focus();
    });
  }

  function nsStatus(msg, cls) {
    const s = el.view.querySelector("#ns-status");
    if (s) { s.textContent = msg; s.className = "ns-status " + (cls || ""); }
  }
  function nsBar(done, total) {
    const bar = el.view.querySelector("#ns-bar");
    if (bar) bar.style.width = total ? `${Math.round((done / total) * 100)}%` : "0%";
  }

  async function createScene() {
    const name = el.view.querySelector("#ns-name").value.trim();
    const source = el.view.querySelector("#ns-src").value.trim();
    const mode = el.view.querySelector("#ns-mode").value;
    const fps = el.view.querySelector("#ns-fps").value || "2";
    if (!name) { nsStatus("name required", "err"); return; }
    el.view.querySelector("#ns-create").disabled = true;
    let scene;
    try {
      nsStatus("creating scene…");
      const r = await api("/api/scene-new", { method: "POST", body: JSON.stringify({ name }) });
      if (r.error) throw new Error(r.error);
      scene = r.scene;
      nsStatus(`created ${scene}` + (source ? " · ingesting…" : ""), "ok");
    } catch (e) {
      nsStatus("create failed: " + e.message, "err");
      el.view.querySelector("#ns-create").disabled = false;
      return;
    }
    if (!source) { await finishNewScene(scene); return; }   // empty scene, no ingest
    try {
      const ig = await api(`/api/scene/${encodeURIComponent(scene)}/ingest`,
        { method: "POST", body: JSON.stringify({ source, mode, fps: Number(fps) }) });
      if (ig.error) throw new Error(ig.error);
      await pollIngest(scene);
    } catch (e) {
      nsStatus("ingest failed: " + e.message, "err");
      el.view.querySelector("#ns-create").disabled = false;
    }
  }

  async function pollIngest(scene) {
    for (let i = 0; i < 100000; i++) {
      let st;
      try { st = await api(`/api/scene/${encodeURIComponent(scene)}/ingest-status`); }
      catch { st = { status: "none" }; }
      if (st.phase) nsStatus(`${st.phase} ${st.done || 0}/${st.total || 0}`, st.status === "error" ? "err" : "");
      nsBar(st.done || 0, st.total || 0);
      if (st.status === "done") { nsStatus(`ingested ${st.n} images`, "ok"); return finishNewScene(scene); }
      if (st.status === "error") { nsStatus("ingest error: " + (st.error || ""), "err");
        el.view.querySelector("#ns-create").disabled = false; return; }
      await new Promise((r) => setTimeout(r, 700));
    }
  }

  async function finishNewScene(scene) {
    await loadScenes();          // refresh selector (uses /api/all-scenes)
    S.scene = scene; highlight();
    selectView("meta");          // land on Metadata for the new scene
  }

  // ---- wire-up ----------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    if (!el.tabBar) return;   // page without the Scenes shell
    el.tabBar.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => selectTab(b.dataset.tab)));
    el.viewBar.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => selectView(b.dataset.view)));
    el.newBtn.addEventListener("click", onNewScene);
    el.scLeft.addEventListener("click", () =>
      el.strip.scrollBy({ left: -el.strip.clientWidth * 0.8, behavior: "smooth" }));
    el.scRight.addEventListener("click", () =>
      el.strip.scrollBy({ left: el.strip.clientWidth * 0.8, behavior: "smooth" }));

    // Deep-link: ?tab=scenes opens the Scenes tab on load.
    const tab = new URLSearchParams(location.search).get("tab");
    if (tab === "scenes") selectTab("scenes");
  });

  // Expose a tiny registry so later stories (147/148/153) attach view
  // renderers without editing this shell.
  window.scenesViews = window.scenesViews || {};
})();
