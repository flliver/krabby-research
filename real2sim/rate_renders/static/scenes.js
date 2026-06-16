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
      S.scenes = await api("/api/scenes");   // [{name, thumb}]
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

  // ---- New Scene (stub → STO-SCN-149) ----------------------------------
  function onNewScene() {
    el.view.innerHTML =
      `<div class="view-placeholder">` +
      `<h3>New Scene</h3>` +
      `<p>The ingest + canonicalize flow mounts here (<code>STO-SCN-149</code>).</p>` +
      `</div>`;
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
