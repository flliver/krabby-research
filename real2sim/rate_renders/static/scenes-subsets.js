/* ==========================================================================
 * Scenes tab — Camera Subsets view (STO-SCN-148).
 *
 * Registers window.scenesViews.subsets. Left: subset list (primary flagged,
 * member/solve counts, datum badge). Right: a paged photo grid of the selected
 * subset's member images (layouts 1 / 2×1 / 2×2 / 3×3 / 4×4), served from
 * GET /api/photo/<scene>/<hash>.jpg. Subset list from /api/scene/<scene>/subsets.
 * ========================================================================== */
"use strict";

(function sceneSubsetsView() {
  window.scenesViews = window.scenesViews || {};

  const LAYOUTS = [
    { n: 1, cols: 1, label: "1" },
    { n: 2, cols: 2, label: "2×1" },
    { n: 4, cols: 2, label: "2×2" },
    { n: 9, cols: 3, label: "3×3" },
    { n: 16, cols: 4, label: "4×4" },
  ];

  const V = { scene: null, subsets: [], sel: null, layout: 4, page: 0 };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  async function render(container, scene) {
    if (V.scene !== scene) { V.scene = scene; V.subsets = []; V.sel = null; V.page = 0; }
    container.innerHTML = `<div class="ss-wrap">
        <div class="ss-list" id="ss-list">Loading subsets…</div>
        <div class="ss-grid-pane">
          <div class="ss-gridbar" id="ss-gridbar"></div>
          <div class="ss-grid" id="ss-grid"></div>
        </div>
      </div>`;
    if (!V.subsets.length) {
      try {
        const r = await fetch(`/api/scene/${encodeURIComponent(scene)}/subsets`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        V.subsets = d.subsets || [];
      } catch (e) {
        container.querySelector("#ss-list").innerHTML =
          `<div class="meta-err">Failed: ${esc(e.message)}</div>`;
        return;
      }
      if (!V.sel && V.subsets.length) V.sel = V.subsets[0].id;
    }
    drawList(container);
    drawBar(container);
    drawGrid(container);
  }

  function drawList(container) {
    const el = container.querySelector("#ss-list");
    if (!V.subsets.length) { el.innerHTML = `<div class="meta-err">No subsets.</div>`; return; }
    el.innerHTML = V.subsets.map((s) => `
      <div class="ss-item ${s.id === V.sel ? "sel" : ""}" data-id="${s.id}">
        <div class="ss-item-top">
          <span class="ss-id">${esc(s.label || s.id)}</span>
          ${s.is_primary ? `<span class="ss-badge prim">PRIMARY</span>` : ""}
          ${s.has_datum ? `<span class="ss-badge datum">datum</span>` : ""}
        </div>
        <div class="ss-item-sub">${esc(s.member_count)} imgs · ${esc((s.solves || []).length)} solve(s)${s.mechanism ? " · " + esc(s.mechanism) : ""}</div>
      </div>`).join("");
    el.querySelectorAll(".ss-item").forEach((it) =>
      it.addEventListener("click", () => {
        V.sel = it.dataset.id; V.page = 0;
        drawList(container); drawGrid(container);
      }));
  }

  function drawBar(container) {
    const el = container.querySelector("#ss-gridbar");
    el.innerHTML =
      `<div class="ss-layouts">` +
      LAYOUTS.map((l) => `<button data-n="${l.n}" class="${l.n === V.layout ? "active" : ""}">${l.label}</button>`).join("") +
      `</div>` +
      `<div class="ss-pager"><button id="ss-prev">←</button><span id="ss-pageind"></span><button id="ss-next">→</button></div>`;
    el.querySelectorAll(".ss-layouts button").forEach((b) =>
      b.addEventListener("click", () => { V.layout = Number(b.dataset.n); V.page = 0; drawBar(container); drawGrid(container); }));
    el.querySelector("#ss-prev").addEventListener("click", () => { stepPage(-1, container); });
    el.querySelector("#ss-next").addEventListener("click", () => { stepPage(1, container); });
  }

  function curSubset() { return V.subsets.find((s) => s.id === V.sel); }

  function stepPage(delta, container) {
    const s = curSubset(); if (!s) return;
    const pages = Math.max(1, Math.ceil(s.member_count / V.layout));
    V.page = Math.max(0, Math.min(pages - 1, V.page + delta));
    drawGrid(container);
  }

  function drawGrid(container) {
    const grid = container.querySelector("#ss-grid");
    const s = curSubset();
    if (!s) { grid.innerHTML = `<em>Select a subset.</em>`; return; }
    const members = s.members || [];
    const cols = (LAYOUTS.find((l) => l.n === V.layout) || {}).cols || 2;
    const pages = Math.max(1, Math.ceil(members.length / V.layout));
    V.page = Math.min(V.page, pages - 1);
    const start = V.page * V.layout;
    const slice = members.slice(start, start + V.layout);
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    grid.innerHTML = slice.map((h) =>
      `<div class="ss-cell"><img loading="lazy" src="/api/photo/${encodeURIComponent(V.scene)}/${encodeURIComponent(h)}.jpg" alt="${esc(h)}" title="${esc(h)}"></div>`).join("");
    const ind = container.querySelector("#ss-pageind");
    if (ind) ind.textContent = `${V.page + 1} / ${pages}`;
  }

  window.scenesViews.subsets = render;
})();
