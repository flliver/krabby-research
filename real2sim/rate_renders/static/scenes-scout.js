/* ==========================================================================
 * Scenes tab — Scout view (STO-SCN-151).
 *
 * Registers window.scenesViews.scout. Embeds the verify_viewer scout viewer
 * (DA3 gaussian + posed frustums, the STO-SCN-095/105 surface) for the scene,
 * built on demand via build_verify behind /api/scene/<s>/scout-build and served
 * from /api/scene/<s>/verify/. Plus a Render Views panel (list + author the
 * standard overview view). Read-only 3D; building runs build_verify on a numpy
 * python (a few seconds).
 * ========================================================================== */
"use strict";

(function sceneScoutView() {
  window.scenesViews = window.scenesViews || {};
  let poll = null;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  async function jget(u) { return (await fetch(u)).json(); }
  async function jpost(u, b) {
    return (await fetch(u, { method: "POST", body: JSON.stringify(b || {}) })).json();
  }

  async function render(container, scene) {
    if (poll) { clearInterval(poll); poll = null; }
    container.innerHTML = `<div class="sc-wrap"><div id="sc-main">Loading scout…</div>
      <div class="sc-side">
        <h4>Render views</h4>
        <div id="sc-views" class="sc-views">…</div>
        <button id="sc-add">+ Overview view</button>
        <p class="sc-hint">Named cameras the renderer uses (<code>views/&lt;name&gt;/view.json</code>).
          Overview = a pulled-back look-at over the scene extent.</p>
      </div></div>`;
    drawViews(container, scene);
    container.querySelector("#sc-add").onclick = async () => {
      const b = container.querySelector("#sc-add"); b.disabled = true; b.textContent = "authoring…";
      const r = await jpost(`/api/scene/${encodeURIComponent(scene)}/view-author`);
      b.disabled = false; b.textContent = "+ Overview view";
      if (r.error) alert("author failed: " + r.error); else drawViews(container, scene, r.views);
    };
    refreshMain(container, scene);
  }

  async function drawViews(container, scene, views) {
    const el = container.querySelector("#sc-views");
    if (!views) { try { views = (await jget(`/api/scene/${encodeURIComponent(scene)}/views`)).views; } catch { views = []; } }
    el.innerHTML = (views && views.length)
      ? views.map((v) => `<div class="sc-view">▸ ${esc(v.name)}</div>`).join("")
      : `<div class="sc-empty">none yet</div>`;
  }

  async function refreshMain(container, scene) {
    const main = container.querySelector("#sc-main");
    let st;
    try { st = await jget(`/api/scene/${encodeURIComponent(scene)}/scout-status`); }
    catch (e) { main.innerHTML = `<div class="sc-err">status failed: ${esc(e.message)}</div>`; return; }
    if (st.built) {
      const src = `/api/scene/${encodeURIComponent(scene)}/verify/viewer.html?v=${Date.now()}`;
      main.innerHTML = `<iframe class="sc-frame" src="${src}" title="scout ${esc(scene)}"></iframe>`;
      return;
    }
    if (!st.scout) {
      main.innerHTML = `<div class="sc-build"><p>No scout gaussian yet — run the
        <b>Pipeline</b> (scout phase) first.</p></div>`;
      return;
    }
    main.innerHTML = `<div class="sc-build">
      <p>Scout gaussian found (<code>${esc(st.scout.scout)}</code>). Build the verify surface to view it.</p>
      <button id="sc-build-btn">Build scout view</button>
      <div id="sc-bmsg" class="sc-bmsg"></div></div>`;
    container.querySelector("#sc-build-btn").onclick = async () => {
      container.querySelector("#sc-bmsg").textContent = "building (a few seconds)…";
      const r = await jpost(`/api/scene/${encodeURIComponent(scene)}/scout-build`);
      if (r.error) { container.querySelector("#sc-bmsg").textContent = r.error; return; }
      if (poll) clearInterval(poll);
      poll = setInterval(async () => {
        const s = await jget(`/api/scene/${encodeURIComponent(scene)}/scout-status`);
        const m = container.querySelector("#sc-bmsg");
        if (m) m.textContent = s.phase || s.status || "";
        if (s.built || s.status === "done") { clearInterval(poll); poll = null; refreshMain(container, scene); }
        if (s.status === "error") { clearInterval(poll); poll = null; if (m) m.textContent = "build failed: " + (s.error || ""); }
      }, 1200);
    };
  }

  window.scenesViews.scout = render;
})();
