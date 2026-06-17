/* ==========================================================================
 * Scenes tab — Measure + Normalize Units view (STO-SCN-152).
 *
 * Registers window.scenesViews.measure. Embeds the verify_viewer MEASURE tool
 * (match.html, STO-SCN-144 two-view triangulation) for the scene, and wires a
 * Normalize Units action: the match.html export → /api/scene/<s>/normalize →
 * recompute(scale) → build_datum → apply_to_gauge → datum.json.
 *
 * Reuses the shipped MEASURE/metric_scale/calibrate_datum/datum_frame backend
 * (001-patio calibrated at s=4.45). Writing the datum is operator-driven
 * (T-020); Dry-run computes the scale without writing.
 * ========================================================================== */
"use strict";

(function sceneMeasureView() {
  window.scenesViews = window.scenesViews || {};

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  async function jget(u) { return (await fetch(u)).json(); }
  async function jpost(u, b) {
    return (await fetch(u, { method: "POST", body: JSON.stringify(b || {}) })).json();
  }

  async function render(container, scene) {
    let st;
    try { st = await jget(`/api/scene/${encodeURIComponent(scene)}/scout-status`); }
    catch { st = {}; }
    if (!st.built) {
      container.innerHTML = `<div class="ms-build"><p>The MEASURE tool needs the verify
        surface. Open the <b>Scout</b> tab and <b>Build scout view</b> first.</p></div>`;
      return;
    }
    const src = `/api/scene/${encodeURIComponent(scene)}/verify/match.html?v=${Date.now()}`;
    container.innerHTML = `
      <div class="ms-wrap">
        <iframe class="ms-frame" src="${src}" title="MEASURE ${esc(scene)}"></iframe>
        <div class="ms-side">
          <h4>Normalize units</h4>
          <ol class="ms-steps">
            <li>In the viewer: <b>M</b> to measure; <b>[ ]</b> photo nav.</li>
            <li>Click P1 in ≥1 photos → <b>E</b>; click P2 → <b>E</b>.</li>
            <li>Enter the P1:P2 distance (meters); <b>export</b>.</li>
            <li>Paste the export below → <b>Normalize Units</b>.</li>
          </ol>
          <textarea id="ms-export" placeholder="paste the match.html MEASURE export JSON"></textarea>
          <label class="ms-dry"><input type="checkbox" id="ms-dry" checked> Dry-run (compute, don't write)</label>
          <button id="ms-go">Normalize Units</button>
          <div id="ms-result" class="ms-result"></div>
        </div>
      </div>`;
    container.querySelector("#ms-go").onclick = () => normalize(container, scene);
  }

  async function normalize(container, scene) {
    const res = container.querySelector("#ms-result");
    let exp;
    try { exp = JSON.parse(container.querySelector("#ms-export").value); }
    catch { res.innerHTML = `<span class="err">export is not valid JSON</span>`; return; }
    const dry = container.querySelector("#ms-dry").checked;
    res.textContent = "computing…";
    const go = container.querySelector("#ms-go"); go.disabled = true;
    let r;
    try {
      r = await jpost(`/api/scene/${encodeURIComponent(scene)}/normalize`, { export: exp, dry });
    } catch (e) { res.innerHTML = `<span class="err">${esc(e.message)}</span>`; go.disabled = false; return; }
    go.disabled = false;
    if (r.error && r.exists) {
      res.innerHTML = `<span class="err">datum.json already exists.</span>
        <button id="ms-force">Overwrite</button>`;
      container.querySelector("#ms-force").onclick = async () => {
        const f = await jpost(`/api/scene/${encodeURIComponent(scene)}/normalize`, { export: exp, dry: false, force: true });
        drawResult(res, f);
      };
      return;
    }
    drawResult(res, r);
  }

  function drawResult(res, r) {
    if (r.error) { res.innerHTML = `<span class="err">${esc(r.error)}</span>`; return; }
    const flags = [];
    if (r.anisotropy) flags.push("⚠ anisotropy (axes disagree >1.5× — distortion?)");
    if (r.weak_triangulation) flags.push("⚠ weak triangulation (near-parallel rays)");
    if (r.da3_scouts_disagree) flags.push("⚠ DA3 scouts disagree (prior unreliable)");
    res.innerHTML =
      `<div class="ms-scale ${r.dry ? "" : "ok"}"><b>${(+r.scale).toFixed(4)}</b> m / unit</div>` +
      `<div class="ms-sub">${esc(r.n_distances)} distance(s) · spread ${(+r.spread).toFixed(3)}` +
      (r.dry ? ` · <i>dry-run (not written)</i>` : (r.datum_json ? ` · <span class="ok">datum.json written</span>` : "")) + `</div>` +
      (flags.length ? `<div class="ms-flags">${flags.map((f) => `<div>${esc(f)}</div>`).join("")}</div>` : "");
  }

  window.scenesViews.measure = render;
})();
