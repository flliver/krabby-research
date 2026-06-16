/* ==========================================================================
 * Scenes tab — Metadata view (STO-SCN-153).
 *
 * Registers window.scenesViews.meta (the registry the 146 shell calls). Read-
 * only: identity + capture mode + counts + scale/datum status + pipeline
 * state, served from GET /api/scene/<scene>/meta.
 * ========================================================================== */
"use strict";

(function sceneMetaView() {
  window.scenesViews = window.scenesViews || {};

  function pill(on, label) {
    return `<span class="pl ${on ? "on" : "off"}">${on ? "✓" : "·"} ${label}</span>`;
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  async function renderMeta(container, scene) {
    container.innerHTML = `<div class="meta-loading">Loading metadata for ${esc(scene)}…</div>`;
    let m;
    try {
      const r = await fetch(`/api/scene/${encodeURIComponent(scene)}/meta`);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      m = await r.json();
      if (m.error) throw new Error(m.error);
    } catch (e) {
      container.innerHTML = `<div class="meta-err">Failed to load metadata: ${esc(e.message)}</div>`;
      return;
    }

    const c = m.counts || {};
    const d = m.datum || {};
    const st = m.state || {};
    const datumBlock = d.calibrated
      ? `<div class="meta-datum on">
           <div class="big">${esc(d.scale_m_per_unit)} <span>m / unit</span></div>
           <div class="meta-sub">${esc(d.method || "")}</div>
           ${d.status ? `<div class="meta-sub dim">${esc(d.status)}</div>` : ""}
           ${d.scene_extent_m ? `<div class="meta-sub dim">extent: ${esc(d.scene_extent_m)}</div>` : ""}
           ${d.path ? `<div class="meta-sub mono">${esc(d.path)}</div>` : ""}
         </div>`
      : `<div class="meta-datum off"><div class="big">uncalibrated</div>
           <div class="meta-sub dim">no datum.json — run MEASURE + Normalize Units (STO-SCN-152)</div></div>`;

    container.innerHTML = `
      <div class="meta-grid">
        <div class="meta-head">
          <span class="meta-code">${esc(m.code)}</span>
          <span class="meta-name">${esc(m.name)}</span>
          <span class="meta-mode">${esc(m.capture_mode)}</span>
        </div>
        <div class="meta-counts">
          <div class="ct"><b>${esc(c.images)}</b><span>images</span></div>
          <div class="ct"><b>${esc(c.subsets)}</b><span>subsets</span></div>
          <div class="ct"><b>${esc(c.solves)}</b><span>solves</span></div>
          <div class="ct"><b>${esc(c.render_views)}</b><span>render views</span></div>
        </div>
        <div class="meta-section"><h4>Scale / datum</h4>${datumBlock}</div>
        <div class="meta-section"><h4>Pipeline state</h4>
          <div class="meta-pills">
            ${pill(st.ingested, "ingested")}
            ${pill(st.solved, "solved")}
            ${pill(st.scouted, "scouted")}
            ${pill(st.meshed, "meshed")}
            ${pill(st.calibrated, "calibrated")}
          </div>
        </div>
      </div>`;
  }

  window.scenesViews.meta = renderMeta;
})();
