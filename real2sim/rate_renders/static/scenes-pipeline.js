/* ==========================================================================
 * Scenes tab — Pipeline view (STO-SCN-150).
 *
 * Registers window.scenesViews.pipeline. Pick a GPU host, preview the command
 * plan (dry-run), or launch the ingest-scene pipeline
 * (precull → solve → covis → scout → reconstruct-da3). Live phase progress +
 * log tail polled from /api/scene/<scene>/pipeline-status.
 *
 * NOTE: a real run executes v4exec on a GPU host (ssh + docker) — heavy, and
 * operator-driven. Dry-run is the safe pre-flight.
 * ========================================================================== */
"use strict";

(function scenePipelineView() {
  window.scenesViews = window.scenesViews || {};
  let pollTimer = null;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  async function jget(u) { const r = await fetch(u); return r.json(); }
  async function jpost(u, b) {
    const r = await fetch(u, { method: "POST", body: JSON.stringify(b) });
    return r.json();
  }

  const BADGE = { pending: "·", running: "▶", done: "✓", planned: "◌", error: "✕", skipped: "⊘" };

  async function render(container, scene) {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    let hosts = ["tbeeprz"];
    try { hosts = (await jget("/api/hosts")).hosts || hosts; } catch { /* default */ }
    container.innerHTML = `
      <div class="pl-wrap">
        <div class="pl-bar">
          <label>Host <select id="pl-host">${hosts.map((h) => `<option>${esc(h)}</option>`).join("")}</select></label>
          <button id="pl-dry">Preview plan</button>
          <button id="pl-run">Run Preprocessors</button>
          <span id="pl-msg" class="pl-msg"></span>
        </div>
        <div id="pl-phases" class="pl-phases"></div>
        <pre id="pl-log" class="pl-log"></pre>
        <p class="pl-hint">precull → solve → covis → DA3 gaussian → DA3 mesh · runs <code>v4exec</code>
          on the host (ssh + GPU + docker). Phases are idempotent (re-run resumes). Preview first.</p>
      </div>`;
    const host = () => container.querySelector("#pl-host").value;
    const msg = (t, c) => { const m = container.querySelector("#pl-msg"); m.textContent = t; m.className = "pl-msg " + (c || ""); };

    container.querySelector("#pl-dry").onclick = async () => {
      msg("building plan…");
      const r = await jpost(`/api/scene/${encodeURIComponent(scene)}/pipeline`, { host: host(), dry_run: true });
      if (r.error) return msg(r.error, "err");
      startPoll(container, scene, "plan previewed (dry-run)");
    };
    container.querySelector("#pl-run").onclick = async () => {
      if (!confirm(`Run the preprocessors on ${host()}? This executes GPU work (ssh+docker) and can take a while.`)) return;
      msg("launching…");
      const r = await jpost(`/api/scene/${encodeURIComponent(scene)}/pipeline`, { host: host(), dry_run: false });
      if (r.error) return msg(r.error, "err");
      startPoll(container, scene, "running…");
    };

    // show any existing status on load
    drawStatus(container, await jget(`/api/scene/${encodeURIComponent(scene)}/pipeline-status`));
  }

  function startPoll(container, scene, note) {
    if (pollTimer) clearInterval(pollTimer);
    container.querySelector("#pl-msg").textContent = note || "";
    pollTimer = setInterval(async () => {
      const st = await jget(`/api/scene/${encodeURIComponent(scene)}/pipeline-status`);
      drawStatus(container, st);
      if (st.status === "done" || st.status === "error") { clearInterval(pollTimer); pollTimer = null; }
    }, 1000);
  }

  function drawStatus(container, st) {
    const ph = container.querySelector("#pl-phases");
    const log = container.querySelector("#pl-log");
    if (!st || st.status === "none" || !st.phases) {
      ph.innerHTML = `<div class="pl-empty">No pipeline run yet. Preview the plan, then Run.</div>`;
      log.textContent = ""; return;
    }
    const msg = container.querySelector("#pl-msg");
    if (msg) msg.className = "pl-msg " + (st.status === "error" ? "err" : st.status === "done" ? "ok" : "");
    if (msg && (st.status === "done" || st.status === "error"))
      msg.textContent = st.dry_run ? "plan ready" : (st.status === "done" ? "pipeline complete" : "failed — see log");
    ph.innerHTML = st.phases.map((p) => {
      // ingest (local) carries a human note + a non-v4exec cmd; host phases are
      // [python, v4exec.py, <verb>, …] so slice(2) trims the interpreter noise.
      const detail = p.note ? esc(p.note)
        : (p.cmd ? esc((p.key === "ingest" ? p.cmd : p.cmd.slice(2)).join(" ")) : "");
      return `
      <div class="pl-phase ${p.status}">
        <span class="pl-badge">${BADGE[p.status] || "·"}</span>
        <span class="pl-label">${esc(p.label || p.key)}</span>
        ${detail ? `<code class="pl-cmd">${detail}</code>` : ""}
        ${p.rc != null ? `<span class="pl-rc">rc ${p.rc}</span>` : ""}
      </div>`;
    }).join("");
    log.textContent = st.log_tail || "";
  }

  window.scenesViews.pipeline = render;
})();
