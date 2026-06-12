/* ==========================================================================
 * Rate-renders — frontend logic.
 * Vanilla JS, no framework. Modules-as-functions; state in a single object.
 * Drag-and-drop uses the HTML5 native API (dragstart / dragover / drop).
 * ========================================================================== */
"use strict";

// ---- State ---------------------------------------------------------------

const state = {
  scene: null,
  view: null,
  scenes: [],
  views: [],
  variants: [],
  manifests: {},
  rendered: {},     // {view: [variant, ...]}  which renders exist
  layout: 1,        // grid cell count (1, 2, 4, 9, 16); 1 = focus mode
  pageIdx: 0,       // for paging through variants when grid < total
  tiers: [],        // [[v, ...], [v, ...], ...]  index 0 = tier 1 (best)
  pool: [],
  focusVariant: null,    // for manifest panel
  rater: localStorage.getItem("rater") || "",
  // Per-view draft rankings — each view keeps its own in-progress tier
  // assignment so switching views doesn't clobber work in progress.
  // Cleared when scene changes.
  drafts: {},       // { viewName: { tiers: [[v, ...], ...], pool: [v, ...] } }
};

// ---- Wire-up: DOM -------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const els = {
  scenePicker: $("#scene-picker"),
  viewPicker: $("#view-picker"),
  raterSelect: $("#rater-select"),
  layoutBtns: $$("#layout-buttons button"),
  prevPage: $("#prev-page"),
  nextPage: $("#next-page"),
  pageIndicator: $("#page-indicator"),
  grid: $("#grid"),
  poolDrop: $("#pool-drop"),
  tiers: $("#tiers"),
  resetBtn: $("#reset-tiers"),
  addTierBtn: $("#add-tier"),
  submitBtn: $("#submit-btn"),
  submitStatus: $("#submit-status"),
  manifest: $("#manifest-content"),
  results: $("#results-content"),
  status: $("#status-msg"),
};

// ---- API ----------------------------------------------------------------

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json();
}

async function loadScenes() {
  state.scenes = await api("/api/scenes");
  els.scenePicker.innerHTML = state.scenes.map(s => `<option>${s}</option>`).join("");
  if (state.scenes.length) {
    state.scene = state.scenes[0];
    els.scenePicker.value = state.scene;
  }
}

async function loadScene() {
  if (!state.scene) return;
  setStatus(`Loading ${state.scene}…`);
  const d = await api(`/api/scene/${state.scene}`);
  state.views = d.views || [];
  state.variants = d.variants || [];
  state.manifests = d.manifests || {};
  state.labels = d.labels || {};   // v4: identity -> human label
  state.missing = d.missing || {};  // v4 (STO-SCN-085): view -> [identities without a render]
  state.taskGaps = d.task_gaps || []; // v4 (STO-SCN-087): graph-level gaps (GPU tier)
  state.rendered = d.rendered || {};
  state.knownRaters = d.raters || [];
  // Pull all submissions so we can fall back to "show your last submission
  // in the tiers" when no in-progress draft exists for a (rater, view).
  try {
    state.submissions = await api(`/api/rankings/${state.scene}`);
  } catch (e) {
    state.submissions = [];
    console.warn("rankings fetch failed:", e);
  }
  rebuildRaterSelect();
  if (state.views.length) {
    if (!state.views.includes(state.view)) state.view = state.views[0];
  }
  els.viewPicker.innerHTML = state.views.map(v => `<option>${v}</option>`).join("");
  els.viewPicker.value = state.view;
  // Restore any persisted per-view drafts for this scene from localStorage,
  // then apply the current view's draft (or start fresh if none).
  loadPersistedDrafts();
  // post-migration hygiene: drop drafts referencing retired variant labels
  for (const [view, draft] of Object.entries(state.drafts || {})) {
    const known = (vs) => vs.every(v => state.variants.includes(v));
    if (draft && draft.tiers && !known(Object.values(draft.tiers).flat())) {
      delete state.drafts[view];
    }
  }
  loadDraftForView(state.view);
  setStatus(`${state.scene} — ${state.variants.length} variants, ${state.views.length} views`);
  if (!state.focusVariant || !state.variants.includes(state.focusVariant)) {
    state.focusVariant = state.variants[0] || null;   // manifest visible immediately
  }
  await refreshAll();
}

async function refreshAll() {
  renderGrid();
  renderTiers();
  renderManifest();
  await refreshResults();
  updatePageIndicator();
  updateSubmitButton();
}

async function refreshResults() {
  if (!state.scene) return;
  try {
    const agg = await api(`/api/aggregate/${state.scene}`);
    renderResults(agg);
  } catch (e) {
    els.results.innerHTML = `<em>Error: ${e.message}</em>`;
  }
}

async function submitRanking() {
  if (!state.rater.trim()) {
    setSubmitStatus("Enter your name first.", "err");
    return;
  }
  if (state.pool.length > 0) {
    setSubmitStatus("Pool not empty — rank everything before submitting.", "err");
    return;
  }
  // Build {variant: rank} from tiers (collapse empty tiers)
  const ranks = {};
  let rankIdx = 1;
  for (const tier of state.tiers) {
    if (tier.length === 0) continue;
    for (const v of tier) ranks[v] = rankIdx;
    rankIdx++;
  }
  const body = JSON.stringify({
    rater: state.rater.trim(),
    view: state.view,
    ranks,
  });
  try {
    const res = await fetch(`/api/rankings/${state.scene}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(`${res.status}: ${t}`);
    }
    const d = await res.json();
    setSubmitStatus(`✓ Submitted at ${(d.row && d.row.submitted_at) || (d.rows && d.rows[0] && d.rows[0].ts) || "now"}`, "ok");
    // Re-pull scene to refresh known-raters list (so new raters appear
    // for everyone the next time they hit the page or change scene).
    if (!state.knownRaters.includes(state.rater)) {
      state.knownRaters.push(state.rater);
      rebuildRaterSelect();
    }
    await refreshResults();
  } catch (e) {
    setSubmitStatus(`Error: ${e.message}`, "err");
  }
}

// ---- State helpers ------------------------------------------------------

function resetTiers() {
  // Default: N tiers (one per variant), all variants in pool.
  const n = Math.max(state.variants.length, 1);
  state.tiers = Array.from({ length: n }, () => []);
  state.pool = [...state.variants];
}

function saveDraftForView(view) {
  // Snapshot tiers/pool into drafts[view] so we can come back to it.
  if (!view) return;
  state.drafts[view] = {
    tiers: state.tiers.map(t => [...t]),
    pool: [...state.pool],
  };
}

function loadDraftForView(view) {
  // Three-tier fallback when restoring tier state on view-change / reload:
  //   1) localStorage draft *with content* (in-progress edits) — highest priority
  //   2) latest submission by current rater on this view — auto-restore so
  //      reloading after submitting doesn't look like the data was lost
  //   3) empty draft if one happens to exist (preserves any "+ Tier" rows
  //      the user added) — fallback when no submission either
  //   4) fresh state (everything in pool) — never-ranked view
  // An EMPTY draft (all variants in pool) does not shadow the submission —
  // it's almost certainly auto-saved transient state, not real work.
  const d = state.drafts[view];
  const draftHasContent = d && Array.isArray(d.tiers) &&
    d.tiers.some(t => Array.isArray(t) && t.length > 0);
  if (draftHasContent) {
    state.tiers = d.tiers.map(t => [...t]);
    state.pool = [...d.pool];
    return;
  }
  const sub = lastSubmissionFor(state.rater, view);
  if (sub) {
    const derived = deriveTiersFromRanks(sub.ranks);
    state.tiers = derived.tiers;
    state.pool = derived.pool;
    return;
  }
  if (d) {
    state.tiers = d.tiers.map(t => [...t]);
    state.pool = [...d.pool];
  } else {
    resetTiers();
  }
}

function lastSubmissionFor(rater, view) {
  if (!rater || !state.submissions || !state.submissions.length) return null;
  const matches = state.submissions.filter(s =>
    s.rater === rater && s.view === view
  );
  if (!matches.length) return null;
  matches.sort((a, b) =>
    String(b.submitted_at || "").localeCompare(String(a.submitted_at || ""))
  );
  return matches[0];
}

function deriveTiersFromRanks(ranks) {
  // ranks is {variant: rankInteger}. Group by rank, sort, fill tier rows.
  // Variants in our state.variants but missing from ranks fall into pool.
  const byRank = {};
  for (const [v, r] of Object.entries(ranks)) {
    const k = Number(r);
    (byRank[k] ||= []).push(v);
  }
  const sortedRanks = Object.keys(byRank).map(Number).sort((a, b) => a - b);
  const tiers = sortedRanks.map(r => byRank[r]);
  // Pad to N tiers (one per variant) for visual stability
  while (tiers.length < state.variants.length) tiers.push([]);
  // Anything in current scene's variants that wasn't in the submitted ranks
  // (e.g. a new variant added after submission) goes back to pool
  const placed = new Set(Object.keys(ranks));
  const pool = state.variants.filter(v => !placed.has(v));
  return { tiers, pool };
}

// localStorage persistence. Key is per-scene so different scenes don't
// trample each other's drafts. Variants that no longer exist (data drift)
// are pruned and re-added to the pool on load.
function draftsKey() {
  return state.scene ? `drafts:${state.scene}` : null;
}

function persistDrafts() {
  // Always snapshot the current view's working state first — drag-drop and
  // reset both call this, so the live working state survives a reload.
  saveDraftForView(state.view);
  const key = draftsKey();
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(state.drafts));
  } catch (e) {
    // Quota errors etc. — non-fatal, just lose persistence.
    console.warn("persistDrafts failed:", e);
  }
}

// One-shot view-name migrations for localStorage drafts. When we rename a
// view server-side (e.g. main_compare_angle → compare_01), drafts saved
// under the old key would otherwise be orphaned. Add entries here when
// renames happen; the migration is a no-op once the new key is populated.
const DRAFT_VIEW_RENAMES = {
  main_compare_angle: "compare_01",
};

function loadPersistedDrafts() {
  state.drafts = {};
  const key = draftsKey();
  if (!key) return;
  let parsed;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return;
    parsed = JSON.parse(raw);
  } catch (e) {
    console.warn("loadPersistedDrafts parse failed:", e);
    return;
  }
  // Migrate orphaned draft keys for renamed views. When both keys exist
  // (because an empty draft auto-saved at the new key before migration ran),
  // keep whichever has actual content — empty pool/tiers means the user
  // never worked there, so it shouldn't shadow the renamed-from work.
  const hasContent = (d) => d && Array.isArray(d.tiers) &&
    d.tiers.some(t => Array.isArray(t) && t.length > 0);
  let migrated = false;
  for (const [oldName, newName] of Object.entries(DRAFT_VIEW_RENAMES)) {
    if (parsed[oldName] && !parsed[newName]) {
      parsed[newName] = parsed[oldName];
      delete parsed[oldName];
      migrated = true;
      console.info(`Migrated draft: ${oldName} → ${newName}`);
    } else if (parsed[oldName] && parsed[newName]) {
      const oldFull = hasContent(parsed[oldName]);
      const newFull = hasContent(parsed[newName]);
      if (oldFull && !newFull) {
        parsed[newName] = parsed[oldName];
        console.info(`Migrated draft (preferred non-empty): ${oldName} → ${newName}`);
      }
      delete parsed[oldName];
      migrated = true;
    }
  }
  if (migrated) {
    try { localStorage.setItem(key, JSON.stringify(parsed)); }
    catch (e) { console.warn("draft-rename persist failed:", e); }
  }
  // Validate: drop variants no longer in the scene; re-pool any new variants
  // that exist in the scene but aren't in the saved draft.
  const valid = new Set(state.variants);
  const cleaned = {};
  for (const view of Object.keys(parsed || {})) {
    const d = parsed[view];
    if (!d || !Array.isArray(d.tiers) || !Array.isArray(d.pool)) continue;
    const tiers = d.tiers.map(t =>
      Array.isArray(t) ? t.filter(v => valid.has(v)) : []
    );
    const pool = d.pool.filter(v => valid.has(v));
    // Any variants in this scene not yet placed go into the pool
    const placed = new Set([...tiers.flat(), ...pool]);
    for (const v of state.variants) {
      if (!placed.has(v)) pool.push(v);
    }
    // Make sure tier count matches variant count (pad if scene grew)
    while (tiers.length < state.variants.length) tiers.push([]);
    cleaned[view] = { tiers, pool };
  }
  state.drafts = cleaned;
}

function findVariant(v) {
  if (state.pool.includes(v)) return { source: "pool", idx: state.pool.indexOf(v) };
  for (let i = 0; i < state.tiers.length; i++) {
    const idx = state.tiers[i].indexOf(v);
    if (idx >= 0) return { source: "tier", tier: i, idx };
  }
  return null;
}

function moveVariant(v, target) {
  // target: 'pool' | integer tier index
  const found = findVariant(v);
  if (!found) return;
  if (found.source === "pool") state.pool.splice(found.idx, 1);
  else state.tiers[found.tier].splice(found.idx, 1);
  if (target === "pool") state.pool.push(v);
  else state.tiers[target].push(v);
}

// ---- Rendering ----------------------------------------------------------

function renderGrid() {
  // Layout: choose cols/rows from the cell count
  // Layouts: [cols, rows]. 1×2 is vertical (one column, two rows) so 16:9
  // renders get full horizontal width — better for side-by-side detail.
  const layoutDims = { 1: [1, 1], 2: [1, 2], 4: [2, 2], 9: [3, 3], 16: [4, 4] };
  const [cols, rows] = layoutDims[state.layout] || [2, 2];
  els.grid.style.setProperty("--grid-cols", cols);
  els.grid.style.setProperty("--grid-rows", rows);

  // Which variants this page shows
  const cellCount = state.layout;
  let slice;
  if (cellCount === 1) {
    // Single-cell mode: show whatever the user last focused (clicking a
    // ranking-area card, or arrow-keying). Falls back to first variant.
    const v = state.focusVariant || state.variants[0];
    slice = v ? [v] : [];
  } else {
    const start = state.pageIdx * cellCount;
    slice = state.variants.slice(start, start + cellCount);
  }

  els.grid.innerHTML = "";
  for (let i = 0; i < cellCount; i++) {
    const v = slice[i];
    const tile = document.createElement("div");
    tile.className = "tile";
    if (!v) {
      tile.classList.add("empty");
    } else {
      const renderedHere = (state.rendered[state.view] || []).includes(v);
      const src = renderedHere
        ? `/api/render/${state.scene}/${state.view}/${v}.png`
        : null;
      if (src) {
        const img = document.createElement("img");
        img.src = src;
        img.alt = v;
        img.loading = "eager";
        // Disable native image drag — let the tile drive the dragstart so
        // the dataTransfer payload is the variant name, not an image URL.
        img.draggable = false;
        tile.appendChild(img);
      } else {
        tile.innerHTML = `<em style="color: var(--text-dim);">no render: ${v}</em>`;
      }
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = labelOf(v);
      tile.appendChild(label);

      // The big tile is draggable too — same payload shape as the small
      // cards, so the existing tier-drop handler routes it correctly.
      tile.draggable = true;
      tile.dataset.variant = v;
      tile.addEventListener("dragstart", (e) => {
        tile.classList.add("dragging");
        e.dataTransfer.setData("text/plain", v);
        e.dataTransfer.effectAllowed = "move";
      });
      tile.addEventListener("dragend", () => tile.classList.remove("dragging"));
      tile.addEventListener("mouseenter", () => setFocusVariant(v));
      tile.addEventListener("click", () => setFocusVariant(v));
    }
    els.grid.appendChild(tile);
  }

  renderMissingPool();
}

// ---- STO-SCN-085/086/087: missing pool (small fixed chips) ---------------
// Operator (2026-06-11): gap buttons must consume a FIXED small space —
// a "pool" of missing photos; hover/click shows details in the manifest
// area on the left.

function renderMissingPool() {
  const pool = document.getElementById("missing-pool");
  if (!pool) return;
  pool.innerHTML = "";
  const busy = state.materializing;
  const chip = (color, icon, title) => {
    const c = document.createElement("div");
    c.style.cssText = `width:96px; min-width:96px; height:56px; min-height:56px; flex:0 0 auto;` +
      ` border:1.5px dashed ${color}; background:rgba(127,127,127,0.08);` +
      ` border-radius:6px; display:flex; flex-direction:column; align-items:center;` +
      ` justify-content:center; cursor:pointer; font-size:11px; line-height:1.3;` +
      ` color:var(--text-dim); overflow:hidden; text-align:center;`;
    c.innerHTML = `<div>${busy && color !== "#7a5af8" ? "⏳" : icon}</div>` +
                  `<div style="white-space:nowrap; max-width:80px; overflow:hidden; text-overflow:ellipsis;">${title}</div>`;
    return c;
  };
  for (const mv of (state.missing && state.missing[state.view]) || []) {
    const c = chip("var(--text-dim)", "⬚", labelOf(mv));
    c.addEventListener("mouseenter", () => showGapDetail({type: "render", id: mv}));
    c.addEventListener("click", () => { showGapDetail({type: "render", id: mv}); if (!busy) materializeScene(); });
    pool.appendChild(c);
  }
  for (const g of (state.taskGaps || [])) {
    const c = chip("#7a5af8", "🧬", g.label);
    c.addEventListener("mouseenter", () => showGapDetail({type: "task", gap: g}));
    c.addEventListener("click", () => showGapDetail({type: "task", gap: g}));
    pool.appendChild(c);
  }
}

function showGapDetail(d) {
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  let html;
  if (d.type === "render") {
    const m = state.manifests[d.id] || {};
    html = `<div><strong>MISSING RENDER</strong></div>` +
      `<div style="margin-top:4px;">${esc(labelOf(d.id))} <span style="color:var(--text-dim); font-size:0.85em;">${esc(d.id)}</span></div>` +
      `<div style="margin-top:4px;">view ${esc(state.view)} has no render for this mesh yet.</div>` +
      (m.mesh && m.mesh.verts ? `<div style="margin-top:4px; color:var(--text-dim);">mesh: ${(m.mesh.verts/1e6).toFixed(1)}M verts · ${m.mesh.size_mb} MB</div>` : "") +
      `<div style="margin-top:6px;"><em>${state.materializing ? "materializing — it will appear when done" : "click the chip to materialize (local Blender render)"}</em></div>`;
  } else {
    const g = d.gap;
    html = `<div><strong>GRAPH GAP — never run</strong></div>` +
      `<div style="margin-top:4px;">${esc(g.task)}</div>` +
      `<div>${esc(g.label)}</div>` +
      (g.settings ? `<div style="margin-top:4px; color:var(--text-dim);">defaults: ${esc(JSON.stringify(g.settings))}</div>` : "") +
      ((g.license_flags || []).length ? `<div style="margin-top:4px; color:#e06c5a;">⚠ NC — evaluation only: ${esc(g.license_flags.join("; "))}</div>` : "") +
      `<div style="margin-top:6px;"><em>GPU job — needs the v4 executor (STO-SCN-088) + your host choice. Not clickable yet, honestly.</em></div>`;
  }
  els.manifest.innerHTML = html;
}

// ---- STO-SCN-086: click-to-materialize ----------------------------------

async function materializeScene() {
  try {
    const d = await api(`/api/materialize/${state.scene}`, { method: "POST" });
    state.materializing = true;
    renderGrid();
    setStatus(d.already_running
      ? "materialize already running — joining its progress"
      : `materializing missing renders for ${state.scene}…`);
    pollMaterialize();
  } catch (e) {
    setStatus(`materialize failed to start: ${e}`);
  }
}

async function pollMaterialize() {
  if (state._matPoll) clearTimeout(state._matPoll);
  try {
    const s = await api(`/api/materialize/${state.scene}`);
    if (s.running) {
      // refresh payload mid-flight so finished renders flip in early
      await loadScene();
      state.materializing = true;
      renderGrid();
      state._matPoll = setTimeout(pollMaterialize, 8000);
      return;
    }
    state.materializing = false;
    await loadScene();
    const o = (s.last && s.last.outcome) || {};
    setStatus(`materialize done: rendered ${o.rendered ?? "?"}, NOOP ${o.noop ?? "?"}, failed ${o.failed ?? "?"}`);
  } catch (e) {
    state.materializing = false;
    setStatus(`materialize poll failed: ${e}`);
  }
}

function renderTiers() {
  // Pool drop zone
  els.poolDrop.innerHTML = "";
  for (const v of state.pool) els.poolDrop.appendChild(makeCard(v));

  // Tier rows
  els.tiers.innerHTML = "";
  state.tiers.forEach((tierItems, idx) => {
    const tierNum = idx + 1;
    const row = document.createElement("div");
    row.className = "tier-row";
    row.dataset.tier = tierNum;

    const label = document.createElement("div");
    label.className = "tier-label";
    label.innerHTML = `${tierNum}<span class="tier-sub">${tierLabelSub(tierNum)}</span>`;

    const drop = document.createElement("div");
    drop.className = "tier-drop";
    drop.dataset.tier = String(idx);
    for (const v of tierItems) drop.appendChild(makeCard(v));
    wireDropZone(drop, idx);

    row.appendChild(label);
    row.appendChild(drop);
    els.tiers.appendChild(row);
  });
  wireDropZone(els.poolDrop, "pool");
}

function tierLabelSub(n) {
  if (n === 1) return "best";
  if (n === state.tiers.length) return "worst";
  return "";
}

function makeCard(v) {
  const renderedHere = (state.rendered[state.view] || []).includes(v);
  const card = document.createElement("div");
  card.className = "card";
  card.draggable = true;
  card.dataset.variant = v;

  if (renderedHere) {
    const img = document.createElement("img");
    img.src = `/api/render/${state.scene}/${state.view}/${v}.png`;
    img.alt = v;
    card.appendChild(img);
  } else {
    card.style.background = "var(--bg-elev-2)";
  }
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = labelOf(v);
  card.appendChild(name);

  card.addEventListener("dragstart", (e) => {
    card.classList.add("dragging");
    e.dataTransfer.setData("text/plain", v);
    e.dataTransfer.effectAllowed = "move";
  });
  card.addEventListener("dragend", () => card.classList.remove("dragging"));
  card.addEventListener("mouseenter", () => setFocusVariant(v));
  card.addEventListener("click", () => setFocusVariant(v));
  return card;
}

function wireDropZone(el, target) {
  el.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.classList.add("dragover");
    e.dataTransfer.dropEffect = "move";
  });
  el.addEventListener("dragleave", (e) => {
    if (!el.contains(e.relatedTarget)) el.classList.remove("dragover");
  });
  el.addEventListener("drop", (e) => {
    e.preventDefault();
    el.classList.remove("dragover");
    const v = e.dataTransfer.getData("text/plain");
    if (!v) return;
    const tgt = target === "pool" ? "pool" : Number(target);
    moveVariant(v, tgt);
    renderTiers();
    updateSubmitButton();
    persistDrafts();   // survive reload
  });
}

function labelOf(v) {
  return (state.labels && state.labels[v]) || v;
}

function setFocusVariant(v) {
  state.focusVariant = v;
  renderManifest();
  // In single-cell layout, the grid always shows the focused variant —
  // re-render so a click on a ranking card immediately swaps the big image.
  if (state.layout === 1) {
    renderGrid();
    updatePageIndicator();
  }
  // Visual focus on grid tiles + cards
  $$(".tile.focus, .card.focus").forEach(el => el.classList.remove("focus"));
  $$(`.card[data-variant="${cssEscape(v)}"]`).forEach(el => el.classList.add("focus"));
  $$(".tile .label").forEach(label => {
    if (label.textContent === labelOf(v)) label.parentElement.classList.add("focus");
  });
}

function cssEscape(s) {
  return s.replace(/"/g, '\\"');
}

function renderManifest() {
  // SETTINGS-FIRST (STO-SCN-045): each variant is one run = one
  // parameterization of its transforms. The runoff compares those
  // settings, so the panel renders every transform's specification
  // parameters generically (no hardcoded per-pipeline fields), plus
  // the measured stats from results.json.
  if (!state.focusVariant) {
    els.manifest.innerHTML = "<em>Hover a card to see settings.</em>";
    return;
  }
  const m = state.manifests[state.focusVariant] || {};
  const fmt = (x) => (x === null || x === undefined || x === "") ? "—"
    : (Array.isArray(x) ? x.join(", ") : (typeof x === "object" ? JSON.stringify(x) : x));
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  let html = `<div><strong>${esc(labelOf(state.focusVariant))}</strong>` +
    (labelOf(state.focusVariant) !== state.focusVariant
      ? ` <span style="color: var(--text-dim); font-size: 0.85em;">${esc(state.focusVariant)}</span>` : "") +
    `</div>`;
  const ms = m.mesh || {};
  if (ms.verts) {
    const fmtN = (n) => n >= 1e6 ? (n/1e6).toFixed(1) + "M" : n >= 1e3 ? (n/1e3).toFixed(0) + "k" : n;
    html += `<div style="margin-top:6px; color: var(--text-dim);">mesh: ` +
            `${fmtN(ms.verts)} verts · ${fmtN(ms.faces)} tris · ${ms.size_mb} MB</div>`;
  }
  const transforms = m.transforms || {};
  const tNames = Object.keys(transforms);
  if (!tNames.length) {
    html += `<div style="margin-top:6px;"><em>No transform specifications captured.</em></div>`;
  }
  for (const tName of tNames) {
    const t = transforms[tName] || {};
    html += `<div style="margin-top:8px;"><strong>${esc(tName)}</strong>` +
            (t.kind ? ` <span style="color: var(--text-dim);">(${esc(t.kind)})</span>` : "") +
            `</div>`;
    const params = t.parameters || {};
    // Settings only: drop nulls/empties and long provenance blobs
    // (tool_args_raw etc.) — those are reproducibility records, not
    // the comparison axes a rater weighs.
    const keys = Object.keys(params).filter(k => {
      const v = params[k];
      if (v === null || v === "") return false;
      if (typeof v === "string" && v.length > 100) return false;
      return true;
    });
    if (keys.length) {
      html += `<div style="margin-top:4px;">` + keys.map(k =>
        `<div>${esc(k)}: <code>${esc(fmt(params[k]))}</code></div>`
      ).join("") + `</div>`;
    } else {
      html += `<div style="margin-top:4px;"><em>no parameters recorded</em></div>`;
    }
    const meas = t.measured || {};
    if (Object.keys(meas).length) {
      html += `<div style="margin-top:4px; color: var(--text-dim);">` +
        `${fmt(meas.status)} · ${fmt(meas.host)} · ${fmt(meas.duration_s)}s` +
        (meas.peak_vram_mib ? ` · peak ${fmt(meas.peak_vram_mib)} MiB` : "") +
        (meas.provenance ? ` · ${esc(meas.provenance)}` : "") +
        `</div>`;
    }
  }
  if (m.notes) html += `<div style="margin-top:8px;"><em>${esc(m.notes)}</em></div>`;
  els.manifest.innerHTML = html;
}

function renderResults(agg) {
  if (!agg || !agg.n_submissions) {
    els.results.innerHTML = "<em>No rankings submitted yet.</em>";
    return;
  }
  let html = `<div style="color: var(--text-dim);">${agg.n_submissions} submission${agg.n_submissions === 1 ? "" : "s"} total</div>`;
  if (agg.overall && agg.overall.length) {
    html += "<h4>Overall</h4><ol>";
    agg.overall.forEach((row, i) => {
      const cls = i === 0 ? "winner" : "";
      html += `<li class="${cls}">${row.variant} <span style="color:var(--text-dim);">— ${row.score.toFixed(3)}</span></li>`;
    });
    html += "</ol>";
  }
  for (const view of Object.keys(agg.per_view).sort()) {
    const v = agg.per_view[view];
    html += `<h4>${view} <span style="color:var(--text-dim); font-weight:400;">(${v.n_submissions})</span></h4><ol>`;
    v.leaderboard.forEach((row, i) => {
      const cls = i === 0 ? "winner" : "";
      html += `<li class="${cls}">${row.variant} <span style="color:var(--text-dim);">— ${row.score.toFixed(2)}</span></li>`;
    });
    html += "</ol>";
  }
  els.results.innerHTML = html;
}

// ---- Rater select ------------------------------------------------------

const NEW_RATER_SENTINEL = "__new__";

function rebuildRaterSelect() {
  // Combine raters from server (people who've submitted on this scene) with
  // the locally-stored last-used name (in case it's a new rater who hasn't
  // submitted yet) and any locally-cached list of names. Sort, dedupe.
  const local = JSON.parse(localStorage.getItem("rater-list") || "[]");
  const all = Array.from(new Set([
    ...(state.knownRaters || []),
    ...local,
    state.rater,
  ].filter(s => s && s.trim()))).sort((a, b) =>
    a.localeCompare(b, undefined, { sensitivity: "base" })
  );
  // If state.rater is empty but we know about raters, default-select the
  // first one. Otherwise the dropdown shows a name visually but state.rater
  // stays empty → submit is disabled and tier state can't restore.
  if (!state.rater && all.length) {
    state.rater = all[0];
    localStorage.setItem("rater", state.rater);
  }
  let html = "";
  if (!all.length) {
    html += `<option value="" disabled selected>(none yet)</option>`;
  } else {
    html += all.map(n =>
      `<option value="${escapeHtml(n)}"${n === state.rater ? " selected" : ""}>${escapeHtml(n)}</option>`
    ).join("");
  }
  html += `<option value="${NEW_RATER_SENTINEL}">+ New rater…</option>`;
  els.raterSelect.innerHTML = html;
  if (state.rater) els.raterSelect.value = state.rater;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function handleRaterSelect() {
  const v = els.raterSelect.value;
  if (v === NEW_RATER_SENTINEL) {
    const name = (prompt("New rater name:") || "").trim();
    if (!name) {
      // User cancelled — restore previous selection
      els.raterSelect.value = state.rater || "";
      return;
    }
    // Cache locally so it's selectable next time even before any submission
    const local = JSON.parse(localStorage.getItem("rater-list") || "[]");
    if (!local.includes(name)) {
      local.push(name);
      localStorage.setItem("rater-list", JSON.stringify(local));
    }
    state.rater = name;
    localStorage.setItem("rater", name);
    rebuildRaterSelect();
  } else {
    state.rater = v;
    localStorage.setItem("rater", v);
  }
  // Switching rater changes which submission backs the current view's tier
  // state — reload the draft (which falls back to this rater's last
  // submission if no in-progress draft exists).
  loadDraftForView(state.view);
  renderTiers();
  updateSubmitButton();
}

// ---- UI helpers ---------------------------------------------------------

function setStatus(msg) { els.status.textContent = msg; }
function setSubmitStatus(msg, cls) {
  els.submitStatus.textContent = msg;
  els.submitStatus.className = cls || "";
}

function updatePageIndicator() {
  if (state.layout === 1) {
    // "1 / 4" reflects which variant of N is showing; wrap-around so
    // never disabled.
    if (state.variants.length === 0) {
      els.pageIndicator.textContent = "0 / 0";
    } else {
      const cur = Math.max(0, state.variants.indexOf(state.focusVariant));
      els.pageIndicator.textContent = `${cur + 1} / ${state.variants.length}`;
    }
    els.prevPage.disabled = false;
    els.nextPage.disabled = false;
    return;
  }
  const pages = Math.max(1, Math.ceil(state.variants.length / state.layout));
  els.pageIndicator.textContent = `${state.pageIdx + 1} / ${pages}`;
  els.prevPage.disabled = state.pageIdx === 0;
  els.nextPage.disabled = state.pageIdx + 1 >= pages;
}

function updateSubmitButton() {
  els.submitBtn.disabled = state.pool.length > 0 || !state.rater.trim();
}

// ---- Event wiring -------------------------------------------------------

function setLayout(n) {
  state.layout = Number(n);
  state.pageIdx = 0;
  els.layoutBtns.forEach(b => b.classList.toggle("active", Number(b.dataset.layout) === state.layout));
  renderGrid();
  updatePageIndicator();
}

function nextPage(delta) {
  if (state.layout === 1) {
    // In single-cell mode "page" cycles through variants with wrap-around,
    // matching the user's mental model of fast back-and-forth A/B viewing.
    if (state.variants.length === 0) return;
    const cur = state.variants.indexOf(state.focusVariant);
    const idx = (cur < 0 ? 0 : cur + delta + state.variants.length) % state.variants.length;
    setFocusVariant(state.variants[idx]);
    return;
  }
  const pages = Math.max(1, Math.ceil(state.variants.length / state.layout));
  state.pageIdx = Math.max(0, Math.min(pages - 1, state.pageIdx + delta));
  renderGrid();
  updatePageIndicator();
}

function addTier() {
  state.tiers.push([]);
  renderTiers();
}

document.addEventListener("DOMContentLoaded", async () => {
  els.raterSelect.addEventListener("change", handleRaterSelect);
  els.scenePicker.addEventListener("change", () => {
    state.scene = els.scenePicker.value;
    state.pageIdx = 0;
    // Drafts are scoped per-scene in localStorage; loadScene → loadPersistedDrafts
    // will pull THIS scene's drafts. Don't touch other scenes' drafts.
    loadScene();
  });
  els.viewPicker.addEventListener("change", () => {
    // Save current view's draft, switch, then restore new view's draft
    // (or start fresh). Persist on every view change so a reload mid-rank
    // doesn't lose the in-progress state for either view.
    saveDraftForView(state.view);
    persistDrafts();
    state.view = els.viewPicker.value;
    loadDraftForView(state.view);
    refreshAll();
  });
  els.layoutBtns.forEach(b => {
    b.addEventListener("click", () => setLayout(b.dataset.layout));
  });
  els.prevPage.addEventListener("click", () => nextPage(-1));
  els.nextPage.addEventListener("click", () => nextPage(1));
  els.resetBtn.addEventListener("click", () => {
    resetTiers();
    renderTiers();
    updateSubmitButton();
    persistDrafts();
  });
  els.addTierBtn.addEventListener("click", () => {
    addTier();
    persistDrafts();
  });
  els.submitBtn.addEventListener("click", submitRanking);

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Ignore when typing in an input
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowRight") { nextPage(1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { nextPage(-1); e.preventDefault(); }
    else if (e.key === "r") { resetTiers(); renderTiers(); updateSubmitButton(); persistDrafts(); }
    else if (e.key === "Enter") submitRanking();
    else if (e.key === "1") setLayout(1);
    else if (e.key === "2") setLayout(2);
    else if (e.key === "4") setLayout(4);
    else if (e.key === "9") setLayout(9);
    else if (e.key === "6") setLayout(16);  // "6" for 4×4 since '16' is two keys
  });

  await loadScenes();
  if (state.scene) await loadScene();
});
