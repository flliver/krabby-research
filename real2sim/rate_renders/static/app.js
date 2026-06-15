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
  profiles: [],     // STO-SCN-108: server-side rater identities (origin-independent)
  // Per-view draft rankings — each view keeps its own in-progress tier
  // assignment so switching views doesn't clobber work in progress.
  // Cleared when scene changes.
  drafts: {},       // { viewName: { tiers: [[v, ...], ...], pool: [v, ...] } }
};

// ---- Wire-up: DOM -------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const els = {
  sceneStrip: $("#scene-strip"),
  sceneLeft: $("#scene-left"),
  sceneRight: $("#scene-right"),
  viewCard: $("#view-card"),                 // STO-SCN-110: View selector card (replaced #view-picker)
  viewTitle: $("#view-title"),               // STO-SCN-110: "View X of Y" large title
  raterSelect: $("#rater-select"),
  layoutBtns: $$("#layout-buttons button"),
  prevPage: $("#prev-page"),
  nextPage: $("#next-page"),
  pageIndicator: $("#page-indicator"),
  grid: $("#grid"),
  poolDrop: $("#pool-drop"),
  tiers: $("#tiers"),
  resetBtn: $("#reset-tiers"),
  submitBtn: $("#submit-btn"),
  submitStatus: $("#submit-status"),
  manifest: $("#manifest-content"),
  copyManifestBtn: $("#copy-manifest"),      // STO-SCN-110: copy manifest as Markdown
  copyLinkBtn: $("#copy-link"),              // STO-SCN-110/111: copy deep-link
  results: $("#results-content"),
  status: $("#status-msg"),
};

// ---- API ----------------------------------------------------------------

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json();
}

// STO-SCN-111: deep-link — ?scene=&view=&variant= navigates to a selected rendering on load.
const _dl = new URLSearchParams(location.search);
const deepLink = { scene: _dl.get("scene"), view: _dl.get("view"), variant: _dl.get("variant") };

async function loadScenes() {
  // [{name, thumb}] — representative image = #1-ranked variant's render
  state.scenes = await api("/api/scenes");
  renderSceneStrip();
  if (state.scenes.length && !state.scene) {
    state.scene = (deepLink.scene && state.scenes.some(s => s.name === deepLink.scene))
      ? deepLink.scene : state.scenes[0].name;        // STO-SCN-111
  }
  highlightSceneCard();
}

function renderSceneStrip() {
  els.sceneStrip.innerHTML = state.scenes.map(sc => `
    <div class="scene-card" data-scene="${sc.name}" title="${sc.name}">
      ${sc.thumb ? `<img src="${sc.thumb}" loading="lazy" alt="${sc.name}">`
                 : `<div class="noimg">&#9633;</div>`}
      <div class="nm">${sc.name}</div>
    </div>`).join("");
  els.sceneStrip.querySelectorAll(".scene-card").forEach(card => {
    card.addEventListener("click", () => {
      state.scene = card.dataset.scene;
      highlightSceneCard();
      state.view = null;
      state.pageIdx = 0;
      // Drafts are per-scene in localStorage; loadScene restores them.
      loadScene();
    });
  });
}

function highlightSceneCard() {
  els.sceneStrip.querySelectorAll(".scene-card").forEach(c =>
    c.classList.toggle("selected", c.dataset.scene === state.scene));
  const sel = els.sceneStrip.querySelector(".scene-card.selected");
  if (sel) sel.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
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
    if (deepLink.view && state.views.includes(deepLink.view)) state.view = deepLink.view;  // STO-SCN-111
    else if (!state.views.includes(state.view)) state.view = state.views[0];
  }
  renderViewCard();                          // STO-SCN-110: card selector instead of a dropdown
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
  await loadAggregate();   // STO-SCN-110: leaderboard ready before we pick the focus
  if (deepLink.variant && state.variants.includes(deepLink.variant)) {   // STO-SCN-111
    state.focusVariant = deepLink.variant;
  } else {
    state.focusVariant = topRankedVariant(state.view);   // STO-SCN-110: auto-show highest-ranked
  }
  deepLink.scene = deepLink.view = deepLink.variant = null;   // one-shot: applied on first load
  await refreshAll();
}

async function refreshAll() {
  renderViewCard();      // STO-SCN-110: keep the View card in sync (renders may have loaded)
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
    state.agg = await api(`/api/aggregate/${state.scene}`);
    renderResults(state.agg);
  } catch (e) {
    els.results.innerHTML = `<em>Error: ${e.message}</em>`;
  }
}

// STO-SCN-110: cache the leaderboard so we can auto-focus the top-ranked render.
async function loadAggregate() {
  if (!state.scene) { state.agg = null; return; }
  try { state.agg = await api(`/api/aggregate/${state.scene}`); }
  catch (e) { state.agg = null; }
}

// STO-SCN-110: the highest-ranking variant for a view — per-view leaderboard first, then
// overall, restricted to variants that actually have a render in that view (so the focus
// tile shows an image); falls back to the first rendered / first variant.
function topRankedVariant(view) {
  const rendered = state.rendered[view] || [];
  const ok = (v) => state.variants.includes(v) && rendered.includes(v);
  const agg = state.agg;
  if (agg) {
    const pv = agg.per_view && agg.per_view[view];
    const fromPv = pv && pv.leaderboard && pv.leaderboard.find(r => ok(r.variant));
    if (fromPv) return fromPv.variant;
    const fromOverall = agg.overall && agg.overall.find(r => ok(r.variant));
    if (fromOverall) return fromOverall.variant;
  }
  return rendered[0] || state.variants[0] || null;
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
  const layoutDims = { 1: [1, 1], 2: [1, 2], 4: [2, 2], 9: [3, 3] };   // STO-SCN-110: 4×4 removed
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

      // Description (STO-SCN-106): ultra-succinct narrative of how this render
      // was built (derived from the manifest provenance).
      const ddesc = (state.manifests[v] || {}).description;
      if (ddesc) {
        const dv = document.createElement("div");
        dv.className = "desc";
        dv.textContent = ddesc;
        tile.appendChild(dv);
      }

      // STO-SCN-112: per-tier letter badge (circle = tier color, black letter
      // = position within the tier). Follows the render everywhere it appears.
      const tbadge = badgeEl(v);
      if (tbadge) tile.appendChild(tbadge);

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

// STO-SCN-112: per-tier letter badge. A render placed in a ranking tier carries a circular
// tag — the circle in that tier's color, a black letter (A, B, C…) for its position within
// the tier — and the SAME tag is drawn wherever the render appears (grid tile, ranking card,
// live-results row). Pool (un-ranked) renders get no badge. Single source of truth so all
// surfaces agree; recomputed on every render as tier membership changes.

// Extended fallback palette for tiers beyond the 6 CSS-defined colors (--tier-1..6), so every
// tier has a stable color. Index 0 == tier 7.
const TIER_FALLBACK = ["#9b59b6", "#1abc9c", "#e67e22", "#3498db", "#2ecc71", "#e74c3c"];
function tierColor(tierNum) {
  // tierNum is 1-based. Prefer the CSS custom property the tier labels use.
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--tier-${tierNum}`).trim();
  if (v) return v;
  return TIER_FALLBACK[(tierNum - 7) % TIER_FALLBACK.length] || "#888";
}
// The LETTER is a stable, unique per-variant identity across the WHOLE view (A, B, C, D…,
// then AA, AB… past 26) — never repeated, independent of which tier the card sits in. The
// COLOR is the tier the card currently belongs to (pool / un-ranked → neutral pool color).
function letterForIndex(i) {
  let s = "", n = i + 1;
  while (n > 0) { const r = (n - 1) % 26; s = String.fromCharCode(65 + r) + s; n = Math.floor((n - 1) / 26); }
  return s;
}
function variantLetter(v) {
  const i = state.variants.indexOf(v);
  return i >= 0 ? letterForIndex(i) : "?";
}
// → { tier, letter, color }. Always returns a badge (every card carries its unique letter).
function badgeFor(v) {
  const letter = variantLetter(v);
  for (let t = 0; t < state.tiers.length; t++) {
    if (state.tiers[t].indexOf(v) >= 0) {
      return { tier: t + 1, letter, color: tierColor(t + 1) };
    }
  }
  const pool = getComputedStyle(document.documentElement).getPropertyValue("--pool-color").trim();
  return { tier: null, letter, color: pool || "#666" };
}
function badgeEl(v) {
  const b = badgeFor(v);
  if (!b) return null;
  const el = document.createElement("div");
  el.className = "tier-badge";
  el.style.background = b.color;
  el.textContent = b.letter;
  el.title = `${b.tier ? "tier " + b.tier : "pool"} · ${b.letter}`;
  return el;
}
// Inline (list-row) form of the same badge — for the Live-Results rows.
function badgeHtml(v) {
  const b = badgeFor(v);
  if (!b) return "";
  return `<span class="tier-badge inline" style="background:${b.color}" title="${b.tier ? "tier " + b.tier : "pool"} · ${b.letter}">${b.letter}</span>`;
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
  // STO-SCN-110: "+ Tier" is its own row at the bottom — click anywhere on it to add a tier.
  const addRow = document.createElement("div");
  addRow.className = "tier-row tier-add-row";
  addRow.title = "Add another tier";
  addRow.innerHTML = `<div class="tier-add">+ Tier</div>`;
  addRow.addEventListener("click", () => { addTier(); persistDrafts(); });
  els.tiers.appendChild(addRow);
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

  // Description (STO-SCN-106): how this render was built. The small ranking card
  // shows it as a hover tooltip (it's tiny); the big grid tile shows it visibly.
  const cdesc = (state.manifests[v] || {}).description;
  if (cdesc) card.title = `${labelOf(v)}\n${cdesc}`;

  // STO-SCN-112: same per-tier letter badge as the grid tile.
  const cbadge = badgeEl(v);
  if (cbadge) card.appendChild(cbadge);

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
    if (state.layout === 1) renderGrid();        // STO-SCN-112: badge on the focused tile follows tier moves
    if (state.agg) renderResults(state.agg);     // STO-SCN-112: results-row badges track tier membership
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

// STO-SCN-111: deep-link URL to the focused rendering (shared by Copy MD + Copy Link).
function deepLinkUrl() {
  if (!state.focusVariant) return "";
  return `${location.origin}/rank?scene=${encodeURIComponent(state.scene)}` +
         `&view=${encodeURIComponent(state.view)}&variant=${encodeURIComponent(state.focusVariant)}`;
}

// Clipboard write with an execCommand fallback for non-secure (http) contexts.
async function copyText(text, okMsg) {
  try { await navigator.clipboard.writeText(text); setStatus(okMsg); return; }
  catch (e) { /* fall through */ }
  const ta = document.createElement("textarea");
  ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
  document.body.appendChild(ta); ta.select();
  try { document.execCommand("copy"); setStatus(okMsg + " (fallback)"); }
  catch (_) { setStatus("Copy failed — select the text manually."); }
  ta.remove();
}

// STO-SCN-110: the focused variant's manifest as Markdown, with a STO-SCN-111 deep-link.
function manifestMarkdown() {
  const v = state.focusVariant;
  if (!v) return "";
  const m = state.manifests[v] || {};
  const label = labelOf(v);
  const link = deepLinkUrl();
  let md = `### ${label}\n`;
  if (label !== v) md += `\`${v}\`\n`;
  if (m.description) md += `\n_${m.description}_\n`;
  const ms = m.mesh || {};
  if (ms.verts) md += `\n- **mesh:** ${ms.verts} verts · ${ms.faces} tris · ${ms.size_mb} MB\n`;
  for (const [tn, t] of Object.entries(m.transforms || {})) {
    const params = (t && t.parameters) || {};
    const ps = Object.entries(params)
      .filter(([, val]) => val !== null && val !== "" && !(typeof val === "string" && val.length > 100))
      .map(([k, val]) => `${k}=${typeof val === "object" ? JSON.stringify(val) : val}`).join(" · ");
    md += `- **${tn}**${t && t.kind ? ` (${t.kind})` : ""}${ps ? `: ${ps}` : ""}\n`;
  }
  if (m.notes) md += `\n> ${m.notes}\n`;
  md += `\n**Scene** \`${state.scene}\` · **View** \`${state.view}\`\n`;
  md += `\n[↩ Open this rendering](${link})\n`;
  return md;
}

async function copyManifestMarkdown() {
  const md = manifestMarkdown();
  if (!md) { setStatus("Hover/select a render first — nothing to copy."); return; }
  await copyText(md, "Manifest copied as Markdown (with deep-link).");
}

async function copyLink() {                              // STO-SCN-110/111
  const url = deepLinkUrl();
  if (!url) { setStatus("Hover/select a render first — nothing to link."); return; }
  await copyText(url, "Deep-link copied.");
}

function renderResults(agg) {
  if (!agg || !agg.n_submissions) {
    els.results.innerHTML = "<em>No rankings submitted yet.</em>";
    return;
  }
  // STO-SCN-110: ranked items are clickable → show that render (per-view items also switch view).
  const li = (row, i, dp, view) => {
    const known = state.variants.includes(row.variant);
    const cls = (i === 0 ? "winner " : "") + (known ? "rank-item" : "");
    const attr = known
      ? ` data-variant="${escapeHtml(row.variant)}"${view ? ` data-view="${escapeHtml(view)}"` : ""} title="Show this render"`
      : "";
    return `<li class="${cls}"${attr}>${badgeHtml(row.variant)}${escapeHtml(labelOf(row.variant))} ` +
           `<span style="color:var(--text-dim);">— ${row.score.toFixed(dp)}</span></li>`;
  };
  let html = `<div style="color: var(--text-dim);">${agg.n_submissions} submission${agg.n_submissions === 1 ? "" : "s"} total</div>`;
  if (agg.overall && agg.overall.length) {
    html += "<h4>Overall</h4><ol>" + agg.overall.map((r, i) => li(r, i, 3)).join("") + "</ol>";
  }
  for (const view of Object.keys(agg.per_view).sort()) {
    const v = agg.per_view[view];
    html += `<h4>${escapeHtml(view)} <span style="color:var(--text-dim); font-weight:400;">(${v.n_submissions})</span></h4>` +
            "<ol>" + v.leaderboard.map((r, i) => li(r, i, 2, view)).join("") + "</ol>";
  }
  els.results.innerHTML = html;
}

// STO-SCN-110: click a Live-Results item → show its render (switch view first if needed).
function onResultClick(e) {
  const el = e.target.closest("[data-variant]");
  if (!el) return;
  const v = el.dataset.variant;
  const view = el.dataset.view;
  if (view && view !== state.view && state.views.includes(view)) {
    saveDraftForView(state.view); persistDrafts();
    state.view = view; loadDraftForView(state.view);
    state.focusVariant = v; renderViewCard(); refreshAll();
  } else {
    setFocusVariant(v);
  }
}

// ---- Rater select ------------------------------------------------------

const NEW_RATER_SENTINEL = "__new__";

function rebuildRaterSelect() {
  // Combine raters from server (people who've submitted on this scene) with
  // the locally-stored last-used name (in case it's a new rater who hasn't
  // submitted yet) and any locally-cached list of names. Sort, dedupe.
  const local = JSON.parse(localStorage.getItem("rater-list") || "[]");
  const all = Array.from(new Set([
    ...(state.profiles || []),       // STO-SCN-108: server-side profiles (primary, shared)
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

async function handleRaterSelect() {
  const v = els.raterSelect.value;
  if (v === NEW_RATER_SENTINEL) {
    const name = (prompt("New profile name (no password):") || "").trim();
    if (!name) {
      // User cancelled — restore previous selection
      els.raterSelect.value = state.rater || "";
      return;
    }
    state.rater = name;
    localStorage.setItem("rater", name);   // cache the current selection only
    // STO-SCN-108: persist the profile SERVER-SIDE so it's available at any
    // origin (not just this browser). localStorage rater-list kept as a cache.
    const local = JSON.parse(localStorage.getItem("rater-list") || "[]");
    if (!local.includes(name)) {
      local.push(name);
      localStorage.setItem("rater-list", JSON.stringify(local));
    }
    try {
      const d = await api("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      state.profiles = d.profiles || state.profiles;
    } catch (e) {
      console.warn("profile save failed:", e);
    }
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

// STO-SCN-110: View selector as a scene-card-style card with ◀/▶ arrows (replaces the
// "View" dropdown), placed where the "Rank these" heading used to be. Thumbnail = a render
// of the current view (first available variant). Arrows step state.view.
function renderViewCard() {
  if (!els.viewCard) return;
  const views = state.views || [];
  const i = Math.max(0, views.indexOf(state.view));
  const n = views.length;
  const variant = (state.rendered[state.view] || [])[0];
  const thumb = variant ? `/api/render/${state.scene}/${state.view}/${variant}.png` : null;
  const nm = escapeHtml(state.view || "—");
  els.viewCard.innerHTML =
    `<button class="vc-arrow" id="vc-prev" ${n < 2 ? "disabled" : ""} title="Previous view">‹</button>` +
    `<div class="vc-card" title="${nm}">` +
      (thumb ? `<img src="${thumb}" alt="${nm}">` : `<div class="vc-noimg">no render</div>`) +
    `</div>` +
    `<button class="vc-arrow" id="vc-next" ${n < 2 ? "disabled" : ""} title="Next view">›</button>`;
  const prev = els.viewCard.querySelector("#vc-prev");
  const next = els.viewCard.querySelector("#vc-next");
  if (prev) prev.onclick = () => stepView(-1);
  if (next) next.onclick = () => stepView(1);
  if (els.viewTitle) els.viewTitle.textContent = `View ${n ? i + 1 : 0} of ${n}`;
}

function stepView(delta) {
  const views = state.views || [];
  if (views.length < 2) return;
  const i = Math.max(0, views.indexOf(state.view));
  saveDraftForView(state.view);
  persistDrafts();
  state.view = views[(i + delta + views.length) % views.length];
  loadDraftForView(state.view);
  state.focusVariant = topRankedVariant(state.view);   // STO-SCN-110: view always shows highest-ranked
  renderViewCard();
  refreshAll();
}

document.addEventListener("DOMContentLoaded", async () => {
  els.raterSelect.addEventListener("change", handleRaterSelect);
  els.sceneLeft.addEventListener("click", () =>
    els.sceneStrip.scrollBy({ left: -els.sceneStrip.clientWidth * 0.8, behavior: "smooth" }));
  els.sceneRight.addEventListener("click", () =>
    els.sceneStrip.scrollBy({ left: els.sceneStrip.clientWidth * 0.8, behavior: "smooth" }));
  // STO-SCN-110: view switching is via the View card's ◀/▶ (wired in renderViewCard).
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
  // STO-SCN-110: "+ Tier" is the bottom tier row (wired per-render in renderTiers).
  els.submitBtn.addEventListener("click", submitRanking);
  els.copyManifestBtn.addEventListener("click", copyManifestMarkdown);   // STO-SCN-110
  els.copyLinkBtn.addEventListener("click", copyLink);                   // STO-SCN-110/111
  els.results.addEventListener("click", onResultClick);                  // STO-SCN-110: clickable leaderboard

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
  });

  await loadProfiles();    // STO-SCN-108: server-side rater list (before the dropdown builds)
  await loadScenes();
  if (state.scene) await loadScene();
});

// STO-SCN-108: pull the store-level profile list (origin-independent rater identities).
async function loadProfiles() {
  try {
    const d = await api("/api/profiles");
    state.profiles = d.profiles || [];
  } catch (e) {
    state.profiles = [];
    console.warn("profiles fetch failed:", e);
  }
}
