/* ==========================================================================
 * Scenes tab — Spine view (STO-SCN-147).
 *
 * Registers window.scenesViews.spine. Embeds the focused Three.js spine viewer
 * (spine.html) in an iframe, fed by GET /api/scene/<scene>/spine (posed
 * frustums + gravity-up + per-subset colours + legend). Read-only.
 * ========================================================================== */
"use strict";

(function sceneSpineView() {
  window.scenesViews = window.scenesViews || {};

  function render(container, scene) {
    const data = `/api/scene/${encodeURIComponent(scene)}/spine`;
    const src = `/static/spine.html?data=${encodeURIComponent(data)}&cb=${Date.now()}`;
    container.innerHTML =
      `<iframe class="spine-frame" src="${src}" title="camera spine for ${scene}"></iframe>`;
  }

  window.scenesViews.spine = render;
})();
