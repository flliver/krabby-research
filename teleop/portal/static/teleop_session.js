(function () {
  var PATH = typeof window.TELEOP_WS_PATH === 'string' ? window.TELEOP_WS_PATH : '/ws';
  var videosEl = document.getElementById('videos');
  var status = document.getElementById('status');
  var latencyEl = document.getElementById('latency');
  var streamStatus = document.getElementById('streamStatus');
  var debugEl = document.getElementById('debugRtc');
  var catalogListEl = document.getElementById('catalogList');
  var controllerStatusEl = document.getElementById('controllerStatus');
  var controlChannelStatusEl = document.getElementById('controlChannelStatus');
  var controlSendStatusEl = document.getElementById('controlSendStatus');
  var controllerStateMirrorEl = document.getElementById('controllerStateMirror');
  var virtualGamepadEl = document.getElementById('virtualGamepad');
  var virtualGamepadStatusEl = document.getElementById('virtualGamepadStatus');
  var virtualGamepadResetEl = document.getElementById('virtualGamepadReset');
  var operatorOverrideStatusEl = document.getElementById('operatorOverrideStatus');
  var controllerDiagnosticDetails = document.getElementById('controllerDiagnosticPanel');
  if (controllerDiagnosticDetails) {
    controllerDiagnosticDetails.open = false;
  }

  function setConnectionStatus(message) {
    if (status) {
      status.textContent = message;
    }
    var m = String(message || '').toLowerCase();
    var phase = 'idle';
    if (m.indexOf('playing') !== -1) {
      phase = 'live';
    } else if (m.indexOf('error') !== -1 || m.indexOf('websocket error') !== -1) {
      phase = 'error';
    } else if (
      m.indexOf('negotiating') !== -1 ||
      m.indexOf('connecting') !== -1 ||
      m.indexOf('waiting') !== -1 ||
      m.indexOf('loading') !== -1 ||
      m.indexOf('hello') !== -1 ||
      m.indexOf('not ready') !== -1
    ) {
      phase = 'connecting';
    }
    document.body.setAttribute('data-teleop-phase', phase);
  }
  var params = new URLSearchParams(location.search);
  var token = params.get('token');
  var qs = token ? '?token=' + encodeURIComponent(token) : '';
  var httpProto = location.protocol === 'https:' ? 'https:' : 'http:';
  var cfgUrl = httpProto + '//' + location.host + '/api/teleop-config' + qs;
  var wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var wsUrl = wsProto + '//' + location.host + PATH + qs;

  /** Radius [0,1) for radial stick deadzone (resting jitter); raise if a pad needs more margin. */
  var TELEOP_GAMEPAD_STICK_DEADZONE = 0.1;
  var VIRTUAL_BUTTON_NAMES = ['LT', 'LB', 'LS', 'RS', 'RT', 'RB'];
  var virtualControllerButtons = {
    LT: false, LB: false, LS: false, RS: false, RT: false, RB: false
  };
  var virtualControllerAxes = { LX: 0.0, LY: 0.0, RX: 0.0, RY: 0.0 };
  var virtualStickEls = {};
  var activeVirtualPointers = {};
  var virtualControlsEnabled = false;
  var virtualInputSelected = false;

  var stunTurnServers = [{ urls: 'stun:stun.l.google.com:19302' }];
  var ws = null;
  var pc = null;
  var controlDc = null;
  var answerWaiter = null;
  var availableCatalogIds = [];
  var lastOffsetMs = null;
  var captureByCatalogIdNs = {};
  var gamepadLoopTimer = null;
  var controlSendCount = 0;
  var lastControlSendMs = 0;
  var teleopOperatorOverrideEl = document.getElementById('teleopOperatorOverride');
  /** First offer must run after ``hello_ack`` so ``catalog_ids`` matches checkboxes (not ``[]``). */
  var initialRtcStarted = false;
  var initialRtcFallbackTimer = null;

  function clampAxis(v) {
    if (v < -1) return -1;
    if (v > 1) return 1;
    return v;
  }

  /** Per-stick radial deadzone: zero inside radius ``zone``, smooth rescale toward full ±1 beyond. */
  function stickPairAfterDeadzone(x, y, zone) {
    if (!(zone > 0)) {
      return [x, y];
    }
    var m = Math.sqrt(x * x + y * y);
    if (m <= zone) {
      return [0.0, 0.0];
    }
    var gain = (m - zone) / (m * (1.0 - zone));
    return [clampAxis(x * gain), clampAxis(y * gain)];
  }

  function gamepadApiAvailable() {
    return typeof navigator !== 'undefined' && typeof navigator.getGamepads === 'function';
  }

  function gamepadSnapshot() {
    if (!gamepadApiAvailable()) {
      return [];
    }
    try {
      return navigator.getGamepads() || [];
    } catch (e) {
      return [];
    }
  }

  function copyNeutralControllerState() {
    return {
      LT: false, LB: false, LS: false, RS: false, RT: false, RB: false,
      LX: 0.0, LY: 0.0, RX: 0.0, RY: 0.0
    };
  }

  function operatorOverrideEnabled() {
    return !!(teleopOperatorOverrideEl && teleopOperatorOverrideEl.checked);
  }

  function readVirtualControllerState() {
    var st = copyNeutralControllerState();
    VIRTUAL_BUTTON_NAMES.forEach(function (name) {
      st[name] = !!virtualControllerButtons[name];
    });
    st.LX = clampAxis(virtualControllerAxes.LX);
    st.LY = clampAxis(virtualControllerAxes.LY);
    st.RX = clampAxis(virtualControllerAxes.RX);
    st.RY = clampAxis(virtualControllerAxes.RY);
    return st;
  }

  function normalizeStickPair(x, y) {
    x = clampAxis(x);
    y = clampAxis(y);
    var m = Math.sqrt(x * x + y * y);
    if (m > 1) {
      x /= m;
      y /= m;
    }
    return [x, y];
  }

  function setVirtualStick(name, x, y) {
    var pair = normalizeStickPair(x, y);
    if (name === 'left') {
      virtualControllerAxes.LX = pair[0];
      virtualControllerAxes.LY = pair[1];
    } else if (name === 'right') {
      virtualControllerAxes.RX = pair[0];
      virtualControllerAxes.RY = pair[1];
    }
    updateVirtualStickKnob(name);
  }

  function updateVirtualStickKnob(name) {
    var stick = virtualStickEls[name];
    if (!stick) return;
    var knob = stick.querySelector('[data-virtual-knob]');
    if (!knob) return;
    var x = name === 'left' ? virtualControllerAxes.LX : virtualControllerAxes.RX;
    var y = name === 'left' ? virtualControllerAxes.LY : virtualControllerAxes.RY;
    knob.style.left = (50 + (x * 31)) + '%';
    knob.style.top = (50 + (y * 31)) + '%';
  }

  function pointerEventToStickPair(stick, ev) {
    var rect = stick.getBoundingClientRect();
    var halfW = rect.width / 2.0;
    var halfH = rect.height / 2.0;
    var x = halfW > 0 ? (ev.clientX - (rect.left + halfW)) / halfW : 0.0;
    var y = halfH > 0 ? (ev.clientY - (rect.top + halfH)) / halfH : 0.0;
    return normalizeStickPair(x, y);
  }

  function syncVirtualButtonEls() {
    if (!virtualGamepadEl) return;
    VIRTUAL_BUTTON_NAMES.forEach(function (name) {
      var btn = virtualGamepadEl.querySelector('[data-virtual-button="' + name + '"]');
      if (btn) {
        btn.setAttribute('aria-pressed', virtualControllerButtons[name] ? 'true' : 'false');
      }
    });
  }

  function setVirtualControlsEnabled(enabled) {
    virtualControlsEnabled = !!enabled;
    if (virtualGamepadEl) {
      virtualGamepadEl.setAttribute('data-enabled', virtualControlsEnabled ? 'true' : 'false');
      virtualGamepadEl.setAttribute('aria-disabled', virtualControlsEnabled ? 'false' : 'true');
    }
    Object.keys(virtualStickEls).forEach(function (name) {
      var stick = virtualStickEls[name];
      if (!stick) return;
      stick.setAttribute('aria-disabled', virtualControlsEnabled ? 'false' : 'true');
      stick.setAttribute('tabindex', virtualControlsEnabled ? '0' : '-1');
    });
    if (virtualGamepadEl) {
      VIRTUAL_BUTTON_NAMES.forEach(function (name) {
        var btn = virtualGamepadEl.querySelector('[data-virtual-button="' + name + '"]');
        if (btn) {
          btn.disabled = !virtualControlsEnabled;
        }
      });
    }
    if (virtualGamepadResetEl) {
      virtualGamepadResetEl.disabled = !virtualControlsEnabled;
    }
    if (!virtualControlsEnabled) {
      activeVirtualPointers = {};
      resetVirtualController();
    }
  }

  function resetVirtualController() {
    VIRTUAL_BUTTON_NAMES.forEach(function (name) {
      virtualControllerButtons[name] = false;
    });
    setVirtualStick('left', 0.0, 0.0);
    setVirtualStick('right', 0.0, 0.0);
    syncVirtualButtonEls();
  }

  function updateVirtualGamepadControlState(operatorOverrideOn, physicalGamepad) {
    var enabled = operatorOverrideOn;
    setVirtualControlsEnabled(enabled);
    if (virtualGamepadEl) {
      virtualGamepadEl.setAttribute('data-active', (!physicalGamepad || virtualInputSelected) ? 'true' : 'false');
    }
    if (virtualGamepadStatusEl) {
      if (!operatorOverrideOn) {
        virtualGamepadStatusEl.textContent = 'Enable Operator override to use virtual controls.';
      } else if (physicalGamepad && virtualInputSelected) {
        virtualGamepadStatusEl.textContent =
          'Virtual controls active; physical gamepad input resumes when virtual controls are released.';
      } else if (physicalGamepad) {
        virtualGamepadStatusEl.textContent =
          'Physical gamepad active. Touch or click virtual controls to use them instead.';
      } else {
        virtualGamepadStatusEl.textContent = gamepadApiAvailable() ?
          'Virtual controls enabled. Connect a physical gamepad to use hardware by default.' :
          'Virtual controls enabled because this browser has no Gamepad API.';
      }
    }
  }

  function initVirtualGamepad() {
    if (!virtualGamepadEl) return;
    var sticks = virtualGamepadEl.querySelectorAll('[data-virtual-stick]');
    sticks.forEach(function (stick) {
      var name = stick.getAttribute('data-virtual-stick');
      if (!name) return;
      virtualStickEls[name] = stick;
      updateVirtualStickKnob(name);
      stick.addEventListener('pointerdown', function (ev) {
        if (!virtualControlsEnabled) {
          ev.preventDefault();
          return;
        }
        ev.preventDefault();
        virtualInputSelected = true;
        activeVirtualPointers[name] = ev.pointerId;
        if (stick.setPointerCapture) {
          stick.setPointerCapture(ev.pointerId);
        }
        var pair = pointerEventToStickPair(stick, ev);
        setVirtualStick(name, pair[0], pair[1]);
      });
      stick.addEventListener('pointermove', function (ev) {
        if (!virtualControlsEnabled) return;
        if (activeVirtualPointers[name] !== ev.pointerId) return;
        ev.preventDefault();
        var pair = pointerEventToStickPair(stick, ev);
        setVirtualStick(name, pair[0], pair[1]);
      });
      function endPointer(ev) {
        if (activeVirtualPointers[name] !== ev.pointerId) return;
        ev.preventDefault();
        delete activeVirtualPointers[name];
        setVirtualStick(name, 0.0, 0.0);
        if (stick.releasePointerCapture) {
          try {
            stick.releasePointerCapture(ev.pointerId);
          } catch (e) {}
        }
      }
      stick.addEventListener('pointerup', endPointer);
      stick.addEventListener('pointercancel', endPointer);
      stick.addEventListener('lostpointercapture', function (ev) {
        if (activeVirtualPointers[name] !== ev.pointerId) return;
        delete activeVirtualPointers[name];
        setVirtualStick(name, 0.0, 0.0);
      });
      stick.addEventListener('keydown', function (ev) {
        if (!virtualControlsEnabled) return;
        var step = ev.shiftKey ? 1.0 : 0.45;
        if (ev.key === 'ArrowLeft') {
          ev.preventDefault();
          virtualInputSelected = true;
          setVirtualStick(name, -step, 0.0);
        } else if (ev.key === 'ArrowRight') {
          ev.preventDefault();
          virtualInputSelected = true;
          setVirtualStick(name, step, 0.0);
        } else if (ev.key === 'ArrowUp') {
          ev.preventDefault();
          virtualInputSelected = true;
          setVirtualStick(name, 0.0, -step);
        } else if (ev.key === 'ArrowDown') {
          ev.preventDefault();
          virtualInputSelected = true;
          setVirtualStick(name, 0.0, step);
        } else if (ev.key === ' ' || ev.key === 'Enter') {
          ev.preventDefault();
          virtualInputSelected = true;
          setVirtualStick(name, 0.0, 0.0);
        }
      });
      stick.addEventListener('keyup', function (ev) {
        if (
          ev.key === 'ArrowLeft' ||
          ev.key === 'ArrowRight' ||
          ev.key === 'ArrowUp' ||
          ev.key === 'ArrowDown'
        ) {
          ev.preventDefault();
          setVirtualStick(name, 0.0, 0.0);
        }
      });
    });
    VIRTUAL_BUTTON_NAMES.forEach(function (name) {
      var btn = virtualGamepadEl.querySelector('[data-virtual-button="' + name + '"]');
      if (!btn) return;
      btn.addEventListener('click', function () {
        if (!virtualControlsEnabled) return;
        virtualInputSelected = true;
        virtualControllerButtons[name] = !virtualControllerButtons[name];
        syncVirtualButtonEls();
      });
    });
    if (virtualGamepadResetEl) {
      virtualGamepadResetEl.addEventListener('click', resetVirtualController);
    }
    window.addEventListener('blur', resetVirtualController);
    window.addEventListener('pagehide', resetVirtualController);
    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState === 'hidden') {
        resetVirtualController();
      }
    });
    if (teleopOperatorOverrideEl) {
      teleopOperatorOverrideEl.addEventListener('change', function () {
        if (!operatorOverrideEnabled()) {
          virtualInputSelected = false;
          resetVirtualController();
        }
        updateControllerDebugPanel();
      });
    }
    setVirtualControlsEnabled(false);
    syncVirtualButtonEls();
  }

  function readPhysicalGamepadState(p) {
    function b(idx) { return !!(p.buttons && p.buttons[idx] && p.buttons[idx].pressed); }
    function a(idx) {
      var v = (p.axes && typeof p.axes[idx] === 'number') ? p.axes[idx] : 0.0;
      return clampAxis(v);
    }
    var lx0 = a(0);
    var ly0 = a(1);
    var rx0 = a(2);
    var ry0 = a(3);
    var left = stickPairAfterDeadzone(lx0, ly0, TELEOP_GAMEPAD_STICK_DEADZONE);
    var right = stickPairAfterDeadzone(rx0, ry0, TELEOP_GAMEPAD_STICK_DEADZONE);
    return {
      LT: b(6), LB: b(4), LS: b(10), RS: b(11), RT: b(7), RB: b(5),
      LX: left[0],
      LY: left[1],
      RX: right[0],
      RY: right[1]
    };
  }

  function readGamepadState() {
    if (!operatorOverrideEnabled()) {
      return copyNeutralControllerState();
    }
    var p = firstConnectedGamepad();
    var virtualState = readVirtualControllerState();
    if (!p) {
      return virtualState;
    }
    var physicalState = readPhysicalGamepadState(p);
    if (virtualInputSelected) {
      if (controllerStateHasInput(virtualState)) {
        return virtualState;
      }
      virtualInputSelected = false;
    }
    return physicalState;
  }

  function controllerStateHasInput(st) {
    if (!st) return false;
    if (st.LT || st.LB || st.LS || st.RS || st.RT || st.RB) {
      return true;
    }
    return (
      Math.abs(st.LX || 0) > 0.01 ||
      Math.abs(st.LY || 0) > 0.01 ||
      Math.abs(st.RX || 0) > 0.01 ||
      Math.abs(st.RY || 0) > 0.01
    );
  }

  function updateOperatorOverrideStatus(enabled, hasInput) {
    if (!operatorOverrideStatusEl) return;
    operatorOverrideStatusEl.setAttribute('data-enabled', enabled ? 'true' : 'false');
    if (enabled) {
      operatorOverrideStatusEl.textContent =
        'Override ON: joystick commands can move the robot.';
    } else {
      operatorOverrideStatusEl.textContent =
        'Override OFF: joystick controls are disabled and will not move the robot.';
    }
  }

  function firstConnectedGamepad() {
    var pads = gamepadSnapshot();
    for (var i = 0; i < pads.length; i += 1) {
      if (pads[i]) return pads[i];
    }
    return null;
  }

  function updateControllerDebugPanel() {
    var gp = firstConnectedGamepad();
    var st = readGamepadState();
    var operatorOverrideOn = !!(teleopOperatorOverrideEl && teleopOperatorOverrideEl.checked);
    var hasInput = controllerStateHasInput(st);
    updateOperatorOverrideStatus(operatorOverrideOn, hasInput);
    if (controllerStatusEl) {
      if (virtualInputSelected) {
        controllerStatusEl.textContent = gp ?
          'Controller: virtual controls active (physical gamepad resumes on release)' :
          'Controller: virtual controls active';
      } else if (gp) {
        controllerStatusEl.textContent =
          'Controller: connected (' + (gp.id || 'unknown') + ', index ' + gp.index + ')';
      } else {
        controllerStatusEl.textContent = gamepadApiAvailable() ?
          'Controller: virtual fallback (no physical gamepad connected)' :
          'Controller: virtual fallback (Gamepad API unavailable)';
      }
    }
    updateVirtualGamepadControlState(operatorOverrideOn, !!gp);
    if (controlChannelStatusEl) {
      var dcState = controlDc ? controlDc.readyState : 'n/a';
      controlChannelStatusEl.textContent =
        'Control channel (WebRTC krabby-control-v1): ' + dcState;
    }
    if (controlSendStatusEl) {
      if (controlDc && controlDc.readyState === 'open') {
        if (!operatorOverrideOn) {
          controlSendStatusEl.textContent =
            'Relayed to robot: override off; joystick controls disabled';
        } else if (lastControlSendMs > 0) {
          controlSendStatusEl.textContent =
            'Relayed to robot: ' + controlSendCount + ' msgs, last ' +
            (Date.now() - lastControlSendMs) + 'ms ago';
        } else {
          controlSendStatusEl.textContent =
            'Relayed to robot: channel open (first control frame pending…)';
        }
      } else {
        controlSendStatusEl.textContent =
          'Relayed to robot: idle (no open WebRTC channel; local capture above still updates)';
      }
    }
    if (controllerStateMirrorEl) {
      controllerStateMirrorEl.textContent = JSON.stringify(st, null, 2);
    }
  }

  function stopGamepadLoop() {
    if (gamepadLoopTimer !== null) {
      clearInterval(gamepadLoopTimer);
      gamepadLoopTimer = null;
    }
  }

  function startGamepadLoop() {
    stopGamepadLoop();
    controlSendCount = 0;
    lastControlSendMs = 0;
    gamepadLoopTimer = setInterval(function () {
      if (!controlDc || controlDc.readyState !== 'open') return;
      var st = readGamepadState();
      controlDc.send(
        JSON.stringify({
          type: 'control',
          sent_browser_ms: Date.now(),
          operator_override: !!(teleopOperatorOverrideEl && teleopOperatorOverrideEl.checked),
          state: st
        })
      );
      controlSendCount += 1;
      lastControlSendMs = Date.now();
    }, 20); // 50 Hz
  }

  function waitGatheringComplete(p) {
    return new Promise(function (resolve) {
      if (p.iceGatheringState === 'complete') return resolve();
      p.addEventListener('icegatheringstatechange', function onGathering() {
        if (p.iceGatheringState === 'complete') {
          p.removeEventListener('icegatheringstatechange', onGathering);
          resolve();
        }
      });
    });
  }

  function updateDebug() {
    if (!debugEl || !pc) return;
    debugEl.textContent =
      'RTCPeerConnection: ' +
      pc.connectionState +
      ' | media: ' +
      pc.iceConnectionState +
      ' | signaling: ' +
      pc.signalingState;
  }

  function selectedCatalogIdsFromCheckboxes() {
    if (!catalogListEl) return [];
    var nodes = catalogListEl.querySelectorAll('input[type="checkbox"][data-catalog-id]:checked');
    var out = [];
    nodes.forEach(function (n) {
      var v = n.getAttribute('data-catalog-id');
      if (v) out.push(v);
    });
    return out;
  }

  function renderCatalogList(ids) {
    if (!catalogListEl) return;
    if (!ids || !ids.length) {
      catalogListEl.textContent = 'Available sensors: waiting for robot hello...';
      return;
    }
    catalogListEl.innerHTML = '';
    var label = document.createElement('div');
    label.className = 'catalog-list-heading';
    label.textContent = 'Available sensors';
    catalogListEl.appendChild(label);
    ids.forEach(function (cid) {
      var row = document.createElement('label');
      row.className = 'catalog-row';
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.setAttribute('data-catalog-id', cid);
      cb.checked = true;
      row.appendChild(cb);
      row.appendChild(document.createTextNode(' ' + cid));
      catalogListEl.appendChild(row);
    });
  }

  /** Matches checked count, or ``1`` recvonly line when none checked (robot picks its default camera). */
  function rtcRecvonlyVideoLineCount() {
    var ids = readCatalogIdsArray();
    return ids.length > 0 ? ids.length : 1;
  }

  function selectedCatalogIdsInOrder() {
    return readCatalogIdsArray();
  }

  /** Checked catalog ids in DOM order. None checked -> ``[]`` (offer omits viewer ids; robot uses its default listing). */
  function readCatalogIdsArray() {
    return selectedCatalogIdsFromCheckboxes();
  }

  function helloPayload() {
    return {
      type: 'hello',
      role: 'browser',
      version: 1,
      catalog_ids: readCatalogIdsArray(),
    };
  }

  function offerPayload(sdp) {
    return { type: 'offer', sdp: sdp, catalog_ids: readCatalogIdsArray() };
  }

  async function startRtc(numStreams) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setConnectionStatus('WebSocket not ready');
      return;
    }
    if (pc) {
      pc.close();
      pc = null;
    }
    stopGamepadLoop();
    controlDc = null;
    videosEl.innerHTML = '';
    var videoLabelsForSession = readCatalogIdsArray().slice();
    if (streamStatus) {
      streamStatus.textContent = 'Requested ' + numStreams + ' stream(s); negotiating...';
    }
    setConnectionStatus('Negotiating WebRTC...');

    pc = new RTCPeerConnection({ iceServers: stunTurnServers });
    controlDc = pc.createDataChannel('krabby-control-v1', { ordered: true });
    controlDc.onopen = function () {
      startGamepadLoop();
    };
    controlDc.onclose = function () {
      stopGamepadLoop();
    };
    pc.addEventListener('connectionstatechange', updateDebug);
    pc.addEventListener('iceconnectionstatechange', updateDebug);
    pc.addEventListener('signalingstatechange', updateDebug);
    updateDebug();

    var nTracks = 0;
    pc.ontrack = function (ev) {
      // Unified-plan answers often expose every recv track on the **same** remote MediaStream.
      // Reusing ``ev.streams[0]`` on multiple <video> elements makes each tag show that stream’s
      // default behavior (typically the first/combined video track) — duplicated tiles.
      // One-element MediaStream pins this element to exactly one MediaStreamTrack.
      var isolate = new MediaStream([ev.track]);
      var trackIndex = nTracks;
      nTracks += 1;
      if (streamStatus) {
        streamStatus.textContent = 'Receiving ' + nTracks + ' / ' + numStreams + ' video track(s)';
      }
      var label;
      if (videoLabelsForSession.length > trackIndex && videoLabelsForSession[trackIndex]) {
        label = videoLabelsForSession[trackIndex];
      } else if (videoLabelsForSession.length === 0) {
        label = numStreams <= 1 ? 'Robot default' : 'Video ' + (trackIndex + 1);
      } else {
        label = 'Video ' + (trackIndex + 1);
      }
      var tile = document.createElement('div');
      tile.className = 'video-tile';
      var cap = document.createElement('div');
      cap.className = 'video-tile-label';
      cap.textContent = label;
      var v = document.createElement('video');
      v.autoplay = true;
      v.playsInline = true;
      v.muted = true;
      v.style.maxWidth = '100%';
      v.style.background = '#111';
      v.srcObject = isolate;
      tile.appendChild(cap);
      tile.appendChild(v);
      videosEl.appendChild(tile);
    };

    var i;
    for (i = 0; i < numStreams; i += 1) {
      pc.addTransceiver('video', { direction: 'recvonly' });
    }
    var offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitGatheringComplete(pc);

    var ans = await new Promise(function (resolve, reject) {
      answerWaiter = { resolve: resolve, reject: reject };
      ws.send(JSON.stringify(offerPayload(pc.localDescription.sdp)));
    });
    await pc.setRemoteDescription({ type: 'answer', sdp: ans.sdp });
    setConnectionStatus('Playing');
    updateDebug();
  }

  function tryStartInitialRtc() {
    if (initialRtcStarted) {
      return;
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    initialRtcStarted = true;
    if (initialRtcFallbackTimer !== null) {
      clearTimeout(initialRtcFallbackTimer);
      initialRtcFallbackTimer = null;
    }
    startRtc(rtcRecvonlyVideoLineCount()).catch(function (err) {
      setConnectionStatus('WebRTC error: ' + (err && err.message ? err.message : String(err)));
    });
  }

  function openWebSocket() {
    initialRtcStarted = false;
    if (initialRtcFallbackTimer !== null) {
      clearTimeout(initialRtcFallbackTimer);
      initialRtcFallbackTimer = null;
    }
    ws = new WebSocket(wsUrl);
    ws.onerror = function () {
      setConnectionStatus('WebSocket error');
      stopGamepadLoop();
    };
    ws.onclose = function () {
      stopGamepadLoop();
    };

    ws.onmessage = function (e) {
      var msg = JSON.parse(e.data);
      if (msg.type === 'error' && answerWaiter) {
        answerWaiter.reject(new Error(msg.message || 'server error'));
        answerWaiter = null;
        return;
      }
      if (msg.type === 'answer' && answerWaiter) {
        answerWaiter.resolve(msg);
        answerWaiter = null;
        return;
      }
      if (msg.type === 'hello_ack') {
        if (Array.isArray(msg.available_catalog_ids)) {
          availableCatalogIds = msg.available_catalog_ids
            .map(function (s) {
              return String(s || '').trim();
            })
            .filter(Boolean);
          renderCatalogList(availableCatalogIds);
        }
        tryStartInitialRtc();
      }
      if (msg.type === 'pong' && typeof msg.t === 'number' && latencyEl) {
        var nowPerf = performance.now();
        var rttMs = Math.round(nowPerf - msg.t);
        if (typeof msg.t_wall_ms === 'number' && typeof msg.server_ms === 'number') {
          var t0 = msg.t_wall_ms;
          var t1 = msg.server_ms;
          var t3 = Date.now();
          var offset = t1 - ((t0 + t3) / 2.0); // robot wall-clock minus browser wall-clock
          if (lastOffsetMs === null) {
            lastOffsetMs = offset;
          } else {
            // Smooth jitter with light EMA.
            lastOffsetMs = (0.85 * lastOffsetMs) + (0.15 * offset);
          }
        }
        if (msg.capture_timestamps_ns && typeof msg.capture_timestamps_ns === 'object') {
          captureByCatalogIdNs = msg.capture_timestamps_ns;
        }
        var g2gText = 'g2g: n/a';
        var selected = selectedCatalogIdsInOrder();
        if (selected.length > 0 && lastOffsetMs !== null) {
          var cid = selected[0];
          var capNs = captureByCatalogIdNs[cid];
          if (typeof capNs === 'number') {
            var capMsRobot = capNs / 1e6;
            var capMsBrowser = capMsRobot - lastOffsetMs;
            var estG2g = Date.now() - capMsBrowser;
            if (isFinite(estG2g)) {
              g2gText = 'g2g~' + Math.max(0, Math.round(estG2g)) + ' ms (capture->render est, stream ' + cid + ')';
            }
          }
        }
        latencyEl.textContent =
          'RTT~' + rttMs + ' ms; offset~' +
          (lastOffsetMs === null ? 'n/a' : Math.round(lastOffsetMs) + ' ms') +
          '; ' + g2gText;
      }
    };

    ws.onopen = function () {
      setConnectionStatus('Waiting for robot hello…');
      try {
        ws.send(JSON.stringify(helloPayload()));
      } catch (e) {}
      initialRtcFallbackTimer = setTimeout(function () {
        initialRtcFallbackTimer = null;
        if (!initialRtcStarted && ws && ws.readyState === WebSocket.OPEN) {
          setConnectionStatus('No hello_ack yet; starting WebRTC with current selection…');
          tryStartInitialRtc();
        }
      }, 8000);
      setInterval(function () {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping', t: performance.now(), t_wall_ms: Date.now() }));
        }
      }, 2000);
    };
  }

  function boot() {
    setConnectionStatus('Loading config...');
    fetch(cfgUrl)
      .then(function (r) {
        if (!r.ok) return Promise.reject(new Error('config ' + r.status));
        return r.json();
      })
      .then(function (j) {
        if (j && Array.isArray(j.iceServers) && j.iceServers.length) {
          stunTurnServers = j.iceServers;
        }
        setConnectionStatus('Connecting...');
        openWebSocket();
      })
      .catch(function () {
        setConnectionStatus('Connecting...');
        openWebSocket();
      });
  }

  var applyCatalogBtn = document.getElementById('applyCatalogStreams');
  if (applyCatalogBtn) {
    applyCatalogBtn.addEventListener('click', function () {
      startRtc(rtcRecvonlyVideoLineCount()).catch(function (err) {
        setConnectionStatus('WebRTC error: ' + (err && err.message ? err.message : String(err)));
      });
    });
  }

  initVirtualGamepad();
  setInterval(updateControllerDebugPanel, 250);

  boot();
})();
