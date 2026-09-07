"""Single-screen interactive browser UI for the OLED simulator."""
from __future__ import annotations

import json

from krab import KrabState


def default_state() -> KrabState:
    return KrabState()


def build() -> str:
    state = default_state()
    initial = {
        "role": state.role,
        "roll": state.roll,
        "pitch": state.pitch,
        "imu_valid": state.imu_valid,
        "battery_volts": list(state.battery_volts),
        "front": state.front,
        "left": state.left,
        "right": state.right,
        "legs": state.legs,
    }
    return _PAGE.replace("__INITIAL_STATE__", json.dumps(initial))


_PAGE = r"""
<title>Krab OLED simulator</title>
<main>
  <header>
    <div>
      <h1>Krab OLED simulator</h1>
      <p>One simulated robot state through the production model builder and renderer.</p>
    </div>
    <button id="reset" type="button">Reset</button>
  </header>

  <section class="screen-card">
    <div class="bezel"><canvas id="screen" width="640" height="320"></canvas></div>
    <p id="status">Rendering…</p>
  </section>

  <section class="controls">
    <fieldset>
      <legend>System</legend>
      <label>Role
        <select id="role">
          <option>FRONT</option><option>LEFT</option><option>RIGHT</option><option>UNKWN</option>
        </select>
      </label>
      <label class="check"><input id="front" type="checkbox"> FRONT present</label>
      <label class="check"><input id="left" type="checkbox"> LEFT telemetry present</label>
      <label class="check"><input id="right" type="checkbox"> RIGHT telemetry present</label>
    </fieldset>

    <fieldset>
      <legend>IMU</legend>
      <label class="check"><input id="imu_valid" type="checkbox"> Reading valid</label>
      <label>Roll <output id="roll_value"></output>
        <input id="roll" type="range" min="-99" max="99" step="1">
      </label>
      <label>Pitch <output id="pitch_value"></output>
        <input id="pitch" type="range" min="-89" max="89" step="1">
      </label>
    </fieldset>

    <fieldset>
      <legend>Power</legend>
      <label>Battery A <output id="battery_a_value"></output>
        <input id="battery_a" type="range" min="10" max="14.6" step="0.1">
      </label>
      <label>Battery B <output id="battery_b_value"></output>
        <input id="battery_b" type="range" min="10" max="14.6" step="0.1">
      </label>
    </fieldset>
  </section>

  <section class="joints-card">
    <div class="section-title">
      <h2>Actuators</h2>
      <p>Missing LEFT or RIGHT telemetry overrides that board’s six actuator controls.</p>
    </div>
    <div id="joints"></div>
  </section>
</main>

<script>
const initial = __INITIAL_STATE__;
const legNames = ["FL", "FR", "ML", "MR", "RL", "RR"];
const jointNames = ["Yaw", "Hip", "Knee"];
const glyphs = ["hold", "extend", "retract", "disc", "unverified"];
const scale = 5;
let timer = null;
let generation = 0;

function jointControls() {
  const host = document.getElementById("joints");
  for (let leg = 0; leg < legNames.length; leg++) {
    const row = document.createElement("div");
    row.className = "joint-row";
    row.innerHTML = `<strong>${legNames[leg]}</strong>`;
    for (let joint = 0; joint < jointNames.length; joint++) {
      const label = document.createElement("label");
      label.textContent = jointNames[joint];
      const select = document.createElement("select");
      select.dataset.leg = leg;
      select.dataset.joint = joint;
      select.className = "joint";
      for (const glyph of glyphs) {
        const option = document.createElement("option");
        option.value = glyph;
        option.textContent = glyph;
        select.appendChild(option);
      }
      label.appendChild(select);
      row.appendChild(label);
    }
    host.appendChild(row);
  }
}

function applyState(state) {
  for (const key of ["role", "roll", "pitch", "front", "left", "right", "imu_valid"])
    document.getElementById(key)[document.getElementById(key).type === "checkbox" ? "checked" : "value"] = state[key];
  document.getElementById("battery_a").value = state.battery_volts[0];
  document.getElementById("battery_b").value = state.battery_volts[1];
  document.querySelectorAll("select.joint").forEach(select => {
    select.value = state.legs[Number(select.dataset.leg)][Number(select.dataset.joint)];
  });
  updateOutputs();
}

function readState() {
  const legs = Array.from({length: 6}, () => Array(3));
  document.querySelectorAll("select.joint").forEach(select => {
    legs[Number(select.dataset.leg)][Number(select.dataset.joint)] = select.value;
  });
  return {
    role: document.getElementById("role").value,
    roll: Number(document.getElementById("roll").value),
    pitch: Number(document.getElementById("pitch").value),
    imu_valid: document.getElementById("imu_valid").checked,
    battery_volts: [Number(document.getElementById("battery_a").value), Number(document.getElementById("battery_b").value)],
    front: document.getElementById("front").checked,
    left: document.getElementById("left").checked,
    right: document.getElementById("right").checked,
    legs,
  };
}

function updateOutputs() {
  document.getElementById("roll_value").textContent = `${document.getElementById("roll").value}°`;
  document.getElementById("pitch_value").textContent = `${document.getElementById("pitch").value}°`;
  document.getElementById("battery_a_value").textContent = `${Number(document.getElementById("battery_a").value).toFixed(1)} V`;
  document.getElementById("battery_b_value").textContent = `${Number(document.getElementById("battery_b").value).toFixed(1)} V`;
}

function paint(points) {
  const canvas = document.getElementById("screen");
  const context = canvas.getContext("2d");
  context.shadowBlur = 0;
  context.fillStyle = "#05080b";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.shadowColor = "#86e8ff";
  context.shadowBlur = 3;
  context.fillStyle = "#86e8ff";
  for (const [x, y] of points)
    context.fillRect(x * scale, y * scale, scale, scale);
}

async function render() {
  const mine = ++generation;
  const status = document.getElementById("status");
  try {
    const response = await fetch("/render", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(readState()),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (mine !== generation) return;
    paint(result.pixels);
    status.textContent = "128×64 · production state model + renderer";
    status.className = "";
  } catch (error) {
    if (mine !== generation) return;
    status.textContent = error.message;
    status.className = "error";
  }
}

function schedule() {
  updateOutputs();
  clearTimeout(timer);
  timer = setTimeout(render, 70);
}

jointControls();
applyState(initial);
document.querySelectorAll("input, select").forEach(control => control.addEventListener("input", schedule));
document.getElementById("reset").addEventListener("click", () => { applyState(initial); render(); });
render();
</script>

<style>
:root { --bg:#070a0d; --card:#11171c; --line:#26323b; --ink:#d5e0e7; --mute:#71828d; --cyan:#86e8ff; --amber:#e6a545; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:13px ui-monospace,"SF Mono",Menlo,monospace; }
main { width:min(1060px, calc(100% - 32px)); margin:0 auto; padding:28px 0 48px; }
header { display:flex; justify-content:space-between; gap:24px; align-items:start; margin-bottom:18px; }
h1,h2,p { margin:0; }
h1 { font:650 21px ui-sans-serif,system-ui,sans-serif; }
h2 { font:650 15px ui-sans-serif,system-ui,sans-serif; }
header p,.section-title p { color:var(--mute); margin-top:6px; line-height:1.45; }
button,select,input { accent-color:var(--cyan); }
button,select { color:var(--ink); background:#182128; border:1px solid var(--line); border-radius:5px; padding:6px 8px; font:inherit; }
button { cursor:pointer; }
.screen-card,.joints-card,fieldset { background:var(--card); border:1px solid var(--line); border-radius:9px; }
.screen-card { padding:14px; }
.bezel { background:#161d24; border:1px solid #293741; border-radius:8px; padding:13px; max-width:668px; margin:auto; }
canvas { width:100%; height:auto; image-rendering:pixelated; display:block; border-radius:3px; background:#05080b; }
#status { max-width:668px; margin:9px auto 0; color:var(--mute); font-size:11px; }
#status.error { color:#ff8080; }
.controls { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:14px 0; }
fieldset { min-width:0; margin:0; padding:13px; }
legend { color:var(--cyan); padding:0 5px; }
fieldset label { display:block; margin:9px 0; color:var(--mute); }
fieldset label output { float:right; color:var(--ink); }
fieldset input[type=range] { width:100%; margin:7px 0 0; }
fieldset .check { color:var(--ink); }
.joints-card { padding:15px; }
.section-title { margin-bottom:12px; }
#joints { display:grid; grid-template-columns:repeat(2,1fr); gap:8px 18px; }
.joint-row { display:grid; grid-template-columns:32px repeat(3,1fr); gap:8px; align-items:end; padding:8px 0; border-top:1px solid #1c262d; }
.joint-row strong { color:var(--cyan); align-self:center; }
.joint-row label { color:var(--mute); font-size:10px; }
.joint-row select { display:block; width:100%; margin-top:4px; font-size:11px; }
@media (max-width:760px) { .controls,#joints { grid-template-columns:1fr; } }
</style>
"""


if __name__ == "__main__":
    print("Run firmware/oled_sim/serve.py and open its URL.")
