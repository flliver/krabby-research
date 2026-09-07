#!/usr/bin/env python3
"""Serve the interactive OLED simulator with automatic reloads."""
from __future__ import annotations

import http.server
import importlib
import io
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ssd1306  # noqa: E402
import krab      # noqa: E402
import viewer    # noqa: E402

# Browser reload inputs.
_WATCH = [HERE / "krab.py", HERE / "ssd1306.py", HERE / "viewer.py"]
_WATCH += sorted((HERE / "native").rglob("*"))
_WATCH += sorted((HERE.parent / "arduino" / "src" / "display").rglob("*"))

_LIVE_JS = """
<script>
(function () {
  let last = null;
  async function tick() {
    try {
      const t = await (await fetch('/mtime', {cache: 'no-store'})).text();
      if (last !== null && t !== last) return location.reload();
      last = t;
    } catch (e) { /* server restarting; ignore */ }
    setTimeout(tick, 700);
  }
  tick();
})();
</script>
"""


def _mtime() -> str:
    return str(max((f.stat().st_mtime for f in _WATCH if f.exists()), default=0))


def _render_page() -> bytes:
    """Reload the simulator modules and rebuild the interactive page."""
    importlib.reload(ssd1306)
    importlib.reload(krab)
    importlib.reload(viewer)
    try:
        html = viewer.build()
    except Exception:
        # Show render failures in the browser.
        tb = traceback.format_exc()
        html = (f"<pre style='color:#f66;background:#0a0e12;padding:24px;"
                f"font:13px ui-monospace,monospace;white-space:pre-wrap'>"
                f"render error — fix krab.py and save:\n\n{tb}</pre>")
    return ("<!doctype html><meta charset='utf-8'>" + _LIVE_JS + html).encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype="text/html; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/mtime"):
            self._send(_mtime().encode(), "text/plain")
        else:
            self._send(_render_page())

    def do_POST(self):
        if self.path != "/render":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            state = krab.KrabState.from_payload(payload)
            frame = krab.render(state)
            pixels = [
                [x, y]
                for y, row in enumerate(frame.to_rows())
                for x, cell in enumerate(row)
                if cell == "#"
            ]
            body = json.dumps({"pixels": pixels}).encode()
            self._send(body, "application/json")
        except Exception as error:
            body = json.dumps({"error": str(error)}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        pass  # quiet


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"krab render viewer -> {url}")
    print("adjust the browser controls; source edits auto-reload. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
