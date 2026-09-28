"use client";

import { useSession } from "next-auth/react";
import { useState } from "react";
import { localproxyFallback } from "@/lib/format";

type TunnelInfo = {
  tunnelId: string;
  sourceAccessToken: string;
  region: string;
};

export function OpenSshButton({ thingName }: { thingName: string }) {
  const { data: session } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tunnel, setTunnel] = useState<TunnelInfo | null>(null);

  async function openTunnel() {
    if (!session?.accessToken) {
      setError("Not signed in — refresh and try again");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/api/devices/${encodeURIComponent(thingName)}/ssh-tunnel`, {
        method: "POST",
        headers: { Authorization: `Bearer ${session.accessToken}` },
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(
          typeof body.detail === "string"
            ? body.detail
            : body.error || `Open tunnel failed (${resp.status})`,
        );
      }
      setTunnel(body as TunnelInfo);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function closeTunnel() {
    if (!tunnel || !session?.accessToken) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(
        `/api/devices/${encodeURIComponent(thingName)}/ssh-tunnel/${encodeURIComponent(tunnel.tunnelId)}`,
        {
          method: "DELETE",
          headers: { Authorization: `Bearer ${session.accessToken}` },
        },
      );
      if (!resp.ok && resp.status !== 204) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || body.error || `Close tunnel failed (${resp.status})`);
      }
      setTunnel(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="actions">
        <button className="primary" type="button" disabled={loading} onClick={openTunnel}>
          {loading && !tunnel ? "Opening…" : "Open SSH"}
        </button>
        {tunnel ? (
          <button type="button" disabled={loading} onClick={closeTunnel}>
            Force-close tunnel
          </button>
        ) : null}
      </div>

      {error ? <p className="error">{error}</p> : null}

      {tunnel ? (
        <div className="panel">
          <h2>SSH session</h2>
          <p className="muted">
            Preferred: run the CLI on your local computer (handles localproxy + cleanup). The browser
            never receives AWS credentials — only a short-lived source tunnel token.
          </p>
          <p className="mono">krabby-fleet ssh {thingName}</p>
          <p className="muted" style={{ marginTop: "1rem" }}>
            Fallback (localproxy already installed; pick a free local port):
          </p>
          <pre className="pre">{localproxyFallback(tunnel.sourceAccessToken, tunnel.region)}</pre>
          <p className="muted" style={{ marginTop: "0.75rem" }}>
            Then: <span className="mono">ssh operator@localhost -p 5555</span>
          </p>
          <p className="muted">
            Tunnel id: <span className="mono">{tunnel.tunnelId}</span>
          </p>
        </div>
      ) : null}
    </div>
  );
}
