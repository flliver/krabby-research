"use client";

import { useSession } from "next-auth/react";
import { useEffect, useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/** Cognito-gated launcher: hands the access token to the existing teleop viewer (signaling + ICE URLs only). */
export default function DeviceTeleopPage() {
  const params = useParams();
  const raw = params?.thingName;
  const thingName = decodeURIComponent(typeof raw === "string" ? raw : Array.isArray(raw) ? raw[0] : "");
  const { data: session, status } = useSession();

  const viewerUrl = useMemo(() => {
    if (!thingName || !session?.accessToken) {
      return null;
    }
    const u = new URL("/teleop/viewer.html", window.location.origin);
    u.searchParams.set("thing", thingName);
    u.searchParams.set("token", session.accessToken);
    return u.toString();
  }, [thingName, session?.accessToken]);

  useEffect(() => {
    if (viewerUrl) {
      window.location.replace(viewerUrl);
    }
  }, [viewerUrl]);

  if (status === "loading") {
    return (
      <main>
        <p className="muted">Checking session…</p>
      </main>
    );
  }

  if (!session?.accessToken) {
    return (
      <main>
        <p className="error">Not signed in.</p>
        <p>
          <Link href="/login">Sign in</Link>
        </p>
      </main>
    );
  }

  return (
    <main>
      <p className="muted">
        Opening teleop for <span className="mono">{thingName}</span>…
      </p>
      <p>
        <Link href={`/devices/${encodeURIComponent(thingName)}`}>← Device</Link>
        {" · "}
        {viewerUrl ? (
          <a href={viewerUrl} target="_blank" rel="noreferrer">
            Open viewer
          </a>
        ) : null}
      </p>
    </main>
  );
}
