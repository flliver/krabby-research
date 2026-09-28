export type DeviceSummary = {
  thingName: string;
  connected: boolean;
  connectivityTimestamp: number | null;
  reported: Record<string, unknown>;
};

export type DeviceDetail = {
  thingName: string;
  thingTypeName: string | null;
  attributes: Record<string, string>;
  connected: boolean;
  connectivityTimestamp: number | null;
  reported: Record<string, unknown>;
};

export type SshTunnelResponse = {
  tunnelId: string;
  sourceAccessToken: string;
  region: string;
};

function fleetBaseUrl(): string {
  const base = process.env.FLEET_SERVICE_URL?.replace(/\/$/, "");
  if (!base) {
    throw new Error("FLEET_SERVICE_URL is not set");
  }
  return base;
}

async function fleetFetch<T>(
  path: string,
  accessToken: string,
  init?: RequestInit,
): Promise<T> {
  const resp = await fetch(`${fleetBaseUrl()}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (resp.status === 401) {
    throw new Error("Fleet API returned 401 — sign in again");
  }
  if (resp.status === 403) {
    throw new Error("Not in the operator group");
  }
  if (resp.status === 404) {
    throw new Error("Not found");
  }
  if (!resp.ok) {
    throw new Error(`Fleet API ${resp.status}: ${await resp.text()}`);
  }
  if (resp.status === 204) {
    return undefined as T;
  }
  return (await resp.json()) as T;
}

export function listDevices(accessToken: string): Promise<DeviceSummary[]> {
  return fleetFetch<DeviceSummary[]>("/devices", accessToken);
}

export function getDevice(thingName: string, accessToken: string): Promise<DeviceDetail> {
  return fleetFetch<DeviceDetail>(`/devices/${encodeURIComponent(thingName)}`, accessToken);
}

export function openSshTunnel(
  thingName: string,
  accessToken: string,
): Promise<SshTunnelResponse> {
  return fleetFetch<SshTunnelResponse>(
    `/devices/${encodeURIComponent(thingName)}/ssh-tunnel`,
    accessToken,
    { method: "POST" },
  );
}

export function closeSshTunnel(
  thingName: string,
  tunnelId: string,
  accessToken: string,
): Promise<void> {
  return fleetFetch<void>(
    `/devices/${encodeURIComponent(thingName)}/ssh-tunnel/${encodeURIComponent(tunnelId)}`,
    accessToken,
    { method: "DELETE" },
  );
}
