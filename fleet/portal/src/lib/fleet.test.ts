import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { closeSshTunnel, getDevice, listDevices, openSshTunnel } from "./fleet";

const BASE = "http://fleet.test/api";

function mockFetch(body: unknown, status = 200, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  });
}

describe("fleet API client", () => {
  beforeEach(() => {
    process.env.FLEET_SERVICE_URL = BASE;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.FLEET_SERVICE_URL;
  });

  it("listDevices GET /devices with Bearer token", async () => {
    const payload = [
      {
        thingName: "bench-krabby-ci",
        connected: true,
        connectivityTimestamp: 1_700_000_000,
        reported: { timestamp: 1_700_000_000 },
      },
    ];
    const fetchMock = mockFetch(payload);
    vi.stubGlobal("fetch", fetchMock);

    const devices = await listDevices("operator-token");

    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/devices`,
      expect.objectContaining({
        headers: { Authorization: "Bearer operator-token" },
        cache: "no-store",
      }),
    );
    expect(devices).toEqual(payload);
  });

  it("getDevice GET /devices/{thing} with encoded thing name", async () => {
    const payload = {
      thingName: "bench/krabby",
      thingTypeName: "Krab",
      attributes: {},
      connected: true,
      connectivityTimestamp: 1_700_000_000,
      reported: { timestamp: 1_700_000_000 },
    };
    const fetchMock = mockFetch(payload);
    vi.stubGlobal("fetch", fetchMock);

    const device = await getDevice("bench/krabby", "tok");

    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/devices/bench%2Fkrabby`,
      expect.objectContaining({
        headers: { Authorization: "Bearer tok" },
      }),
    );
    expect(device.thingName).toBe("bench/krabby");
  });

  it("openSshTunnel POSTs to ssh-tunnel and returns tunnel credentials", async () => {
    const payload = {
      tunnelId: "tunnel-abc",
      sourceAccessToken: "source-tok",
      region: "us-east-2",
    };
    const fetchMock = mockFetch(payload);
    vi.stubGlobal("fetch", fetchMock);

    const tunnel = await openSshTunnel("bench-krabby-ci", "tok");

    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/devices/bench-krabby-ci/ssh-tunnel`,
      expect.objectContaining({
        method: "POST",
        headers: { Authorization: "Bearer tok" },
      }),
    );
    expect(tunnel).toEqual(payload);
  });

  it("closeSshTunnel DELETEs tunnel and accepts 204", async () => {
    const fetchMock = mockFetch(undefined, 204);
    vi.stubGlobal("fetch", fetchMock);

    await closeSshTunnel("bench-krabby-ci", "tunnel-abc", "tok");

    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE}/devices/bench-krabby-ci/ssh-tunnel/tunnel-abc`,
      expect.objectContaining({
        method: "DELETE",
        headers: { Authorization: "Bearer tok" },
      }),
    );
  });

  it("maps fleet auth errors to operator-facing messages", async () => {
    vi.stubGlobal("fetch", mockFetch("", 401, false));
    await expect(listDevices("bad")).rejects.toThrow("sign in again");

    vi.stubGlobal("fetch", mockFetch("", 403, false));
    await expect(listDevices("bad")).rejects.toThrow("operator group");

    vi.stubGlobal("fetch", mockFetch("", 404, false));
    await expect(getDevice("missing", "tok")).rejects.toThrow("Not found");
  });

  it("requires FLEET_SERVICE_URL", async () => {
    delete process.env.FLEET_SERVICE_URL;
    vi.stubGlobal("fetch", mockFetch([]));
    await expect(listDevices("tok")).rejects.toThrow("FLEET_SERVICE_URL is not set");
  });
});
