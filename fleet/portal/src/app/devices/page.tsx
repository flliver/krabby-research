import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/auth";
import { OpenTeleopLink } from "@/components/OpenTeleopLink";
import { listDevices, type DeviceSummary } from "@/lib/fleet";
import { formatLastSeen } from "@/lib/format";

export default async function DevicesPage() {
  const session = await auth();
  if (!session?.accessToken) {
    redirect("/login");
  }

  let devices: DeviceSummary[] = [];
  let error: string | null = null;
  try {
    devices = await listDevices(session.accessToken);
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }

  return (
    <main>
      <h1>Devices</h1>
      <p className="muted">Fleet robots from IoT registry + latest 1/min shadow telemetry.</p>

      {error ? <p className="error">{error}</p> : null}

      {!error && devices.length === 0 ? (
        <p className="muted">No devices found.</p>
      ) : null}

      {devices.length > 0 ? (
        <table className="devices">
          <thead>
            <tr>
              <th>Thing</th>
              <th>Status</th>
              <th>Last seen</th>
              <th>Image</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {devices.map((device) => {
              const image =
                typeof device.reported?.reported_image === "string"
                  ? device.reported.reported_image
                  : "—";
              return (
                <tr key={device.thingName}>
                  <td>
                    <Link className="mono" href={`/devices/${encodeURIComponent(device.thingName)}`}>
                      {device.thingName}
                    </Link>
                  </td>
                  <td>
                    <span className={`badge ${device.connected ? "online" : "offline"}`}>
                      {device.connected ? "online" : "offline"}
                    </span>
                  </td>
                  <td className="mono muted">{formatLastSeen(device.connectivityTimestamp)}</td>
                  <td className="mono muted">{image}</td>
                  <td>
                    <OpenTeleopLink thingName={device.thingName} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}
    </main>
  );
}
