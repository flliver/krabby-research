import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { auth } from "@/auth";
import { OpenSshButton } from "@/components/OpenSshButton";
import { OpenTeleopLink } from "@/components/OpenTeleopLink";
import { getDevice } from "@/lib/fleet";
import { asRecord, asStringList, formatLastSeen } from "@/lib/format";

export default async function DeviceDetailPage({
  params,
}: {
  params: Promise<{ thingName: string }>;
}) {
  const session = await auth();
  if (!session?.accessToken) {
    redirect("/login");
  }

  const { thingName: rawName } = await params;
  const thingName = decodeURIComponent(rawName);

  let device;
  try {
    device = await getDevice(thingName, session.accessToken);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (message === "Not found") {
      notFound();
    }
    return (
      <main>
        <p>
          <Link href="/devices">← Devices</Link>
        </p>
        <h1 className="mono">{thingName}</h1>
        <p className="error">{message}</p>
      </main>
    );
  }

  const reported = device.reported ?? {};
  const health = asRecord(reported.health);
  const imu = asRecord(reported.imu);
  const pose = asRecord(reported.pose);
  const power = asRecord(reported.power);
  const redFlags = asStringList(reported.red_flags);

  return (
    <main>
      <p>
        <Link href="/devices">← Devices</Link>
      </p>
      <h1 className="mono">{device.thingName}</h1>
      <p>
        <span className={`badge ${device.connected ? "online" : "offline"}`}>
          {device.connected ? "online" : "offline"}
        </span>{" "}
        <span className="muted">
          last seen {formatLastSeen(device.connectivityTimestamp)}
        </span>
      </p>

      <div className="actions" style={{ marginBottom: "1rem" }}>
        <OpenTeleopLink thingName={device.thingName} />
      </div>

      <OpenSshButton thingName={device.thingName} />

      <div className="panel">
        <h2>Reported image</h2>
        <p className="mono">
          {typeof reported.reported_image === "string" ? reported.reported_image : "—"}
        </p>
      </div>

      <div className="panel">
        <h2>Health</h2>
        {health ? (
          <dl className="kv">
            {Object.entries(health).map(([key, value]) => (
              <div key={key} style={{ display: "contents" }}>
                <dt>{key}</dt>
                <dd className="mono">{String(value)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="muted">No health fields in latest shadow.</p>
        )}
      </div>

      <div className="panel">
        <h2>IMU / pose</h2>
        {imu || pose ? (
          <dl className="kv">
            {imu
              ? Object.entries(imu).map(([key, value]) => (
                  <div key={`imu-${key}`} style={{ display: "contents" }}>
                    <dt>imu.{key}</dt>
                    <dd className="mono">{JSON.stringify(value)}</dd>
                  </div>
                ))
              : null}
            {pose
              ? Object.entries(pose).map(([key, value]) => (
                  <div key={`pose-${key}`} style={{ display: "contents" }}>
                    <dt>pose.{key}</dt>
                    <dd className="mono">{JSON.stringify(value)}</dd>
                  </div>
                ))
              : null}
          </dl>
        ) : (
          <p className="muted">No IMU/pose in latest shadow (HAL may be down).</p>
        )}
      </div>

      <div className="panel">
        <h2>Power</h2>
        {power ? (
          <pre className="pre">{JSON.stringify(power, null, 2)}</pre>
        ) : (
          <p className="muted">No power readings.</p>
        )}
      </div>

      <div className="panel">
        <h2>Red flags</h2>
        {redFlags.length > 0 ? (
          <div className="flags">
            {redFlags.map((flag) => (
              <span className="flag" key={flag}>
                {flag}
              </span>
            ))}
          </div>
        ) : (
          <p className="muted">None.</p>
        )}
      </div>

      <div className="panel">
        <h2>Raw reported</h2>
        <pre className="pre">{JSON.stringify(reported, null, 2)}</pre>
      </div>
    </main>
  );
}
