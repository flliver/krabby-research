import Link from "next/link";

/** Opens the Cognito-gated teleop route in a new tab (existing teleop viewer + fleet signaling/ICE). */
export function OpenTeleopLink({
  thingName,
  className,
}: {
  thingName: string;
  className?: string;
}) {
  return (
    <Link
      className={className ?? "button primary"}
      href={`/devices/${encodeURIComponent(thingName)}/teleop`}
      target="_blank"
      rel="noreferrer"
    >
      Open teleop
    </Link>
  );
}
