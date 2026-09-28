export function formatLastSeen(connectivityTimestamp: number | null | undefined): string {
  if (connectivityTimestamp == null) {
    return "unknown";
  }
  const seconds =
    connectivityTimestamp > 10_000_000_000
      ? connectivityTimestamp / 1000
      : connectivityTimestamp;
  return new Date(seconds * 1000).toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

export function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

export function localproxyFallback(sourceAccessToken: string, region: string, localPort = 5555): string {
  return (
    `localproxy -s localhost:${localPort} -t '${sourceAccessToken}' -r ${region}`
  );
}
