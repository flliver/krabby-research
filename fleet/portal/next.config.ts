import type { NextConfig } from "next";

const fleetServiceUrl = (process.env.FLEET_SERVICE_URL || "http://127.0.0.1:8080").replace(
  /\/$/,
  "",
);

const nextConfig: NextConfig = {
  output: "standalone",
  // Local `next dev` (no Caddy): browser Open SSH calls hit /api/devices/* and
  // are rewritten to the fleet service. In production Caddy already routes
  // /api/* (except /api/auth/*) to the service, so these rewrites are unused.
  async rewrites() {
    return [
      {
        source: "/api/devices/:path*",
        destination: `${fleetServiceUrl}/devices/:path*`,
      },
      {
        source: "/api/teleop/:path*",
        destination: `${fleetServiceUrl}/teleop/:path*`,
      },
    ];
  },
};

export default nextConfig;
