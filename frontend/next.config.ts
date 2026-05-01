import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    proxyTimeout: 3_600_000, // 1 hour: long interviews can spend a while in TTS
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://api:8080/api/:path*",
      },
    ];
  },
};

export default nextConfig;
