import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    proxyTimeout: 600_000, // 10 minutes — TTS can take several minutes on CPU
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
