import type { NextConfig } from "next";
import path from "path";
import { fileURLToPath } from "url";

// Resolve the project root deterministically for Turbopack module lookups.
const frontendRoot = path.dirname(fileURLToPath(import.meta.url));
const tailwindPkgPath = path.join(frontendRoot, "node_modules", "tailwindcss");

const nextConfig: NextConfig = {
  outputFileTracingRoot: frontendRoot,
  turbopack: {
    root: frontendRoot,
    resolveAlias: {
      tailwindcss: tailwindPkgPath,
    },
  },
  webpack: (config) => {
    config.resolve.alias = {
      ...(config.resolve.alias ?? {}),
      tailwindcss: tailwindPkgPath,
    };

    return config;
  },
};

export default nextConfig;
