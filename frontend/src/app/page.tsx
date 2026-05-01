import { StudioLayout } from "@/components/studio-layout";
import type { Video, VideoVariant } from "@/lib/types";

const API_URL = process.env.API_URL || "http://localhost:8080";

export default async function Home() {
  const [videosRes, variantsRes] = await Promise.all([
    fetch(`${API_URL}/api/videos`, { cache: "no-store" }),
    fetch(`${API_URL}/api/variants`, { cache: "no-store" }),
  ]);
  const videos: Video[] = videosRes.ok ? await videosRes.json() : [];
  const variants: VideoVariant[] = variantsRes.ok ? await variantsRes.json() : [];

  return <StudioLayout videos={videos} initialVariants={variants} />;
}
