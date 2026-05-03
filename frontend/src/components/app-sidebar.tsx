"use client";

import * as React from "react";
import {
  FilmIcon,
  VideoIcon,
  PlayIcon,
  Settings2Icon,
} from "lucide-react";
import { SettingsDialog } from "./settings-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import type { StudioSettings, Video, PipelineState, VideoVariant } from "@/lib/types";

function getVideoStatus(
  video: Video,
  pipelineState: PipelineState,
  variants: VideoVariant[]
): { label: string; variant: "default" | "secondary" | "destructive" | "outline" } {
  const videoVariants = variants.filter((v) => v.sourceVideoId === video.id);
  const hasComplete = videoVariants.some((v) => v.status === "complete");
  const hasProcessing = videoVariants.some((v) => v.status === "processing");

  if (pipelineState.videoId === video.id && pipelineState.status === "running") {
    return { label: "Running", variant: "secondary" };
  }
  if (hasProcessing) return { label: "Running", variant: "secondary" };
  if (hasComplete) return { label: "Done", variant: "default" };
  return { label: "New", variant: "outline" };
}

interface AppSidebarProps extends React.ComponentProps<typeof Sidebar> {
  videos: Video[];
  selectedVideoId: string | null;
  settings: StudioSettings;
  onSelectVideo: (videoId: string) => void;
  pipelineState: PipelineState;
  onStartPipeline: () => void;
}

export function AppSidebar({
  videos,
  selectedVideoId,
  settings,
  onSelectVideo,
  pipelineState,
  onStartPipeline,
  ...props
}: AppSidebarProps) {
  const dubbingLabel = settings.dubbing.includes("aligned") ? "Aligned" : "Baseline";
  const diarizationLabel = settings.diarization.includes("pyannote") ? "Diarization on" : "Diarization off";
  const voiceLabel = settings.voiceCloning.includes("chatterbox") ? "Voice cloning on" : "Voice cloning off";

  return (
    <Sidebar {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" render={<div />}>
              <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                <FilmIcon className="size-4" />
              </div>
              <div className="flex flex-col gap-0.5 leading-none">
                <span className="font-semibold">Foreign Whispers</span>
                <span className="text-xs">Dubbing Studio</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        {/* Video Library */}
        <SidebarGroup>
          <SidebarGroupLabel>Video Library</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {videos.map((video) => {
                const isActive = video.id === selectedVideoId;
                const status = getVideoStatus(video, pipelineState, pipelineState.variants);

                return (
                  <SidebarMenuItem key={video.id}>
                    <SidebarMenuButton
                      isActive={isActive}
                      onClick={() => onSelectVideo(video.id)}
                      tooltip={video.title}
                      className={`h-auto py-1.5 ${isActive ? "border-l-2 border-primary bg-sidebar-accent/80 pl-1.5" : ""}`}
                    >
                      <VideoIcon className="mt-0.5 shrink-0" />
                      <div className="flex flex-col min-w-0">
                        <span className="text-sm leading-snug">{video.title}</span>
                        <span className="text-[10px] text-muted-foreground font-mono">{video.id}</span>
                      </div>
                    </SidebarMenuButton>
                    <SidebarMenuBadge>
                      <Badge variant={status.variant} className="text-[9px] px-1 py-0 leading-tight">
                        {status.label}
                      </Badge>
                    </SidebarMenuBadge>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

      </SidebarContent>

      <SidebarFooter>
        <div className="rounded-md border border-sidebar-border/60 bg-sidebar-accent/30 p-2.5">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium">
            <Settings2Icon className="size-3.5" />
            Pipeline settings
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Badge variant="outline" className="text-[10px]">{dubbingLabel}</Badge>
            <Badge variant={settings.diarization.includes("pyannote") ? "secondary" : "outline"} className="text-[10px]">
              {diarizationLabel}
            </Badge>
            <Badge variant={settings.voiceCloning.includes("chatterbox") ? "secondary" : "outline"} className="text-[10px]">
              {voiceLabel}
            </Badge>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            className="flex-1"
            onClick={onStartPipeline}
            disabled={pipelineState.status === "running"}
          >
            <PlayIcon className="size-3.5 mr-1.5" />
            {pipelineState.status === "running" ? "Processing..." : "Start Pipeline"}
          </Button>
          <SettingsDialog />
        </div>
        <div className="text-center text-[10px] text-muted-foreground/60 pb-1">
          Aegean AI Inc.
        </div>
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
