"use client";

import * as React from "react";
import {
  CheckCircle2Icon,
  FilmIcon,
  VideoIcon,
  PlayIcon,
  Settings2Icon,
  LoaderCircleIcon,
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
): {
  label: string;
  variant: "default" | "secondary" | "destructive" | "outline";
  completeCount: number;
  isComplete: boolean;
  isProcessing: boolean;
} {
  const videoVariants = variants.filter((v) => v.sourceVideoId === video.id);
  const completeCount = videoVariants.filter((v) => v.status === "complete").length;
  const hasComplete = completeCount > 0;
  const hasProcessing = videoVariants.some((v) => v.status === "processing");

  if (pipelineState.videoId === video.id && pipelineState.status === "running") {
    return { label: "Running", variant: "secondary", completeCount, isComplete: hasComplete, isProcessing: true };
  }
  if (hasProcessing) return { label: "Running", variant: "secondary", completeCount, isComplete: hasComplete, isProcessing: true };
  if (hasComplete) return { label: "Processed", variant: "default", completeCount, isComplete: true, isProcessing: false };
  return { label: "New", variant: "outline", completeCount: 0, isComplete: false, isProcessing: false };
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
                const variantsLabel = status.completeCount === 1
                  ? "1 processed variant"
                  : `${status.completeCount} processed variants`;
                const itemClassName = [
                  "h-auto py-2",
                  isActive ? "border-l-2 border-primary bg-sidebar-accent/80 pl-1.5" : "",
                  status.isComplete && !isActive ? "border-l-2 border-primary/70 bg-primary/10 pl-1.5 hover:bg-primary/15" : "",
                ].filter(Boolean).join(" ");

                return (
                  <SidebarMenuItem key={video.id}>
                    <SidebarMenuButton
                      isActive={isActive}
                      onClick={() => onSelectVideo(video.id)}
                      tooltip={video.title}
                      className={itemClassName}
                    >
                      {status.isProcessing ? (
                        <LoaderCircleIcon className="mt-0.5 shrink-0 animate-spin text-primary" />
                      ) : status.isComplete ? (
                        <CheckCircle2Icon className="mt-0.5 shrink-0 text-primary" />
                      ) : (
                        <VideoIcon className="mt-0.5 shrink-0" />
                      )}
                      <div className="flex flex-col min-w-0">
                        <span className="text-sm leading-snug font-medium">{video.title}</span>
                        <span className="text-[10px] text-muted-foreground font-mono">{video.id}</span>
                        {status.isComplete ? (
                          <span className="mt-0.5 text-[10px] font-medium text-primary">
                            {variantsLabel}
                          </span>
                        ) : null}
                      </div>
                    </SidebarMenuButton>
                    <SidebarMenuBadge>
                      <Badge variant={status.variant} className="text-[9px] px-1.5 py-0 leading-tight">
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
