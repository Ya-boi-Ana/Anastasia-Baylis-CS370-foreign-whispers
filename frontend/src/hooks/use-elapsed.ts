"use client";

import { useEffect, useState } from "react";

/**
 * Returns a live elapsed-ms counter that ticks every second while
 * `startedAt` is truthy. Returns `undefined` when inactive.
 */
export function useElapsed(startedAt: number | undefined): number | undefined {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!startedAt) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  return startedAt ? now - startedAt : undefined;
}
