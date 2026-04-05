"use client";

import { useEffect } from "react";

import { usePulseStore } from "@/store/use-pulse-store";

export function Providers({ children }: { children: React.ReactNode }) {
  const hydrateSessionSustainability = usePulseStore(
    (state) => state.hydrateSessionSustainability,
  );

  useEffect(() => {
    hydrateSessionSustainability();
  }, [hydrateSessionSustainability]);

  return children;
}
