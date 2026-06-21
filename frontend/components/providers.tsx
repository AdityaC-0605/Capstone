"use client";

import { useEffect } from "react";
import { MotionConfig } from "framer-motion";

import { ToastViewport } from "@/components/toast-viewport";
import { usePulseStore } from "@/store/use-pulse-store";

export function Providers({ children }: { children: React.ReactNode }) {
  const hydrateSessionSustainability = usePulseStore(
    (state) => state.hydrateSessionSustainability,
  );

  useEffect(() => {
    hydrateSessionSustainability();
  }, [hydrateSessionSustainability]);

  // `reducedMotion="user"` makes every framer-motion animation respect the
  // operating system's "reduce motion" accessibility preference.
  return (
    <MotionConfig reducedMotion="user">
      {children}
      <ToastViewport />
    </MotionConfig>
  );
}
