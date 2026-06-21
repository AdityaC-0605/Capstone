"use client";

import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";

import { cn } from "@/lib/utils";
import { useToastStore, type Toast } from "@/store/use-toast-store";

const ICONS = {
  success: CheckCircle2,
  error: AlertTriangle,
  info: Info,
} as const;

const ACCENTS = {
  success: "border-success/40 text-success",
  error: "border-destructive/40 text-destructive",
  info: "border-accent/40 text-accent",
} as const;

function ToastCard({ toast }: { toast: Toast }) {
  const dismiss = useToastStore((state) => state.dismiss);
  const Icon = ICONS[toast.variant];

  useEffect(() => {
    const timer = window.setTimeout(
      () => dismiss(toast.id),
      toast.duration ?? 4500,
    );
    return () => window.clearTimeout(timer);
  }, [toast.id, toast.duration, dismiss]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 24, scale: 0.96 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 24, scale: 0.96 }}
      transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
      role="status"
      className={cn(
        "pointer-events-auto flex w-[340px] max-w-[calc(100vw-2rem)] items-start gap-3 rounded-md border bg-bg-elevated/95 px-4 py-3 shadow-[0_8px_24px_rgba(0,0,0,0.45)] backdrop-blur-xl",
        ACCENTS[toast.variant],
      )}
    >
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-text-primary">{toast.title}</p>
        {toast.message ? (
          <p className="mt-0.5 break-words text-xs leading-relaxed text-text-secondary">
            {toast.message}
          </p>
        ) : null}
      </div>
      <button
        type="button"
        onClick={() => dismiss(toast.id)}
        className="focus-ring -mr-1 rounded p-0.5 text-text-muted transition-colors hover:text-text-primary"
        aria-label="Dismiss notification"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </motion.div>
  );
}

export function ToastViewport() {
  const toasts = useToastStore((state) => state.toasts);

  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-[80] flex flex-col gap-2"
      aria-live="polite"
      aria-atomic="false"
    >
      <AnimatePresence initial={false}>
        {toasts.map((t) => (
          <ToastCard key={t.id} toast={t} />
        ))}
      </AnimatePresence>
    </div>
  );
}
