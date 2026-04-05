"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard,
  FlaskConical,
  Leaf,
  Network,
  Settings,
  ChevronLeft,
  Menu,
  X,
  Activity,
} from "lucide-react";

import { probeBackends } from "@/lib/api";
import { aggregateStatusColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";
import { SettingsPanel } from "@/components/settings-panel";

const navLinks = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/studio", label: "Studio", icon: FlaskConical },
  { href: "/sustainability", label: "Sustainability", icon: Leaf },
  { href: "/federated", label: "Federated", icon: Network },
];

export function SiteShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const backendStatus = usePulseStore((state) => state.backendStatus);
  const setBackendStatus = usePulseStore((state) => state.setBackendStatus);
  const openSettings = usePulseStore((state) => state.openSettings);
  const closeSettings = usePulseStore((state) => state.closeSettings);
  const setMobileNavOpen = usePulseStore((state) => state.setMobileNavOpen);
  const ui = usePulseStore((state) => state.ui);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const status = await probeBackends(backendConfig);
        if (active) setBackendStatus(status);
      } catch (error) {
        if (!active) return;
        setBackendStatus({
          main: {
            state: "offline",
            label: "Main API",
            detail:
              error instanceof Error
                ? error.message
                : "Failed to fetch. Is backend running?",
          },
          inference: {
            state: "offline",
            label: "Inference Engine",
            detail: "Failed to fetch. Is backend running?",
          },
          features: [],
          overall: {
            state: "offline",
            label: "Offline",
            detail: "Failed to fetch. Is backend running?",
          },
          lastChecked: new Date().toISOString(),
        });
      }
    };

    void refresh();
    const interval = window.setInterval(() => void refresh(), 20000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [backendConfig, setBackendStatus]);

  return (
    <div className="relative flex min-h-screen">
      {/* ─── Desktop Sidebar ─── */}
      <aside className="sidebar fixed inset-y-0 left-0 z-50 hidden flex-col lg:flex">
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center px-4">
            <Link href="/" className="focus-ring flex items-center gap-2.5 rounded-md px-2 py-1.5 transition-all hover:bg-bg-elevated/40">
              <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent/10 border border-accent/20">
                <Activity className="h-4 w-4 text-accent" />
              </div>
              <span className="font-display text-base font-semibold text-text-primary tracking-tight">
                Pulse<span className="text-accent">Ledger</span>
              </span>
            </Link>
          </div>

          {/* Nav Links */}
          <nav className="mt-6 flex-1 space-y-1 px-3">
            <p className="mb-4 px-3 text-[10px] font-semibold uppercase tracking-widest text-text-muted">
              Navigation
            </p>
            {navLinks.map((link) => {
              const isActive =
                link.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(link.href);

              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md font-medium text-sm transition-all duration-120",
                    isActive
                      ? "bg-accent/10 text-text-primary shadow-[inset_2px_0_0_rgb(var(--color-accent))]"
                      : "text-text-muted hover:text-text-primary hover:bg-bg-elevated/50"
                  )}
                >
                  <link.icon className="h-[18px] w-[18px]" />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Bottom Section */}
          <div className="space-y-2 border-t border-border/50 p-3">
            {/* Backend Status */}
            <div className="flex items-center gap-3 rounded-md border border-border bg-bg-surface px-3 py-2.5">
              <span
                className={cn(
                  "status-dot",
                  aggregateStatusColor(backendStatus.overall.state),
                )}
              />
              <div className="min-w-0 flex-1">
                <p className="truncate text-[10px] font-semibold uppercase tracking-widest text-text-muted">
                  Backend
                </p>
                <p className="truncate text-xs font-mono text-text-primary">
                  {backendStatus.overall.label}
                </p>
              </div>
            </div>

            {/* Settings Button */}
            <button
              type="button"
              onClick={openSettings}
              className="flex w-full items-center gap-3 px-3 py-2 rounded-md font-medium text-sm text-text-muted transition-all duration-120 hover:text-text-primary hover:bg-bg-elevated/50"
            >
              <Settings className="h-[18px] w-[18px]" />
              <span>Settings</span>
            </button>
          </div>
        </div>
      </aside>

      {/* ─── Mobile Header ─── */}
      <header className="fixed inset-x-0 top-0 z-50 border-b border-border bg-bg-primary/80 backdrop-blur-xl lg:hidden">
        <div className="flex h-14 items-center justify-between px-4">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent/10 border border-accent/20">
              <Activity className="h-3.5 w-3.5 text-accent" />
            </div>
            <span className="font-display text-sm font-semibold text-text-primary tracking-tight">
              Pulse<span className="text-accent">Ledger</span>
            </span>
          </Link>

          <div className="flex items-center gap-2">
            <span
              className={cn(
                "status-dot",
                aggregateStatusColor(backendStatus.overall.state),
              )}
            />
            <button
              type="button"
              className="focus-ring flex h-9 w-9 items-center justify-center rounded-md border border-border bg-bg-elevated"
              onClick={() => setMobileNavOpen(!ui.mobileNavOpen)}
            >
              <span className="sr-only">Toggle navigation</span>
              {ui.mobileNavOpen ? (
                <X className="h-4 w-4 text-text-primary" />
              ) : (
                <Menu className="h-4 w-4 text-text-primary" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Nav Dropdown */}
        <AnimatePresence>
          {ui.mobileNavOpen ? (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden border-t border-border bg-bg-surface/95 backdrop-blur-xl"
            >
              <nav className="space-y-1 p-3">
                {navLinks.map((link) => {
                  const isActive =
                    link.href === "/"
                      ? pathname === "/"
                      : pathname.startsWith(link.href);

                  return (
                    <Link
                      key={link.href}
                      href={link.href}
                      onClick={() => setMobileNavOpen(false)}
                      className={cn(
                        "flex items-center gap-3 rounded-md px-4 py-3 text-sm font-medium transition",
                        isActive
                          ? "bg-accent/10 text-accent"
                          : "text-text-muted hover:text-text-primary hover:bg-bg-elevated",
                      )}
                    >
                      <link.icon className="h-4 w-4" />
                      {link.label}
                    </Link>
                  );
                })}
                <button
                  type="button"
                  onClick={() => {
                    setMobileNavOpen(false);
                    openSettings();
                  }}
                  className="flex w-full items-center gap-3 rounded-md px-4 py-3 text-left text-sm font-medium text-text-muted hover:text-text-primary hover:bg-bg-elevated"
                >
                  <Settings className="h-4 w-4" />
                  Settings
                </button>
              </nav>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </header>

      {/* ─── Main Content ─── */}
      <div className="flex min-h-screen flex-1 flex-col lg:pl-[var(--sidebar-w)]">
        <main className="flex-1 pt-[calc(56px+1.5rem)] lg:pt-8 pb-8">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={pathname}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>

        <footer className="page-frame pb-6 pt-2">
          <div className="mb-4 h-px bg-border w-full" />
          <div className="flex flex-col gap-1.5 text-xs text-text-muted md:flex-row md:items-center md:justify-between">
            <p className="font-sans">PulseLedger — explainable credit risk, federated learning, and carbon-aware experimentation.</p>
            <p className="font-mono text-[10px] tracking-wider opacity-60">PRODUCTION-GRADE · SHAP · FL · LIVE</p>
          </div>
        </footer>
      </div>

      {/* ─── Settings Overlay ─── */}
      <AnimatePresence>
        {ui.settingsOpen ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] bg-black/60 backdrop-blur-sm"
            onClick={closeSettings}
          >
            <div onClick={(event) => event.stopPropagation()} className="modal-enter h-full">
              <SettingsPanel />
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
