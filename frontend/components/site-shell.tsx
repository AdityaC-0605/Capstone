"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ClipboardList,
  LayoutDashboard,
  Leaf,
  LogOut,
  Menu,
  Network,
  Plus,
  Scale,
  Settings,
  X,
} from "lucide-react";

import { probeBackends } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/store/use-auth-store";
import { usePulseStore } from "@/store/use-pulse-store";
import type { AuthUser } from "@/lib/types";

const navGroups = [
  {
    label: "Workspace",
    items: [
      { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
      { href: "/assessments", label: "Assessments", icon: ClipboardList },
    ],
  },
  {
    label: "Model Operations",
    items: [
      { href: "/federated", label: "Federated", icon: Network },
      { href: "/fairness", label: "Fairness", icon: Scale },
      { href: "/sustainability", label: "Sustainability", icon: Leaf },
    ],
  },
];

const statusDot: Record<string, string> = {
  healthy: "status-online",
  partial: "status-partial",
  offline: "status-offline",
  checking: "status-checking",
};

const titleForPath = (path: string): string => {
  if (path === "/dashboard") return "Dashboard";
  if (path === "/assessments") return "Assessments";
  if (path === "/assessments/new") return "New Assessment";
  if (path === "/assessments/batch") return "Bulk Scoring";
  if (path.startsWith("/assessments/")) return "Assessment";
  if (path.startsWith("/federated")) return "Federated Learning";
  if (path.startsWith("/fairness")) return "Fairness Audit";
  if (path.startsWith("/sustainability")) return "Sustainability";
  if (path.startsWith("/settings")) return "Settings";
  return "PulseLedger";
};

function Wordmark() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="relative flex h-7 w-7 items-center justify-center rounded-[3px] bg-accent">
        <span className="absolute left-1.5 right-1.5 top-[8px] h-px bg-bg-surface/70" />
        <span className="absolute left-1.5 right-1.5 top-[13px] h-px bg-bg-surface/70" />
        <span className="absolute left-1.5 right-1.5 top-[18px] h-px bg-bg-surface/70" />
      </span>
      <span className="font-display text-lg font-semibold tracking-tight text-text-primary">
        PulseLedger
      </span>
    </span>
  );
}

function SidebarContent({
  pathname,
  status,
  user,
  onLogout,
  onNavigate,
}: {
  pathname: string;
  status: string;
  user: AuthUser | null;
  onLogout: () => void;
  onNavigate?: () => void;
}) {
  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-[60px] items-center px-5">
        <Link href="/dashboard" className="focus-ring rounded-[3px]" onClick={onNavigate}>
          <Wordmark />
        </Link>
      </div>

      <div className="px-3 pb-2">
        <Link
          href="/assessments/new"
          onClick={onNavigate}
          className="button-primary w-full"
        >
          <Plus className="h-4 w-4" />
          New assessment
        </Link>
      </div>

      <nav className="mt-3 flex-1 space-y-6 px-3">
        {navGroups.map((group) => (
          <div key={group.label}>
            <p className="px-2.5 pb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-text-muted">
              {group.label}
            </p>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const active = isActive(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={onNavigate}
                    className={cn(
                      "focus-ring flex items-center gap-3 rounded-[3px] px-2.5 py-2 text-sm transition-colors",
                      active
                        ? "bg-accent/8 font-medium text-accent shadow-[inset_2px_0_0_rgb(var(--color-accent))]"
                        : "text-text-secondary hover:bg-bg-elevated hover:text-text-primary",
                    )}
                  >
                    <item.icon className="h-[18px] w-[18px]" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="space-y-1 border-t border-border p-3">
        <div className="flex items-center gap-2.5 rounded-[3px] px-2.5 py-2">
          <span className={cn("status-dot", statusDot[status])} />
          <div className="min-w-0">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted">
              Backend
            </p>
            <p className="truncate text-xs text-text-primary">
              {status === "healthy"
                ? "All systems live"
                : status === "partial"
                  ? "Partial connectivity"
                  : status === "checking"
                    ? "Probing services"
                    : "Offline"}
            </p>
          </div>
        </div>
        <Link
          href="/settings"
          onClick={onNavigate}
          className={cn(
            "focus-ring flex items-center gap-3 rounded-[3px] px-2.5 py-2 text-sm transition-colors",
            pathname.startsWith("/settings")
              ? "bg-accent/8 font-medium text-accent"
              : "text-text-secondary hover:bg-bg-elevated hover:text-text-primary",
          )}
        >
          <Settings className="h-[18px] w-[18px]" />
          Settings
        </Link>

        {user ? (
          <div className="mt-1 flex items-center gap-2.5 rounded-[3px] border border-border bg-bg-elevated/60 px-2.5 py-2">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent font-mono text-xs font-medium uppercase text-bg-surface">
              {(user.full_name || user.email).slice(0, 1)}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-medium text-text-primary">
                {user.full_name || user.email}
              </p>
              <p className="truncate text-[11px] text-text-muted">
                {user.email}
              </p>
            </div>
            <button
              type="button"
              onClick={onLogout}
              className="focus-ring rounded-[3px] p-1 text-text-muted transition-colors hover:text-text-primary"
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function SiteShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const backendStatus = usePulseStore((state) => state.backendStatus);
  const setBackendStatus = usePulseStore((state) => state.setBackendStatus);
  const setBackendConfig = usePulseStore((state) => state.setBackendConfig);
  const token = useAuthStore((state) => state.token);
  const user = useAuthStore((state) => state.user);
  const clearAuth = useAuthStore((state) => state.clearAuth);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  // Gate the app behind login (the landing + /login live outside this shell).
  useEffect(() => {
    if (mounted && !token) router.replace("/login");
  }, [mounted, token, router]);

  const logout = () => {
    clearAuth();
    setBackendConfig({ ...backendConfig, apiKey: "" });
    router.replace("/login");
  };

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const status = await probeBackends(backendConfig);
        if (active) setBackendStatus(status);
      } catch (error) {
        if (!active) return;
        const detail =
          error instanceof Error ? error.message : "Failed to fetch.";
        setBackendStatus({
          main: { state: "offline", label: "Main API", detail },
          inference: {
            state: "offline",
            label: "Inference Engine",
            detail: "Failed to fetch.",
          },
          features: [],
          overall: { state: "offline", label: "Offline", detail },
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

  const overall = backendStatus.overall.state;
  const title = titleForPath(pathname);

  // Avoid an SSR/CSR flash and redirect unauthenticated users to /login.
  if (!mounted || !token) {
    return <div className="min-h-screen bg-bg-primary" />;
  }

  return (
    <div className="min-h-screen">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[248px] border-r border-border bg-bg-surface lg:block">
        <SidebarContent
          pathname={pathname}
          status={overall}
          user={user}
          onLogout={logout}
        />
      </aside>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-text-primary/30 backdrop-blur-sm lg:hidden"
            onClick={() => setMobileOpen(false)}
          >
            <motion.div
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              className="h-full w-[248px] border-r border-border bg-bg-surface"
              onClick={(event) => event.stopPropagation()}
            >
              <SidebarContent
                pathname={pathname}
                status={overall}
                user={user}
                onLogout={logout}
                onNavigate={() => setMobileOpen(false)}
              />
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      {/* Main column */}
      <div className="flex min-h-screen flex-col lg:pl-[248px]">
        <header className="sticky top-0 z-30 flex h-[60px] items-center justify-between border-b border-border bg-bg-primary/85 px-4 backdrop-blur-md md:px-8">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setMobileOpen(true)}
              className="focus-ring flex h-9 w-9 items-center justify-center rounded-[3px] border border-border lg:hidden"
              aria-label="Open navigation"
            >
              <Menu className="h-4 w-4" />
            </button>
            <h1 className="font-display text-lg font-medium text-text-primary">
              {title}
            </h1>
          </div>
          <Link
            href="/settings"
            className="focus-ring flex items-center gap-2 rounded-[3px] border border-border px-2.5 py-1.5 text-xs text-text-secondary transition-colors hover:border-border-strong hover:text-text-primary"
          >
            <span className={cn("status-dot", statusDot[overall])} />
            <span className="font-mono uppercase tracking-wider">
              {overall === "healthy"
                ? "Live"
                : overall === "partial"
                  ? "Partial"
                  : overall === "checking"
                    ? "Probing"
                    : "Offline"}
            </span>
          </Link>
        </header>

        <main className="flex-1">
          <div className="mx-auto w-full max-w-[1200px] px-4 py-7 md:px-8 md:py-10">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
