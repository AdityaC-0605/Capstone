"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft } from "lucide-react";

import { loginUser, registerUser } from "@/lib/api";
import { usePulseStore } from "@/store/use-pulse-store";
import { useAuthStore } from "@/store/use-auth-store";
import { toast } from "@/store/use-toast-store";

const DEMO = { email: "demo@pulseledger.app", password: "demo12345" };

function Wordmark() {
  return (
    <span className="flex items-center justify-center gap-2.5">
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

export default function LoginPage() {
  const router = useRouter();
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const setBackendConfig = usePulseStore((state) => state.setBackendConfig);
  const setAuth = useAuthStore((state) => state.setAuth);

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const authenticate = async (creds: { email: string; password: string }) => {
    setLoading(true);
    setError("");
    try {
      const result =
        mode === "register"
          ? await registerUser(backendConfig, {
              ...creds,
              full_name: fullName,
            })
          : await loginUser(backendConfig, creds);
      setAuth(result.access_token, result.user);
      // The session token doubles as the bearer for scoring requests.
      setBackendConfig({ ...backendConfig, apiKey: result.access_token });
      toast.success(
        mode === "register" ? "Account created" : "Welcome back",
        result.user.email,
      );
      router.push("/dashboard");
    } catch (caught) {
      const message =
        caught instanceof Error ? caught.message : "Authentication failed.";
      setError(message);
      toast.error("Authentication failed", message);
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <Link
        href="/"
        className="focus-ring mb-8 inline-flex items-center gap-1.5 rounded-[3px] text-sm text-text-muted transition-colors hover:text-text-primary"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to site
      </Link>

      <div className="w-full max-w-sm">
        <div className="mb-8 flex justify-center">
          <Wordmark />
        </div>

        <div className="leaf p-7">
          <div className="mb-6 flex gap-1 rounded-[3px] border border-border bg-bg-elevated p-1">
            {(["login", "register"] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => {
                  setMode(value);
                  setError("");
                }}
                className={
                  "focus-ring flex-1 rounded-[2px] px-3 py-1.5 text-sm font-medium transition-colors " +
                  (mode === value
                    ? "bg-bg-surface text-text-primary shadow-sm"
                    : "text-text-muted hover:text-text-primary")
                }
              >
                {value === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>

          <form
            className="space-y-4"
            onSubmit={(event) => {
              event.preventDefault();
              authenticate({ email, password });
            }}
          >
            {mode === "register" ? (
              <label className="block space-y-1.5">
                <span className="text-sm text-text-secondary">Full name</span>
                <input
                  className="input-shell"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  placeholder="Jordan Analyst"
                  autoComplete="name"
                />
              </label>
            ) : null}
            <label className="block space-y-1.5">
              <span className="text-sm text-text-secondary">Email</span>
              <input
                className="input-shell"
                type="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@company.com"
                autoComplete="email"
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm text-text-secondary">Password</span>
              <input
                className="input-shell"
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="At least 8 characters"
                autoComplete={
                  mode === "register" ? "new-password" : "current-password"
                }
              />
            </label>

            {error ? (
              <p className="rounded-[3px] border border-destructive/40 bg-destructive/8 px-3 py-2 text-xs leading-relaxed text-destructive">
                {error}
              </p>
            ) : null}

            <button
              type="submit"
              className="button-primary w-full"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="button-spinner" />
                  Working…
                </>
              ) : mode === "register" ? (
                "Create account"
              ) : (
                "Sign in"
              )}
            </button>
          </form>

          <div className="mt-4 border-t border-border pt-4">
            <button
              type="button"
              className="button-ghost w-full"
              disabled={loading}
              onClick={() => {
                setMode("login");
                setEmail(DEMO.email);
                setPassword(DEMO.password);
                authenticate(DEMO);
              }}
            >
              Use demo account
            </button>
            <p className="mt-2 text-center text-xs text-text-muted">
              Explore the full workspace — no signup required.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
