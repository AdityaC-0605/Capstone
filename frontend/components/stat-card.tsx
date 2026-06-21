import { cn } from "@/lib/utils";

export function StatCard({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div className="leaf p-5">
      <p className="section-kicker">{label}</p>
      <p
        className={cn(
          "mt-2.5 font-display text-3xl font-medium tabular",
          accent ? "text-accent" : "text-text-primary",
        )}
      >
        {value}
      </p>
      {hint ? <p className="mt-1 text-xs text-text-muted">{hint}</p> : null}
    </div>
  );
}
