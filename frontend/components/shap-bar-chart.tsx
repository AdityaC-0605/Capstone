"use client";

import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatFeatureLabel } from "@/lib/format";

/**
 * Horizontal SHAP attribution chart. Positive contributions (push risk up) are
 * rendered in the destructive accent; negative contributions (pull risk down)
 * in the success accent — mirroring the rest of the design system's semantics.
 */
export function ShapBarChart({
  importance,
  limit = 8,
}: {
  importance: Record<string, number>;
  limit?: number;
}) {
  const data = Object.entries(importance)
    .map(([feature, value]) => ({
      feature: formatFeatureLabel(feature),
      value: Number(value),
      abs: Math.abs(Number(value)),
    }))
    .filter((entry) => Number.isFinite(entry.value))
    .sort((a, b) => b.abs - a.abs)
    .slice(0, limit);

  if (data.length === 0) return null;

  const height = Math.max(140, data.length * 34);

  return (
    <div className="rounded-md border border-border bg-bg-elevated p-4">
      <div className="flex items-center justify-between">
        <p className="section-kicker">Feature Attribution</p>
        <div className="flex items-center gap-3 text-[9px] font-bold uppercase tracking-wider">
          <span className="flex items-center gap-1.5 text-destructive">
            <span className="h-2 w-2 rounded-sm bg-destructive" /> Raises risk
          </span>
          <span className="flex items-center gap-1.5 text-success">
            <span className="h-2 w-2 rounded-sm bg-success" /> Lowers risk
          </span>
        </div>
      </div>
      <div className="mt-4" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 0, right: 12, bottom: 0, left: 8 }}
          >
            <XAxis
              type="number"
              tick={{ fill: "rgb(var(--color-text-muted))", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={118}
              tick={{ fill: "rgb(var(--color-text-secondary))", fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <ReferenceLine x={0} stroke="rgb(var(--color-border-strong))" />
            <Tooltip
              cursor={{ fill: "rgb(var(--color-bg-surface))", opacity: 0.5 }}
              contentStyle={{
                background: "rgb(var(--color-bg-elevated))",
                border: "1px solid rgb(var(--color-border-strong))",
                borderRadius: "8px",
                color: "rgb(var(--color-text-primary))",
                fontSize: "12px",
              }}
              formatter={(value: number) => [Number(value).toFixed(4), "SHAP"]}
            />
            <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
              {data.map((entry) => (
                <Cell
                  key={entry.feature}
                  fill={
                    entry.value >= 0
                      ? "rgb(var(--color-destructive))"
                      : "rgb(var(--color-success))"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
