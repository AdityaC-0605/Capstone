"use client";

interface SparklineProps {
  values: number[];
  color: string;
}

export function Sparkline({ values, color }: SparklineProps) {
  if (values.length < 2) return null;

  const max = Math.max(...values, 0.0001);
  const height = 32;
  const width = 100;
  const step = width / (values.length - 1);

  const points = values.map((value, index) => ({
    x: index * step,
    y: height - (value / max) * height * 0.85,
  }));

  const pathD = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="mt-3 h-8 w-full">
      <path
        d={pathD}
        className="sparkline-path"
        stroke={color}
        strokeWidth="1.75"
      />
    </svg>
  );
}
