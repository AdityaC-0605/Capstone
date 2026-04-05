"use client";

export function TerminalPanel({
  label,
  text,
}: {
  label: string;
  text: string;
}) {
  return (
    <div className="rounded-md border border-border bg-bg-elevated p-4 shadow-inner">
      <p className="text-[10px] font-bold uppercase tracking-wider text-success/80">
        {label}
      </p>
      <p className="mt-3 font-mono text-sm leading-relaxed text-success font-medium">
        {text}
        <span className="cursor-blink ml-0.5 inline-block">_</span>
      </p>
    </div>
  );
}
