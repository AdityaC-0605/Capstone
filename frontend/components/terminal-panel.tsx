"use client";

/**
 * Analyst narrative — set as an editorial note (serif, ruled left margin)
 * rather than a terminal readout. The serif voice signals human judgment.
 */
export function TerminalPanel({
  label,
  text,
}: {
  label: string;
  text: string;
}) {
  return (
    <figure className="border-l-2 border-l-accent bg-bg-elevated/60 py-3 pl-4 pr-3">
      <figcaption className="section-kicker">{label}</figcaption>
      <p className="mt-2 font-display text-[15px] leading-relaxed text-text-primary">
        {text}
      </p>
    </figure>
  );
}
