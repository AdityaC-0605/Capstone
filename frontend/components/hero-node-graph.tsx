"use client";

/**
 * Animated mesh gradient background for the hero section.
 * Replaces the previous Three.js canvas for a lighter, CSS-only solution.
 */
export function HeroNodeGraph() {
  return (
    <div className="hero-mesh" aria-hidden="true">
      <div className="hero-mesh-center" />
    </div>
  );
}
