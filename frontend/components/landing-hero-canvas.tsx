"use client";

import { useEffect, useRef } from "react";

/**
 * A faint, slow-drifting node network behind the hero — a nod to the
 * federated-learning topology, rendered as quiet paper texture (low alpha,
 * evergreen). Respects prefers-reduced-motion by drawing a single static frame.
 */
export function LandingHeroCanvas({ className }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let raf = 0;
    let w = 0;
    let h = 0;
    const nodes: Array<{ x: number; y: number; vx: number; vy: number }> = [];
    const COUNT = 42;
    const LINK = 150;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas.width = Math.max(1, w * dpr);
      canvas.height = Math.max(1, h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const init = () => {
      nodes.length = 0;
      for (let i = 0; i < COUNT; i += 1) {
        nodes.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.16,
          vy: (Math.random() - 0.5) * 0.16,
        });
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const node of nodes) {
        if (!reduce) {
          node.x += node.vx;
          node.y += node.vy;
          if (node.x < 0 || node.x > w) node.vx *= -1;
          if (node.y < 0 || node.y > h) node.vy *= -1;
        }
      }
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const a = nodes[i];
          const b = nodes[j];
          const dist = Math.hypot(a.x - b.x, a.y - b.y);
          if (dist < LINK) {
            const opacity = (1 - dist / LINK) * 0.1;
            ctx.strokeStyle = `rgba(28,70,52,${opacity})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      for (const node of nodes) {
        ctx.fillStyle = "rgba(28,70,52,0.3)";
        ctx.beginPath();
        ctx.arc(node.x, node.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!reduce) raf = requestAnimationFrame(draw);
    };

    resize();
    init();
    draw();

    const onResize = () => {
      resize();
      init();
      if (reduce) draw();
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}
