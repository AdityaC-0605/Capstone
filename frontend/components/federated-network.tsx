"use client";

import { cn } from "@/lib/utils";

interface FederatedNetworkProps {
  clients: number;
  running: boolean;
  round: number;
  phase: "idle" | "upload" | "broadcast";
  activeNode: string | null;
}

export function FederatedNetwork({
  clients,
  running,
  round,
  phase,
  activeNode,
}: FederatedNetworkProps) {
  const serverCx = 250;
  const serverCy = 120;
  const viewWidth = 500;
  const viewHeight = 240;
  const nodeRadius = 160;
  const nodeCount = Math.min(clients, 10);

  const clientNodes = Array.from({ length: nodeCount }, (_, i) => {
    const angle = Math.PI + (Math.PI / (nodeCount + 1)) * (i + 1);
    return {
      id: `C${i + 1}`,
      cx: serverCx + Math.cos(angle) * nodeRadius,
      cy: serverCy + Math.sin(angle) * nodeRadius * 0.7,
    };
  });

  return (
    <div className="glass-panel-static rounded-2xl p-5">
      <div className="mb-3 flex items-center justify-between">
        <p className="section-kicker !text-gold">Network Topology</p>
        {running ? (
          <span className="rounded-lg border border-gold/20 bg-gold/6 px-2.5 py-1 font-mono text-[10px] font-semibold text-gold">
            Round {round}
          </span>
        ) : null}
      </div>
      <svg viewBox={`0 0 ${viewWidth} ${viewHeight}`} className="w-full">
        {/* Edges */}
        {clientNodes.map((node) => (
          <line
            key={`edge-${node.id}`}
            x1={serverCx}
            y1={serverCy}
            x2={node.cx}
            y2={node.cy}
            stroke="rgba(245,158,11,0.1)"
            strokeWidth="1.5"
            strokeDasharray={phase === "idle" ? "none" : "6 6"}
            className={phase !== "idle" ? "animate-dash-flow" : ""}
          />
        ))}

        {/* Travel dots during phase */}
        {phase !== "idle" &&
          clientNodes.map((node) => {
            const isUpload = phase === "upload";
            return (
              <circle
                key={`dot-${node.id}`}
                r="3"
                fill="#F59E0B"
                className="travel-dot"
              >
                <animateMotion
                  dur="1.2s"
                  repeatCount="indefinite"
                  path={
                    isUpload
                      ? `M${node.cx},${node.cy} L${serverCx},${serverCy}`
                      : `M${serverCx},${serverCy} L${node.cx},${node.cy}`
                  }
                />
              </circle>
            );
          })}

        {/* Server Node */}
        <circle
          cx={serverCx}
          cy={serverCy}
          r="22"
          fill="rgba(245,158,11,0.06)"
          stroke={activeNode === "SERVER" ? "#F59E0B" : "rgba(245,158,11,0.2)"}
          strokeWidth="2"
          className={activeNode === "SERVER" ? "node-pulse" : ""}
        />
        <text
          x={serverCx}
          y={serverCy + 1}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgb(var(--copy))"
          fontSize="9"
          fontWeight="600"
          fontFamily="var(--font-sans)"
        >
          SERVER
        </text>

        {/* Client Nodes */}
        {clientNodes.map((node) => (
          <g key={node.id}>
            <circle
              cx={node.cx}
              cy={node.cy}
              r="16"
              fill="rgba(45,212,191,0.05)"
              stroke={
                activeNode === node.id
                  ? "#2DD4BF"
                  : "rgba(45,212,191,0.18)"
              }
              strokeWidth="1.5"
              className={activeNode === node.id ? "node-pulse" : ""}
            />
            <text
              x={node.cx}
              y={node.cy + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="rgb(var(--muted))"
              fontSize="8"
              fontWeight="600"
              fontFamily="var(--font-mono)"
            >
              {node.id}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
