"use client";

interface FederatedNetworkProps {
  clients: number;
  running: boolean;
  round: number;
  phase: "idle" | "upload" | "broadcast";
  activeNode: string | null;
}

/**
 * Federated topology — a quiet ink diagram on paper. Clients arc above a
 * central aggregator; dots travel the edges during upload/broadcast phases.
 */
export function FederatedNetwork({
  clients,
  running,
  round,
  phase,
  activeNode,
}: FederatedNetworkProps) {
  const serverCx = 250;
  const serverCy = 130;
  const viewWidth = 500;
  const viewHeight = 220;
  const nodeRadius = 165;
  const nodeCount = Math.min(clients, 10);

  const ink = "rgb(var(--color-text-primary))";
  const accent = "rgb(var(--color-accent))";
  const rule = "rgb(var(--color-border-strong))";
  const muted = "rgb(var(--color-text-muted))";

  const clientNodes = Array.from({ length: nodeCount }, (_, i) => {
    const angle = Math.PI + (Math.PI / (nodeCount + 1)) * (i + 1);
    return {
      id: `C${i + 1}`,
      cx: serverCx + Math.cos(angle) * nodeRadius,
      cy: serverCy + Math.sin(angle) * nodeRadius * 0.62,
    };
  });

  return (
    <div className="leaf p-5">
      <div className="mb-2 flex items-center justify-between">
        <span className="section-kicker">Network Topology</span>
        {running ? (
          <span className="rounded-[3px] border border-accent/40 bg-accent/8 px-2 py-0.5 font-mono text-[11px] text-accent">
            Round {round}
          </span>
        ) : (
          <span className="font-mono text-[11px] text-text-muted">
            {nodeCount} clients
          </span>
        )}
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
            stroke={rule}
            strokeOpacity={0.5}
            strokeWidth="1"
            strokeDasharray={phase === "idle" ? "none" : "4 5"}
          />
        ))}

        {/* Travel dots during a phase */}
        {phase !== "idle" &&
          clientNodes.map((node) => {
            const isUpload = phase === "upload";
            return (
              <circle
                key={`dot-${node.id}`}
                r="3"
                fill={isUpload ? accent : "rgb(var(--color-warning))"}
              >
                <animateMotion
                  dur="1.1s"
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

        {/* Aggregator */}
        <circle
          cx={serverCx}
          cy={serverCy}
          r="24"
          fill={activeNode === "SERVER" ? accent : "rgb(var(--color-bg-elevated))"}
          stroke={accent}
          strokeWidth="1.5"
        />
        <text
          x={serverCx}
          y={serverCy + 1}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={activeNode === "SERVER" ? "rgb(var(--color-bg-surface))" : ink}
          fontSize="9"
          fontFamily="ui-monospace, monospace"
          letterSpacing="0.5"
        >
          AGG
        </text>

        {/* Clients */}
        {clientNodes.map((node) => {
          const active = activeNode === node.id;
          return (
            <g key={node.id}>
              <circle
                cx={node.cx}
                cy={node.cy}
                r="15"
                fill={active ? "rgb(var(--color-accent) / 0.12)" : "rgb(var(--color-bg-elevated))"}
                stroke={active ? accent : rule}
                strokeWidth={active ? 1.5 : 1}
              />
              <text
                x={node.cx}
                y={node.cy + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={active ? accent : muted}
                fontSize="8"
                fontFamily="ui-monospace, monospace"
              >
                {node.id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
