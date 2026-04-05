"use client";

import { useState } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AnimatedNumber } from "@/components/animated-number";
import { FederatedNetwork } from "@/components/federated-network";
import { RangeField } from "@/components/range-field";
import { StateCard } from "@/components/state-card";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";

export default function FederatedPage() {
  const federatedState = usePulseStore((state) => state.federatedState);
  const setFederatedConfig = usePulseStore((state) => state.setFederatedConfig);
  const setFederatedHistory = usePulseStore((state) => state.setFederatedHistory);

  const [currentRound, setCurrentRound] = useState(1);
  const [phase, setPhase] = useState<"idle" | "upload" | "broadcast">("idle");
  const [activeNode, setActiveNode] = useState<string | null>(null);

  const runSimulation = async () => {
    setFederatedConfig({
      running: true,
      completed: false,
      bestValLoss: null,
      lossHistory: [],
    });
    setFederatedHistory([]);
    setPhase("idle");
    setActiveNode(null);

    const nextHistory = [];
    for (let round = 1; round <= federatedState.rounds; round += 1) {
      setCurrentRound(round);
      for (let client = 1; client <= federatedState.clients; client += 1) {
        setActiveNode(`C${client}`);
        await new Promise((resolve) => window.setTimeout(resolve, 170));
      }
      setActiveNode("SERVER");
      await new Promise((resolve) => window.setTimeout(resolve, 240));
      setPhase("upload");
      await new Promise((resolve) => window.setTimeout(resolve, 720));
      setPhase("broadcast");
      await new Promise((resolve) => window.setTimeout(resolve, 720));
      setPhase("idle");
      setActiveNode(null);
      const baseline = 0.82 - round * 0.11;
      const jitter = (Math.random() - 0.5) * 0.04;
      const loss = Math.max(0.12, Number((baseline + jitter).toFixed(3)));
      nextHistory.push({ round, loss });
      setFederatedHistory([...nextHistory]);
      await new Promise((resolve) => window.setTimeout(resolve, 180));
    }

    setFederatedConfig({
      running: false,
      completed: true,
      bestValLoss: Math.min(...nextHistory.map((item) => item.loss)),
    });
    setPhase("idle");
    setActiveNode(null);
  };

  return (
    <div className="page-frame space-y-6">
      {/* ─── Hero ─── */}
      <section className="animate-enter-1 rounded-lg border border-border bg-bg-surface p-6 shadow-sm md:p-8">
        <p className="section-kicker !text-warning">Federated Learning Simulation</p>
        <h1 className="mt-2 font-display text-3xl font-bold text-text-primary md:text-4xl">
          Distributed client coordination visualized as a living network.
        </h1>
        <p className="mt-4 text-sm leading-relaxed text-text-secondary font-medium">
          <span className="font-mono font-bold text-warning drop-shadow-[0_0_8px_rgba(245,158,11,0.5)]">{federatedState.clients}</span> clients ·{" "}
          <span className="font-mono font-bold text-warning drop-shadow-[0_0_8px_rgba(245,158,11,0.5)]">{federatedState.rounds}</span> aggregation rounds ·{" "}
          <span className="font-mono font-bold text-warning drop-shadow-[0_0_8px_rgba(245,158,11,0.5)]">{federatedState.localEpochs}</span> local epochs
        </p>
      </section>

      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        {/* ─── Config Panel ─── */}
        <aside className="animate-enter-2 rounded-lg border border-border bg-bg-surface p-6 shadow-sm self-start">
          <div className="space-y-4">
            <RangeField
              label="Number of Clients"
              min={2}
              max={10}
              value={federatedState.clients}
              valueLabel={`${federatedState.clients}`}
              onChange={(value) => setFederatedConfig({ clients: value })}
            />
            <RangeField
              label="Aggregation Rounds"
              min={1}
              max={10}
              value={federatedState.rounds}
              valueLabel={`${federatedState.rounds}`}
              onChange={(value) => setFederatedConfig({ rounds: value })}
            />
            <RangeField
              label="Local Epochs"
              min={1}
              max={5}
              value={federatedState.localEpochs}
              valueLabel={`${federatedState.localEpochs}`}
              onChange={(value) => setFederatedConfig({ localEpochs: value })}
            />

            <div className="pt-2">
              <button
                type="button"
                className="button-primary w-full"
                onClick={runSimulation}
                disabled={federatedState.running}
              >
                {federatedState.running ? (
                  <>
                    <span className="button-spinner mr-2" />
                    Running...
                  </>
                ) : (
                  "Run Simulation"
                )}
              </button>
            </div>
          </div>
        </aside>

        {/* ─── Results ─── */}
        <section className="animate-enter-3 space-y-6">
          <FederatedNetwork
            clients={federatedState.clients}
            running={federatedState.running}
            round={currentRound}
            phase={phase}
            activeNode={activeNode}
          />

          {/* Best Val Loss + Chart */}
          <div className="rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
            <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
              <div>
                <p className="section-kicker !text-warning">Best Val Loss</p>
                <AnimatedNumber
                  value={federatedState.bestValLoss || 0}
                  formatter={(value) =>
                    federatedState.bestValLoss === null ? "0.000" : value.toFixed(3)
                  }
                  className="mt-3 block font-mono text-4xl font-bold text-text-primary drop-shadow-[0_2px_12px_rgba(245,158,11,0.2)]"
                />
                {federatedState.completed ? (
                  <div className="relative mt-6 overflow-hidden rounded-md border border-success/30 bg-success/10 px-4 py-2.5 text-[10px] font-bold uppercase tracking-wider text-success">
                    Simulation Complete
                    {Array.from({ length: 8 }).map((_, index) => (
                      <span
                        key={index}
                        className="confetti-piece animate-confetti"
                        style={{
                          background:
                            index % 3 === 0
                              ? "rgb(var(--color-success))"
                              : index % 2 === 0
                                ? "rgb(var(--color-warning))"
                                : "rgb(var(--color-accent))",
                          ["--tx" as string]: `${(index - 4) * 16}px`,
                          ["--ty" as string]: `${(index % 2 === 0 ? -1 : 1) * 40}px`,
                          ["--rot" as string]: `${index * 24}deg`,
                          animationDelay: `${index * 0.03}s`,
                        }}
                      />
                    ))}
                  </div>
                ) : null}
              </div>
              <div className="h-[240px]">
                {federatedState.lossHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={federatedState.lossHistory}>
                      <defs>
                        <linearGradient id="lossFill" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="rgb(var(--color-warning))" stopOpacity={0.15} />
                          <stop offset="100%" stopColor="rgb(var(--color-warning))" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis
                        dataKey="round"
                        tick={{ fill: "#8c8a82", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fill: "#8c8a82", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "rgb(var(--color-bg-elevated))",
                          border: "1px solid rgba(245,158,11,0.3)",
                          borderRadius: "8px",
                          color: "rgb(var(--color-text-primary))",
                          fontSize: "12px",
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="loss"
                        stroke="rgb(var(--color-warning))"
                        fill="url(#lossFill)"
                        strokeWidth={2.5}
                        style={{ filter: "drop-shadow(0 0 8px rgba(245,158,11,0.5))" }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <StateCard
                    title="Round History"
                    message="Start the simulation to watch validation loss update."
                  />
                )}
              </div>
            </div>
          </div>

          {/* Round History Table */}
          <div className="rounded-lg border border-border bg-bg-surface p-6 shadow-sm">
            <p className="section-kicker !text-warning">Round History</p>
            {federatedState.lossHistory.length > 0 ? (
              <div className="mt-5 overflow-hidden rounded-md border border-border">
                <table className="w-full border-collapse text-left text-sm">
                  <thead className="bg-bg-elevated text-[10px] font-bold uppercase tracking-wider text-text-muted">
                    <tr>
                      <th className="px-5 py-4">Round</th>
                      <th className="px-5 py-4">Avg Loss</th>
                      <th className="px-5 py-4">Δ from Previous</th>
                    </tr>
                  </thead>
                  <tbody>
                    {federatedState.lossHistory.map((item, index) => {
                      const previous = federatedState.lossHistory[index - 1]?.loss;
                      const delta =
                        previous === undefined ? "—" : (item.loss - previous).toFixed(3);
                      const isNeg = previous !== undefined && item.loss < previous;
                      return (
                        <tr
                          key={item.round}
                          className={cn(
                            "fade-in-row border-t border-border bg-bg-surface hover:bg-bg-elevated/40 transition-colors",
                          )}
                          style={{ animationDelay: `${index * 0.08}s` }}
                        >
                          <td className="px-5 py-4 font-mono font-medium text-text-primary">{item.round}</td>
                          <td className="px-5 py-4 font-mono font-bold text-text-primary">{item.loss.toFixed(3)}</td>
                          <td className={cn(
                            "px-5 py-4 font-mono font-bold",
                            isNeg ? "text-success drop-shadow-[0_0_8px_rgba(34,197,94,0.4)]" : "text-text-secondary w-[80px]",
                          )}>
                            {delta}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <StateCard
                title="No Simulation Yet"
                message="Configure client count, rounds, and epochs, then run the simulation."
              />
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
