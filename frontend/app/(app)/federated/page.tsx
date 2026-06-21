"use client";

import { useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AnimatedNumber } from "@/components/animated-number";
import { FederatedNetwork } from "@/components/federated-network";
import { RangeField } from "@/components/range-field";
import { StateCard } from "@/components/state-card";
import { runFederated } from "@/lib/api";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";
import { toast } from "@/store/use-toast-store";

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

export default function FederatedPage() {
  const federatedState = usePulseStore((state) => state.federatedState);
  const backendConfig = usePulseStore((state) => state.backendConfig);
  const setFederatedConfig = usePulseStore((state) => state.setFederatedConfig);
  const setFederatedHistory = usePulseStore((state) => state.setFederatedHistory);

  const [currentRound, setCurrentRound] = useState(1);
  const [phase, setPhase] = useState<"idle" | "upload" | "broadcast">("idle");
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const animTimer = useRef<number | null>(null);

  const stopAmbientAnimation = () => {
    if (animTimer.current !== null) {
      window.clearInterval(animTimer.current);
      animTimer.current = null;
    }
    setPhase("idle");
    setActiveNode(null);
  };

  const startAmbientAnimation = (clientCount: number) => {
    let tick = 0;
    animTimer.current = window.setInterval(() => {
      tick += 1;
      const slot = tick % (clientCount + 2);
      if (slot < clientCount) {
        setActiveNode(`C${slot + 1}`);
        setPhase("idle");
      } else if (slot === clientCount) {
        setActiveNode("SERVER");
        setPhase("upload");
      } else {
        setActiveNode(null);
        setPhase("broadcast");
      }
    }, 220);
  };

  const runSimulation = async () => {
    setError(null);
    setFederatedConfig({
      running: true,
      completed: false,
      bestValLoss: null,
      bestRound: null,
      wallTimeSeconds: null,
      stoppedEarly: false,
      lossHistory: [],
    });
    setFederatedHistory([]);
    setCurrentRound(1);
    startAmbientAnimation(federatedState.clients);

    try {
      const result = await runFederated(backendConfig, {
        number_of_clients: federatedState.clients,
        aggregation_rounds: federatedState.rounds,
        local_epochs: federatedState.localEpochs,
      });

      stopAmbientAnimation();

      const history = result.round_metrics.map((metric, index) => ({
        round: index + 1,
        loss: Number(metric.average_val_loss.toFixed(4)),
        valAccuracy: Number((metric.average_val_accuracy ?? 0).toFixed(4)),
        clientLoss: Number((metric.average_client_loss ?? 0).toFixed(4)),
      }));

      for (let i = 0; i < history.length; i += 1) {
        setCurrentRound(i + 1);
        setFederatedHistory(history.slice(0, i + 1));
        await sleep(110);
      }

      setFederatedConfig({
        running: false,
        completed: true,
        source: "live",
        bestValLoss: result.best_val_loss,
        bestRound: result.best_round,
        wallTimeSeconds: result.wall_time_seconds,
        stoppedEarly: result.stopped_early,
      });

      toast.success(
        "Federated run complete",
        `Best validation loss ${result.best_val_loss.toFixed(4)} across ${history.length} rounds · ${result.wall_time_seconds}s.`,
      );
    } catch (caught) {
      stopAmbientAnimation();
      const message = caught instanceof Error ? caught.message : "Federated run failed.";
      setError(message);
      setFederatedConfig({ running: false, completed: false });
      toast.error("Federated run failed", message);
    }
  };

  const hasResults = federatedState.lossHistory.length > 0;
  const finalAccuracy = hasResults
    ? federatedState.lossHistory[federatedState.lossHistory.length - 1]?.valAccuracy
    : undefined;

  return (
    <div className="page-frame space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-2xl text-[15px] leading-relaxed text-text-secondary">
          Each client trains on its own shard; the aggregator runs FedAvg and
          reports genuine validation loss and accuracy — computed on the backend,
          never by pooling the underlying data.
        </p>
        <span className="inline-flex shrink-0 items-center gap-1.5 rounded-[3px] border border-success/40 bg-success/8 px-2 py-0.5 font-mono text-[11px] uppercase tracking-wider text-success">
          <span className="status-dot status-online" /> Live FedAvg
        </span>
      </div>

      <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
        {/* Controls */}
        <aside className="animate-enter-1 leaf h-fit p-5">
          <p className="section-kicker">Configuration</p>
          <div className="mt-5 space-y-5">
            <RangeField
              label="Clients"
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
            <button
              type="button"
              className="button-primary w-full"
              onClick={runSimulation}
              disabled={federatedState.running}
            >
              {federatedState.running ? (
                <>
                  <span className="button-spinner" />
                  Training…
                </>
              ) : (
                "Run federated round"
              )}
            </button>
            <p className="text-xs leading-relaxed text-text-muted">
              Runs against{" "}
              <span className="font-mono text-text-secondary">{backendConfig.mainUrl}</span>.
              No API key required.
            </p>
          </div>
        </aside>

        {/* Results */}
        <section className="animate-enter-2 space-y-6">
          {error ? (
            <StateCard tone="error" title="Could not run federated learning" message={error} />
          ) : null}

          <FederatedNetwork
            clients={federatedState.clients}
            running={federatedState.running}
            round={currentRound}
            phase={phase}
            activeNode={activeNode}
          />

          <div className="leaf p-6">
            <div className="grid gap-6 lg:grid-cols-[200px_minmax(0,1fr)]">
              <div className="space-y-5">
                <div>
                  <p className="section-kicker">Best Val Loss</p>
                  <AnimatedNumber
                    value={federatedState.bestValLoss || 0}
                    formatter={(value) =>
                      federatedState.bestValLoss === null ? "—" : value.toFixed(4)
                    }
                    className="mt-2 block font-display text-4xl font-medium text-text-primary tabular"
                  />
                </div>
                {finalAccuracy !== undefined ? (
                  <div>
                    <p className="section-kicker">Final Val Accuracy</p>
                    <p className="mt-1.5 font-mono text-2xl font-medium text-success tabular">
                      {(finalAccuracy * 100).toFixed(1)}%
                    </p>
                  </div>
                ) : null}
                {federatedState.completed ? (
                  <p className="border-l-2 border-l-success pl-3 text-xs leading-relaxed text-text-secondary">
                    Converged at round{" "}
                    <span className="font-mono text-text-primary">
                      {(federatedState.bestRound ?? 0) + 1}
                    </span>
                    {federatedState.stoppedEarly ? " (early stop)" : ""} ·{" "}
                    {federatedState.wallTimeSeconds}s on the backend.
                  </p>
                ) : null}
              </div>
              <div className="h-[260px]">
                {hasResults ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={federatedState.lossHistory}
                      margin={{ top: 8, right: 8, bottom: 0, left: -16 }}
                    >
                      <defs>
                        <linearGradient id="lossFill" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="rgb(var(--color-accent))" stopOpacity={0.16} />
                          <stop offset="100%" stopColor="rgb(var(--color-accent))" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgb(var(--color-border))"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="round"
                        tick={{ fill: "rgb(var(--color-text-muted))", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fill: "rgb(var(--color-text-muted))", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        domain={["auto", "auto"]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "rgb(var(--color-bg-surface))",
                          border: "1px solid rgb(var(--color-border-strong))",
                          borderRadius: "4px",
                          color: "rgb(var(--color-text-primary))",
                          fontSize: "12px",
                        }}
                        formatter={(value: number, name: string) => [
                          typeof value === "number" ? value.toFixed(4) : value,
                          name === "loss" ? "Val Loss" : "Val Accuracy",
                        ]}
                        labelFormatter={(label) => `Round ${label}`}
                      />
                      <Area
                        type="monotone"
                        dataKey="loss"
                        name="loss"
                        stroke="rgb(var(--color-accent))"
                        fill="url(#lossFill)"
                        strokeWidth={2}
                        dot={{ r: 2.5, fill: "rgb(var(--color-accent))" }}
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="valAccuracy"
                        name="valAccuracy"
                        stroke="rgb(var(--color-warning))"
                        strokeWidth={1.75}
                        strokeDasharray="4 3"
                        dot={false}
                        isAnimationActive={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <StateCard
                    title="Convergence Curve"
                    message="Run a round to plot real validation loss and accuracy."
                  />
                )}
              </div>
            </div>
          </div>

          <div className="leaf p-6">
            <p className="section-kicker">Round History</p>
            {hasResults ? (
              <div className="mt-4 overflow-x-auto">
                <table className="ledger-table">
                  <thead>
                    <tr>
                      <th>Round</th>
                      <th>Val Loss</th>
                      <th>Val Acc</th>
                      <th>Δ Loss</th>
                    </tr>
                  </thead>
                  <tbody>
                    {federatedState.lossHistory.map((item, index) => {
                      const previous = federatedState.lossHistory[index - 1]?.loss;
                      const delta =
                        previous === undefined ? "—" : (item.loss - previous).toFixed(4);
                      const isNeg = previous !== undefined && item.loss < previous;
                      return (
                        <tr
                          key={item.round}
                          className="fade-in-row"
                          style={{ animationDelay: `${index * 0.06}s` }}
                        >
                          <td className="font-mono text-text-primary">{item.round}</td>
                          <td className="font-mono font-medium text-text-primary tabular">
                            {item.loss.toFixed(4)}
                          </td>
                          <td className="font-mono text-text-secondary tabular">
                            {item.valAccuracy !== undefined
                              ? `${(item.valAccuracy * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td
                            className={cn(
                              "font-mono tabular",
                              isNeg ? "text-success" : "text-text-muted",
                            )}
                          >
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
                title="No run yet"
                message="Configure clients, rounds, and epochs, then run a federated round."
              />
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
