"use client";

import Link from "next/link";
import { ArrowUpRight, Plus } from "lucide-react";

import { StateCard } from "@/components/state-card";
import {
  formatCurrency,
  formatFeatureLabel,
  formatRiskLabel,
  relativeTime,
  riskColorClasses,
} from "@/lib/format";
import { cn } from "@/lib/utils";
import { usePulseStore } from "@/store/use-pulse-store";

export default function AssessmentsPage() {
  const history = usePulseStore((state) => state.predictionHistory);

  return (
    <div className="space-y-7">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-xl text-[15px] leading-relaxed text-text-secondary">
          Every application you have scored this session, most recent first.
          Open any row for its full explanation.
        </p>
        <div className="flex shrink-0 gap-2">
          <Link href="/assessments/batch" className="button-ghost">
            Bulk score
          </Link>
          <Link href="/assessments/new" className="button-primary">
            <Plus className="h-4 w-4" />
            New assessment
          </Link>
        </div>
      </div>

      {history.length > 0 ? (
        <div className="leaf overflow-hidden">
          <div className="overflow-x-auto">
            <table className="ledger-table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>ID</th>
                  <th>Applicant</th>
                  <th>Amount</th>
                  <th>Purpose</th>
                  <th>Risk</th>
                  <th>Band</th>
                  <th>Conf.</th>
                  <th aria-hidden="true" />
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.result.prediction_id} className="group">
                    <td className="whitespace-nowrap text-text-muted">
                      {relativeTime(item.timestamp)}
                    </td>
                    <td className="font-mono text-[11px] text-text-muted">
                      {item.result.prediction_id.replace("pred_", "")}
                    </td>
                    <td className="whitespace-nowrap text-text-secondary">
                      {item.input.age}y · {formatCurrency(item.input.income)}
                    </td>
                    <td className="font-mono text-text-primary tabular">
                      {formatCurrency(item.input.loan_amount)}
                    </td>
                    <td className="whitespace-nowrap text-text-secondary">
                      {formatFeatureLabel(item.input.loan_purpose)}
                    </td>
                    <td className="font-mono font-medium text-text-primary tabular">
                      {item.result.risk_score.toFixed(2)}
                    </td>
                    <td>
                      <span
                        className={cn(
                          "rounded-[3px] border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                          riskColorClasses(item.result.risk_level),
                        )}
                      >
                        {formatRiskLabel(item.result.risk_level)}
                      </span>
                    </td>
                    <td className="font-mono text-text-secondary tabular">
                      {Math.round((item.result.confidence || 0) * 100)}%
                    </td>
                    <td className="text-right">
                      <Link
                        href={`/assessments/${item.result.prediction_id}`}
                        className="focus-ring inline-flex rounded-[3px] text-text-muted transition-colors group-hover:text-accent"
                        aria-label="Open assessment"
                      >
                        <ArrowUpRight className="h-4 w-4" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <StateCard
          title="No assessments yet"
          message="Score a credit application to start building your assessment ledger. Each run is stored here with its full explanation."
          actionLabel="Run your first assessment"
          onAction={() => {
            window.location.href = "/assessments/new";
          }}
        />
      )}
    </div>
  );
}
