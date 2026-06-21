"use client";

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import {
  defaultBackendConfig,
  defaultBackendStatus,
} from "@/lib/constants";
import type {
  BackendConfig,
  BackendStatus,
  FederatedLossPoint,
  FederatedState,
  NasExperiment,
  PredictionRecord,
  PredictionResponse,
  SessionSustainability,
} from "@/lib/types";

interface UiState {
  settingsOpen: boolean;
  mobileNavOpen: boolean;
  samplePredictionResponse: string;
  samplePredictionError: string;
  samplePredictionLoading: boolean;
  nasRunning: boolean;
  nasExperiments: NasExperiment[];
}

interface PulseState {
  backendConfig: BackendConfig;
  backendStatus: BackendStatus;
  predictionHistory: PredictionRecord[];
  sessionSustainability: SessionSustainability;
  federatedState: FederatedState;
  ui: UiState;
  setBackendConfig: (config: BackendConfig) => void;
  setBackendStatus: (status: BackendStatus) => void;
  addPredictionRecord: (
    input: PredictionRecord["input"],
    result: PredictionResponse,
  ) => void;
  hydrateSessionSustainability: () => void;
  openSettings: () => void;
  closeSettings: () => void;
  setMobileNavOpen: (value: boolean) => void;
  setSamplePredictionState: (payload: Partial<UiState>) => void;
  setFederatedConfig: (payload: Partial<FederatedState>) => void;
  setFederatedRunning: (running: boolean) => void;
  setFederatedHistory: (lossHistory: FederatedLossPoint[]) => void;
  setNasState: (payload: Partial<UiState>) => void;
}

const emptySession = (): SessionSustainability => ({
  totalEnergy: 0,
  totalCarbon: 0,
  totalDuration: 0,
  predictionCount: 0,
});

const summarizeHistory = (
  history: PredictionRecord[],
): SessionSustainability =>
  history.reduce(
    (acc, item) => ({
      totalEnergy: acc.totalEnergy + (item.sustainability_metrics?.energy_kwh || 0),
      totalCarbon:
        acc.totalCarbon + (item.sustainability_metrics?.carbon_emissions || 0),
      totalDuration:
        acc.totalDuration + (item.sustainability_metrics?.duration_seconds || 0),
      predictionCount:
        acc.predictionCount + (item.sustainability_metrics ? 1 : 0),
    }),
    emptySession(),
  );

export const usePulseStore = create<PulseState>()(
  persist(
    (set, get) => ({
      backendConfig: defaultBackendConfig,
      backendStatus: structuredClone(defaultBackendStatus),
      predictionHistory: [],
      sessionSustainability: emptySession(),
      federatedState: {
        running: false,
        rounds: 3,
        lossHistory: [],
        clients: 3,
        localEpochs: 2,
        completed: false,
        bestValLoss: null,
        bestRound: null,
        wallTimeSeconds: null,
        stoppedEarly: false,
        source: "live",
      },
      ui: {
        settingsOpen: false,
        mobileNavOpen: false,
        samplePredictionResponse: "",
        samplePredictionError: "",
        samplePredictionLoading: false,
        nasRunning: false,
        nasExperiments: [],
      },
      setBackendConfig: (config) => set({ backendConfig: config }),
      setBackendStatus: (status) => set({ backendStatus: status }),
      addPredictionRecord: (input, result) => {
        const entry: PredictionRecord = {
          timestamp: result.prediction_timestamp || new Date().toISOString(),
          input,
          result,
          sustainability_metrics: result.sustainability_metrics,
        };
        const nextHistory = [entry, ...get().predictionHistory].slice(0, 20);
        set({
          predictionHistory: nextHistory,
          sessionSustainability: summarizeHistory(nextHistory),
        });
      },
      hydrateSessionSustainability: () =>
        set({
          sessionSustainability: summarizeHistory(get().predictionHistory),
        }),
      openSettings: () =>
        set((state) => ({
          ui: { ...state.ui, settingsOpen: true, mobileNavOpen: false },
        })),
      closeSettings: () =>
        set((state) => ({
          ui: { ...state.ui, settingsOpen: false },
        })),
      setMobileNavOpen: (value) =>
        set((state) => ({
          ui: { ...state.ui, mobileNavOpen: value },
        })),
      setSamplePredictionState: (payload) =>
        set((state) => ({
          ui: { ...state.ui, ...payload },
        })),
      setFederatedConfig: (payload) =>
        set((state) => ({
          federatedState: { ...state.federatedState, ...payload },
        })),
      setFederatedRunning: (running) =>
        set((state) => ({
          federatedState: {
            ...state.federatedState,
            running,
            completed: running ? false : state.federatedState.completed,
          },
        })),
      setFederatedHistory: (lossHistory) =>
        set((state) => ({
          federatedState: {
            ...state.federatedState,
            lossHistory,
            bestValLoss:
              lossHistory.length > 0
                ? Math.min(...lossHistory.map((item) => item.loss))
                : null,
          },
        })),
      setNasState: (payload) =>
        set((state) => ({
          ui: { ...state.ui, ...payload },
        })),
    }),
    {
      name: "pulseledger-store",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        backendConfig: state.backendConfig,
        predictionHistory: state.predictionHistory,
      }),
      onRehydrateStorage: () => (state) => {
        state?.hydrateSessionSustainability();
      },
    },
  ),
);
