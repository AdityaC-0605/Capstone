"use client";

import { create } from "zustand";

export type ToastVariant = "success" | "error" | "info";

export interface Toast {
  id: string;
  variant: ToastVariant;
  title: string;
  message?: string;
  duration?: number;
}

interface ToastState {
  toasts: Toast[];
  push: (toast: Omit<Toast, "id">) => string;
  dismiss: (id: string) => void;
}

let counter = 0;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (toast) => {
    counter += 1;
    const id = `toast_${Date.now()}_${counter}`;
    set((state) => ({
      toasts: [...state.toasts, { id, duration: 4500, ...toast }],
    }));
    return id;
  },
  dismiss: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}));

/**
 * Imperative helper usable from anywhere (including non-React callbacks).
 */
export const toast = {
  success: (title: string, message?: string) =>
    useToastStore.getState().push({ variant: "success", title, message }),
  error: (title: string, message?: string) =>
    useToastStore.getState().push({ variant: "error", title, message }),
  info: (title: string, message?: string) =>
    useToastStore.getState().push({ variant: "info", title, message }),
};
