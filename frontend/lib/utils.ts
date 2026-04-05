import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));
