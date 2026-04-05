import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "rgb(var(--color-bg-primary) / <alpha-value>)",
          surface: "rgb(var(--color-bg-surface) / <alpha-value>)",
          elevated: "rgb(var(--color-bg-elevated) / <alpha-value>)",
        },
        border: {
          DEFAULT: "rgb(var(--color-border) / <alpha-value>)",
          strong: "rgb(var(--color-border-strong) / <alpha-value>)",
        },
        text: {
          primary: "rgb(var(--color-text-primary) / <alpha-value>)",
          secondary: "rgb(var(--color-text-secondary) / <alpha-value>)",
          muted: "rgb(var(--color-text-muted) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--color-accent) / <alpha-value>)",
          hover: "rgb(var(--color-accent-hover) / <alpha-value>)",
        },
        success: "rgb(var(--color-success) / <alpha-value>)",
        warning: "rgb(var(--color-warning) / <alpha-value>)",
        destructive: "rgb(var(--color-destructive) / <alpha-value>)",
        
        // Backward compats for any un-migrated components until Phase 2 is complete
        teal: "rgb(var(--color-accent) / <alpha-value>)",
        emerald: "rgb(var(--color-success) / <alpha-value>)",
        gold: "rgb(var(--color-warning) / <alpha-value>)",
        rose: "rgb(var(--color-destructive) / <alpha-value>)",
        violet: "rgb(var(--color-accent) / <alpha-value>)",
        amber: "rgb(var(--color-warning) / <alpha-value>)",
        red: "rgb(var(--color-destructive) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["var(--font-body)"],
        display: ["var(--font-display)"],
        mono: ["var(--font-mono)"],
      },
      spacing: {
        // Enforcing 4px/8px rhythm as explicitly mapped if users use arbitrary values
        "0.25": "1px",
        "0.5": "2px",
        "1": "4px",
        "2": "8px",
        "3": "12px",
        "4": "16px",
        "5": "20px",
        "6": "24px",
        "8": "32px",
        "10": "40px",
        "12": "48px",
        "16": "64px",
      },
      animation: {
        "enter": "slide-up-fade 400ms cubic-bezier(0.16, 1, 0.3, 1) both",
      },
    },
  },
  plugins: [],
};
export default config;
