import type { Metadata } from "next";

import "@/app/globals.css";
import { Providers } from "@/components/providers";
import { SiteShell } from "@/components/site-shell";

export const metadata: Metadata = {
  title: "PulseLedger — Credit Risk Intelligence",
  description:
    "Production-grade command center for explainable credit risk, federated learning, and carbon-aware AI operations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&family=Sora:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-bg-primary font-sans text-text-primary antialiased">
        <Providers>
          <SiteShell>{children}</SiteShell>
        </Providers>
      </body>
    </html>
  );
}
