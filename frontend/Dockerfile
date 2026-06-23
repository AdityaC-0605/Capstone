# Next.js frontend — multi-stage build producing a small standalone server.
# NEXT_PUBLIC_* are inlined at build time, so they are passed as build args.

# ---- dependencies ----
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- build ----
FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_MAIN_URL
ARG NEXT_PUBLIC_INFERENCE_URL
ENV NEXT_PUBLIC_MAIN_URL=$NEXT_PUBLIC_MAIN_URL \
    NEXT_PUBLIC_INFERENCE_URL=$NEXT_PUBLIC_INFERENCE_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- runtime ----
FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
