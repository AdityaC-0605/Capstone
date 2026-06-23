# Deploying PulseLedger

Three services back the UI — the **platform API** (`:8000`), the **inference API**
(`:8001`), and the **Next.js frontend** (`:3000`) — plus a **Postgres** database.
The app runs on SQLite with zero config locally; production uses Postgres via
`PULSELEDGER_DATABASE_URL`.

> **Resource note:** the API images pull `torch` (for federated learning), so
> they need ~2 GB RAM and a few minutes to build. Size the API hosts
> accordingly (the frontend is light and happy on any free tier).

---

## Option A — Docker Compose (local or a VPS)

```bash
docker compose up --build
```

- Frontend → http://localhost:3000
- Platform API → http://localhost:8000/docs · Inference API → http://localhost:8001/docs
- Sign in with the demo account: `demo@pulseledger.app` / `demo12345`

Compose wires Postgres, both APIs, and the frontend together; tables are created
automatically on first boot.

---

## Option B — Render (one-click Blueprint)

1. Push this repo to GitHub, then **Render ▸ New ▸ Blueprint** and point it at
   `render.yaml`. It provisions Postgres + the three services.
2. Use **Standard** instances for the two API services (torch will OOM on Free).
3. After the first deploy, set these env vars (they can't be auto-wired):
   - On **both APIs** → `PULSELEDGER_ALLOWED_ORIGINS` = the frontend URL
     (e.g. `https://pulseledger-frontend.onrender.com`).
   - On the **frontend** → `NEXT_PUBLIC_MAIN_URL` and `NEXT_PUBLIC_INFERENCE_URL`
     = the two API URLs, then **redeploy the frontend** (these are baked in at
     build time).

---

## Option C — Frontend on Vercel + APIs anywhere

The frontend is a standard Next.js app and deploys cleanly to Vercel:

- **Root directory:** `frontend`
- **Env vars:** `NEXT_PUBLIC_MAIN_URL`, `NEXT_PUBLIC_INFERENCE_URL` → your API URLs.

Host the two APIs via Docker (the root `Dockerfile`) on Fly/Render/a VPS:

```bash
# platform API
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
# inference API
uvicorn app.api.inference_service:asgi_app --factory --host 0.0.0.0 --port 8001
```

---

## Environment variables

| Variable | Service | Purpose |
|----------|---------|---------|
| `PULSELEDGER_DATABASE_URL` | both APIs | Postgres DSN (`postgresql://…`; `postgres://` is auto-normalized). Defaults to local SQLite. |
| `PULSELEDGER_JWT_SECRET` | inference | **Set a strong secret in production.** Signs session tokens. |
| `PULSELEDGER_ALLOWED_ORIGINS` | both APIs | Comma-separated CORS allow-list (the frontend origin). |
| `PULSELEDGER_TRUSTED_HOSTS` | inference | `*` or a comma list of allowed Host headers. Set `*` behind a platform proxy. |
| `PULSELEDGER_DEMO_PASSWORD` | inference | Seeded demo-account password (change or disable in prod). |
| `NEXT_PUBLIC_MAIN_URL` / `NEXT_PUBLIC_INFERENCE_URL` | frontend (build) | API URLs the browser calls. Inlined at build time. |

## Database migrations

Tables are bootstrapped automatically on startup (idempotent `create_all`). For
managed schema changes use Alembic:

```bash
PULSELEDGER_DATABASE_URL=postgresql://… alembic upgrade head
```

## Production checklist

- [ ] Set a strong `PULSELEDGER_JWT_SECRET`.
- [ ] Change or remove the seeded demo account (`PULSELEDGER_DEMO_PASSWORD`).
- [ ] Lock `PULSELEDGER_ALLOWED_ORIGINS` to your frontend origin (not `*`).
- [ ] Use a managed Postgres with backups; run `alembic upgrade head` on deploy.
- [ ] Serve everything over HTTPS (handled by Render/Vercel automatically).
