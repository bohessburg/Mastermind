# DominionZero v2 Web Client

Text-only browser playtest harness for the v2 engine.

## Run

From the repo root:

```bash
PYTHONPATH=build .venv/bin/python -m uvicorn src.v2.web.server.main:app --reload --host 127.0.0.1 --port 8000
```

From `src/v2/web/client`:

```bash
npm install
npm run dev
```

The Vite dev server proxies `/api` and `/ws` to `localhost:8000`.

## Build

```bash
npm run test
npm run build
```

The FastAPI server serves `dist/` when it exists, so the production mode can run as one process.
