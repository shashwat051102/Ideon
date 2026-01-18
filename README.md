# Ideon

Map your ideas, reveal connections, and turn clusters into clear writing. Ideon is a lightweight, Flask-based app that lets you capture ideas, visualize relationships as a graph, and synthesize “collective ideas” with optional LLM help. Data is isolated per Voice profile and secured with cookie-based auth.

## Features

- Voice ID login and per‑profile isolation (DB + vector store)
- Smart autolinking using cosine similarity with tuned thresholds and a resilient fallback when the vector index is sparse
- Collective idea synthesis from a set of linked nodes
- Auto‑tagging: simple keyword extraction to enrich new ideas for search and clustering
- Map‑first UI with Cytoscape: inspect nodes/edges (distance, tag overlap), run autolink presets, synthesize, reset
- Production‑ready HTTP basics: HTTPS redirect support behind a load balancer, HSTS header, health endpoint
## Status: active, single‑repo app
## Stack: Flask + SQLite + ChromaDB + Cytoscape
## OS: Windows/macOS/Linux (dev), Docker (prod)

## Architecture at a glance


Directory highlights

```
app/                 Flask app, routes and templates
core/                Core modules (database, models, pipelines, agents)
	database/          SQLite + Chroma managers
	models/            Embeddings & generator (LLM/local)
	crews/             Agents (graph, idea, style, reflection)
	pipelines/         Ingest/graph/generation pipelines
data/                Sample writing/notes
storage/             Local storage (SQLite, Chroma)
tests/               Pytests for database, embeddings, routes, etc.
tools/               Utilities (reset/reindex, inspection)
```

## Requirements


## Quick start (local)

PowerShell (Windows):

```powershell
# 1) Create & activate venv
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) enable sentence-transformers + LLM
# $env:IDEON_USE_ST = '1'                 # enable Sentence-Transformers
# $env:OPENAI_API_KEY = '<your-key>'      # enable LLM routes

# 4) Run
python -m app
# or
python run.py
```

Then open http://127.0.0.1:5000

First time flow

1) Go to “Voice” to create a profile and get a Voice ID token
# Ideon

Map your ideas, reveal connections, and turn clusters into clear writing. Ideon is a lightweight, Flask‑based app to capture short ideas, visualize the relationships as an interactive graph, and synthesize “collective ideas.” Each user operates inside a private Voice profile, isolated at both the database and vector‑store layers.

- Status: Active, single‑repo app
- Stack: Flask + SQLite + ChromaDB + Cytoscape + Gunicorn
- Runs: Windows/macOS/Linux locally; Docker in production

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Directory layout](#directory-layout)
- [Requirements](#requirements)
- [Local development](#local-development)
- [Configuration (env vars)](#configuration-env-vars)
- [Docker usage](#docker-usage)
- [Data and storage](#data-and-storage)
- [API reference](#api-reference-selected)
- [Autolinking logic](#autolinking-logic)
- [Testing](#testing)
- [Deployment (AWS)](#deployment-guides-aws)
- [Cloudflare Tunnel (no public IPv4)](#cloudflare-tunnel-no-public-ipv4-optional)
- [Cost controls](#cost-controls-on-aws)
- [Security and privacy](#security-and-privacy)
- [Operations](#operations-health-logs-backup)
- [Troubleshooting](#troubleshooting)
- [Roadmap & contributions](#roadmap--contributions)
- [License](#license)

---

## Overview

Ideon helps you work the way thoughts arrive: quick notes first, structure later. Add ideas, let embeddings connect the dots, and promote clusters into clean writing. Everything is scoped to your Voice profile so multiple people (or personas) can safely use the same backend without seeing each other’s data.

## Features

- Voice‑scoped authentication with a profile token stored in a cookie
- Map UI powered by Cytoscape: nodes (ideas), edges (semantic links), contextual actions
- Autolinking with distance/cosine thresholds plus a resilient fallback when the vector index is sparse
- Collective idea synthesis from selected nodes (optional LLM)
- Auto‑tagging on create to enrich search and clustering
- JSON‑first API behavior for unauthenticated requests (401 with `login_required`)
- Health endpoint and production‑safe HTTP behavior (redirect to HTTPS behind a LB, HSTS)

## Architecture

- Web: Flask app factory, blueprints under `app/routes`
- Data: SQLite for metadata, ChromaDB for vectors
- Models: local hashing or Sentence‑Transformers (opt‑in) for embeddings; optional OpenAI for text generation
- Frontend: Bootstrap + Jinja templates, Cytoscape for graph view and controls
- Process: gunicorn serves the app in Docker/EB

## Directory layout

```
app/                 Flask app, routes and templates
core/                Core modules (database, models, pipelines, agents)
	database/          SQLite + Chroma managers
	models/            Embeddings & generator (LLM/local)
	crews/             Agents (graph, idea, style, reflection)
	pipelines/         Ingest/graph/generation pipelines
storage/             Local storage (SQLite, Chroma)
data/                Sample notes & writing seeds
tests/               Pytests for DB, embeddings, routes, etc.
tools/               Utilities (reset, reindex, inspect)
```

## Requirements

- Python 3.11+ (3.12 OK)
- Windows/macOS/Linux
- Optional: Docker 24+

## Local development

PowerShell (Windows):

```powershell
# 1) Create & activate venv
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Optional: enable ST embeddings & LLM
# $env:IDEON_USE_ST = '1'
# $env:OPENAI_API_KEY = '<your-key>'

# 4) Run the app
python -m app
# or
python run.py
```

Open http://127.0.0.1:5000

First‑time flow:

1) Visit Voice → create a profile (you’ll get a token like `name-1234`)
2) Visit Login → paste the token to access your private map

VS Code tasks

- Task: `pytest database + embeddings` runs a fast subset:

```powershell
# From VS Code: Terminal → Run Task → pytest database + embeddings
# Or run directly
pytest -q tests/test_database.py tests/test_embeddings.py
```

## Configuration (env vars)

- `IDEON_DEBUG` / `FLASK_DEBUG`: `1` to enable Flask debug locally
- `IDEON_FORCE_HTTPS` (or `IDEAWEAVER_FORCE_HTTPS`): `1` to enforce HTTP→HTTPS redirects behind a LB
- `IDEON_USE_ST`: `1` to use Sentence‑Transformers embeddings (otherwise a hashing fallback is used)
- `OPENAI_API_KEY`: enable generator/LLM features
- `IDEON_LLM_MODEL`: override model name (default `gpt-4.1-mini`)
- `IDEON_COLLECTIVE_REQUIRE_LLM`: `1` to require LLM for collective synthesis

Example `.env` (local):

```
FLASK_DEBUG=1
IDEON_USE_ST=0
OPENAI_API_KEY=
```

## Docker usage

The repo includes `dockerfile` (lower‑case). Build and run:

```powershell
docker build -t ideon .
docker run --rm -p 5000:5000 --name ideon ideon
start http://localhost:5000
```

## Data and storage

- SQLite: `storage/sqlite/metadata.db`
- Chroma: `storage/chromadb/`
- Back up both to fully restore an environment

## API reference (selected)

Auth behavior: APIs return 401 JSON `{ "login_required": true }` if the `vp_token` cookie is missing/invalid.

- `GET /api/graph` → profile‑scoped nodes & edges
- `POST /api/graph/autolink` → { preset | thresholds } → creates edges among recent nodes
- `POST /api/graph/context_map` → { query } → nearest ideas
- `POST /api/graph/collective` → { seedNodeIds | selection } → synthesized paragraph
- `POST /api/graph/reset` → clears profile DB rows; vectors cleared via Chroma util

Example: autolink request

```json
{
	"preset": "loose",
	"max_edges": 10
}
```

Response (abridged):

```json
{
	"created_edges": [
		{"source": "n1", "target": "n7", "weight": 0.82}
	],
	"stats": {"considered": 25, "linked": 6}
}
```

## Autolinking logic

For a node, Chroma nearest‑neighbors are queried within the same `voice_profile_id`. If too few exist, a local cosine fallback computes similarities over recent nodes. Links are created if distance is small or cosine exceeds `min_cosine`. Strict mode can require mutual‑nearest constraints. Presets in the UI tune these numbers.

## Testing

```powershell
pytest -q

# Fast subset
pytest -q tests/test_database.py tests/test_embeddings.py
```

VS Code tasks (available in this repo) also run the DB/embeddings subset.

## Deployment guides (AWS)

Container image: `dockerfile` (Python 3.11 slim + gunicorn)

Health: `GET /healthz` → 200 OK

### Option A — Elastic Beanstalk (Single‑instance)

Free‑tier friendly. No load balancer.

1) Create environment → Web server → Docker (AL2023) → Single instance
2) Instance type: free‑tier eligible (`t2.micro`/`t3.micro`)
3) Capacity: min=1, max=1; Health check URL: `/healthz`
4) Networking: public subnet(s) with public IPv4 enabled; do NOT use NAT Gateways
5) Deploy and test `http://<env>.elasticbeanstalk.com/healthz`
6) DNS: `CNAME app.ideon.online → <env>.elasticbeanstalk.com`
7) HTTPS:
	 - Quickest: Cloudflare proxy (orange cloud) → free TLS; instance can stay on HTTP:80
	 - Or use certbot on the instance and open 443 in the instance SG

### Option B — Elastic Beanstalk (Load‑balanced with ACM)

1) Environment type: Load balanced, min=1 max=1
2) ACM certificate in the same region for `app.ideon.online` (Issued)
3) ALB listeners:
	 - 443: HTTPS → Instance HTTP:80 (attach ACM cert)
	 - 80: redirect to 443
4) Public subnets only (instances must have Public IP); no NAT GW
5) DNS: `CNAME app.ideon.online → <env>.elasticbeanstalk.com`

### EB click‑ops (quick path)

1) EB → Environments → Create → Web server → Docker (AL2023)
2) Environment type: Single instance (for free‑tier) or Load balanced (if you need ALB)
3) Capacity: Min 1, Max 1; Health check URL `/healthz`
4) Networking: choose Public subnets; enable Public IP for instances
5) If Load balanced: add HTTPS listener with your ACM cert; forward to HTTP:80
6) Deploy; wait for Health “Green”; test `/healthz`
7) Route 53 (or your DNS): CNAME `app.yourdomain.com` → EB CNAME



## Cost controls on AWS

- Public IPv4 addresses are billed hourly (small but not free). To reach literal $0 for networking, remove public IPv4 and use a Cloudflare Tunnel + IPv6 (advanced), or move to a free host.
- Avoid NAT Gateways and Interface VPC Endpoints for this app.
- Budgets: set a monthly budget (e.g., $5) with 50%/80%/100% alerts.

## Security and privacy

- Token in cookie `vp_token` gates all routes by default; APIs return 401 JSON instead of HTML redirects
- All persistence is scoped to `voice_profile_id`
- The token shown on the Voice page is sensitive; keep it private
- Enable HSTS + HTTPS redirect when behind a load balancer

## Operations (health, logs, backup)

- Health: `GET /healthz`
- Logs: Flask/gunicorn logs; set CloudWatch Logs retention (e.g., 7 days) on AWS
- Backup: copy `storage/sqlite/metadata.db` and `storage/chromadb/`

## Troubleshooting

- Autolink JSON error with `<` seen
	- You were unauthenticated; APIs now return 401 JSON. Log in or ensure your client handles 401s.
- Too many redirects on `/login`
	- Clear stale cookie; GET `/login` validates and renders the form if invalid.
- No links are created
	- Try the “Loose” preset; add more ideas; the cosine fallback helps when vectors are sparse.
- 443 timeouts behind ALB
	- Map ALB 443 → instance HTTP:80 (not 443). Keep public subnets; no NAT.

## Roadmap & contributions

- Inline editing and quick‑capture
- Token rotation UI for Voice profiles
- Stronger clustering + topic labels
- Export pipelines (Markdown, Notion)
- Optional ALB + WAF hardening

Contributions: open an issue or PR. Keep changes focused and include a short rationale and tests when possible.

## License

MIT (see `LICENSE`).


