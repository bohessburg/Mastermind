# DominionZero VPS deployment

This deployment image serves the built Vite client and FastAPI application from
one origin. It is intended for a Hetzner CPX21 (x86_64 Ubuntu), but the same
Dockerfile builds natively on an ARM64 Hetzner CAX instance.

## Build on the VPS (recommended for CPX21)

Install Docker Engine and the Docker Compose plugin, then clone the deployment
branch on the VPS:

```sh
git clone --branch v2-phase1 <repository-url> Mastermind
cd Mastermind
mkdir -p checkpoints/remote/campaign15
```

Checkpoints are intentionally not in Git. Copy the 17 MB campaign 15 weight
from the development machine before building. For example, run either command
from the VPS (substitute the development host and repository path):

```sh
scp developer@dev-host:/path/to/Mastermind/checkpoints/remote/campaign15/gen_0045.pt checkpoints/remote/campaign15/
rsync -avP developer@dev-host:/path/to/Mastermind/checkpoints/remote/campaign15/gen_0045.pt checkpoints/remote/campaign15/
```

Build and start it:

```sh
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f dominion
```

The Docker build stops at the checkpoint `COPY` step and names the missing path
if the weight was not copied into the build context.

On a CPX21, Docker builds a native `linux/amd64` image. A CAX instance builds a
native `linux/arm64` image with the same commands; CPU PyTorch wheels and the
C++ binding are built for that platform automatically.

## Cross-build from an Apple Silicon Mac

The checkpoint must be present locally first. Build an AMD64 image under
Buildx, load it into the local image store, and stream it to the VPS:

```sh
docker buildx build --platform linux/amd64 -f deploy/Dockerfile \
  --build-arg CHECKPOINT_SOURCE=checkpoints/remote/campaign15/gen_0045.pt \
  -t dominionzero:latest --load .
docker save dominionzero:latest | ssh root@your-vps 'docker load'
```

After transfer, clone the repository on the VPS for the Compose and Caddy
files, then run `docker compose -f deploy/docker-compose.yml up -d`. The
Compose service uses the transferred `dominionzero:latest` image unless you
explicitly pass `--build`.

## HTTP and HTTPS

The default service is available directly at:

```text
http://YOUR_VPS_IP:8000
```

For automatic HTTPS, point a domain's A/AAAA records at the VPS and ensure
ports 80 and 443 reach it. Then start the optional Caddy profile:

```sh
DOMAIN=play.example.com docker compose -f deploy/docker-compose.yml --profile tls up -d
```

Caddy terminates TLS and proxies to `dominion:8000`; WebSocket upgrades work
through Caddy without extra configuration. The direct port 8000 mapping remains
available, so firewall it if only HTTPS should be public.

## Checkpoints and search budget

To bake a different in-repository weight into a new image, rebuild with its
path relative to the repository root:

```sh
CHECKPOINT_SOURCE=checkpoints/remote/campaign16/gen_0001.pt \
  docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d --force-recreate
```

To use a weight without rebuilding, add a temporary Compose override with an
absolute host path and change the checkpoint environment variable:

```yaml
services:
  dominion:
    environment:
      DOMINION_NN_CHECKPOINT: /weights/alternate.pt
    volumes:
      - /absolute/path/to/alternate.pt:/weights/alternate.pt:ro
```

Run it with:

```sh
docker compose -f deploy/docker-compose.yml -f /path/to/weights.override.yml up -d --force-recreate
```

`NN_MCTS_SIMS` controls the per-decision neural MCTS budget. Its default is
400. Lower values (for example, `NN_MCTS_SIMS=100`) make turns faster but make
the bot weaker; recreate the service after changing it.

## Smoke test

After the service is ready, run:

```sh
./deploy/smoke.sh http://127.0.0.1:8000
```

It verifies the SPA, creates a real human-vs-NN-MCTS session, drives the first
human action through WebSocket, observes an NN-MCTS action, and fetches the
server's session export endpoint.
