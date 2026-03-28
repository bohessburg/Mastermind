FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Build tools + C++20 compiler
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 (Claude Code runtime)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Non-root user (required for --dangerously-skip-permissions)
RUN useradd -m -s /bin/bash dev
USER dev

WORKDIR /workspace

COPY --chmod=755 docker-entrypoint.sh /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
