# syntax=docker/dockerfile:1

# Keep the Python ABI identical in the extension builder and runtime image.
ARG PYTHON_VERSION=3.12
ARG CHECKPOINT_SOURCE=checkpoints/remote/campaign15/gen_0045.pt

FROM python:${PYTHON_VERSION}-slim AS engine

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake g++ ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir pybind11==2.13.6

# The CMake graph unconditionally defines v2_fuzz from tests/v2 even with
# BUILD_TESTS=OFF, so tests/ must be present to configure.
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY tests/ ./tests/

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON=ON \
        -DBUILD_TESTS=OFF \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build build --target dominion_v2_py


FROM node:20-slim AS client

WORKDIR /app/src/v2/web/client

COPY src/v2/web/client/package.json src/v2/web/client/package-lock.json ./
RUN npm ci

# The client imports the generated card definitions from its sibling directory.
COPY src/v2/web/client/ /app/src/v2/web/client/
COPY src/v2/web/client-data/ /app/src/v2/web/client-data/
RUN npm run build


FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.1 \
    && python -m pip install --no-cache-dir \
        fastapi==0.115.12 \
        'uvicorn[standard]==0.34.2' \
        numpy==2.2.6

COPY --from=engine /app/build/dominion_v2_py*.so /app/build/
COPY --from=engine /app/src/v2/train/ /app/src/v2/train/
COPY --from=engine /app/src/v2/web/server/ /app/src/v2/web/server/
COPY --from=engine /app/src/v2/web/client-data/ /app/src/v2/web/client-data/
COPY --from=client /app/src/v2/web/client/dist/ /app/src/v2/web/client/dist/

# An absent build arg source makes Docker's COPY fail with the exact missing
# path; the explicit non-empty check also catches an accidentally empty file.
ARG CHECKPOINT_SOURCE=checkpoints/remote/campaign15/gen_0045.pt
COPY ${CHECKPOINT_SOURCE} /app/checkpoints/gen_0045.pt
RUN test -s /app/checkpoints/gen_0045.pt \
    || { echo >&2 "ERROR: checkpoint is missing or empty: ${CHECKPOINT_SOURCE}"; exit 1; }

ENV PYTHONPATH=/app/build \
    PYTHONUNBUFFERED=1 \
    DOMINION_NN_CHECKPOINT=/app/checkpoints/gen_0045.pt

# Finished games are persisted relative to the process working directory.
RUN mkdir -p /app/exports
VOLUME ["/app/exports"]

EXPOSE 8000

# Do not add --workers: sessions live in process memory, so multiple workers
# would route a game's WebSocket and HTTP requests to different session maps.
CMD ["uvicorn", "src.v2.web.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
