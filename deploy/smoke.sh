#!/usr/bin/env sh
set -eu

base_url=${1:-http://127.0.0.1:8000}
base_url=${base_url%/}
python_bin=${PYTHON:-python3}
tmp_dir=$(mktemp -d)

cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT HUP INT TERM

command -v curl >/dev/null 2>&1 || {
    echo "smoke: curl is required" >&2
    exit 1
}
command -v "$python_bin" >/dev/null 2>&1 || {
    echo "smoke: python3 is required for JSON and WebSocket checks" >&2
    exit 1
}

curl -fsS "$base_url/" -o "$tmp_dir/index.html"
grep -q 'id="root"' "$tmp_dir/index.html"
echo "smoke: SPA index served"

session_json=$(curl -fsS -X POST "$base_url/api/session" \
    -H 'Content-Type: application/json' \
    --data '{"seats":["human","bot:nnmcts"],"seed":20260719,"thinking_delay_ms":0}')

session_id=$(printf '%s' "$session_json" | "$python_bin" -c '
import json, sys
payload = json.load(sys.stdin)
assert isinstance(payload.get("session_id"), str) and payload["session_id"], payload
assert isinstance(payload.get("seat_tokens"), list) and payload["seat_tokens"], payload
print(payload["session_id"])
')
seat_token=$(printf '%s' "$session_json" | "$python_bin" -c '
import json, sys
print(json.load(sys.stdin)["seat_tokens"][0])
')
echo "smoke: created NN-MCTS session $session_id"

# The app deliberately exposes live state via WebSocket, not an HTTP state
# route. This standard-library client takes the first legal human actions until
# the real NN-MCTS seat responds, while the surrounding checks remain curl HTTP
# requests against the running container.
"$python_bin" - "$base_url" "$session_id" "$seat_token" <<'PY'
import base64
import json
import os
import socket
import ssl
import sys
import time
from urllib.parse import quote, urlsplit

base_url, session_id, seat_token = sys.argv[1:]
parsed = urlsplit(base_url)
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    raise SystemExit(f"smoke: expected an http(s) URL, got {base_url!r}")

port = parsed.port or (443 if parsed.scheme == "https" else 80)
prefix = parsed.path.rstrip("/")
path = f"{prefix}/ws/{quote(session_id, safe='')}/{quote(seat_token, safe='')}"
sock = socket.create_connection((parsed.hostname, port), timeout=20)
if parsed.scheme == "https":
    sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parsed.hostname)

host_header = parsed.hostname if parsed.port is None else f"{parsed.hostname}:{port}"
key = base64.b64encode(os.urandom(16)).decode("ascii")
sock.sendall(
    (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")
)

header = bytearray()
while b"\r\n\r\n" not in header:
    chunk = sock.recv(4096)
    if not chunk:
        raise SystemExit("smoke: WebSocket closed during handshake")
    header.extend(chunk)
head, buffered = bytes(header).split(b"\r\n\r\n", 1)
if not head.startswith(b"HTTP/1.1 101"):
    raise SystemExit(f"smoke: WebSocket upgrade failed: {head.decode('latin1')}")


def recv_exact(size: int) -> bytes:
    global buffered
    while len(buffered) < size:
        chunk = sock.recv(4096)
        if not chunk:
            raise EOFError("WebSocket closed")
        buffered += chunk
    result, buffered = buffered[:size], buffered[size:]
    return result


def send_json(value: dict) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    mask = os.urandom(4)
    size = len(payload)
    if size < 126:
        frame = bytes((0x81, 0x80 | size))
    elif size < 65536:
        frame = bytes((0x81, 0x80 | 126)) + size.to_bytes(2, "big")
    else:
        frame = bytes((0x81, 0x80 | 127)) + size.to_bytes(8, "big")
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(frame + mask + masked)


def recv_json() -> dict:
    while True:
        first, second = recv_exact(2)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        size = second & 0x7F
        if size == 126:
            size = int.from_bytes(recv_exact(2), "big")
        elif size == 127:
            size = int.from_bytes(recv_exact(8), "big")
        mask = recv_exact(4) if masked else b""
        payload = recv_exact(size)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        if opcode == 0x8:
            raise EOFError("WebSocket closed")
        if opcode == 0x9:
            # Clients must mask control frames too.
            pong_mask = os.urandom(4)
            pong = bytes(byte ^ pong_mask[index % 4] for index, byte in enumerate(payload))
            sock.sendall(bytes((0x8A, 0x80 | len(payload))) + pong_mask + pong)
            continue
        if opcode != 0x1:
            continue
        return json.loads(payload.decode("utf-8"))


deadline = time.monotonic() + 180
human_actions = 0
bot_line = None
try:
    while time.monotonic() < deadline:
        sock.settimeout(max(1, deadline - time.monotonic()))
        message = recv_json()
        if message.get("type") == "error":
            raise SystemExit(f"smoke: server error: {message.get('message')}")
        if message.get("type") == "log":
            for line in message.get("lines", []):
                if isinstance(line, str) and line.startswith("P2 "):
                    bot_line = line
                    break
        if bot_line is not None:
            print(f"smoke: NN-MCTS bot action observed: {bot_line}")
            break
        if message.get("type") == "decision" and message.get("seat") == 0:
            options = message.get("options") or []
            if not options:
                raise SystemExit("smoke: human decision had no legal options")
            send_json({"type": "act", "action": int(options[0]["action"])})
            human_actions += 1
            if human_actions > 16:
                raise SystemExit("smoke: human turn did not advance to NN-MCTS")
    else:
        raise SystemExit("smoke: timed out waiting for an NN-MCTS action")
finally:
    sock.close()
PY

curl -fsS "$base_url/api/session/$session_id/export" -o "$tmp_dir/export.json"
"$python_bin" - "$tmp_dir/export.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    exported = json.load(handle)
actions = exported.get("actions")
if not isinstance(actions, list) or len(actions) < 2:
    raise SystemExit(f"smoke: expected human and NN-MCTS actions in export, got {actions!r}")
print(f"smoke: session export fetched ({len(actions)} actions recorded)")
PY
