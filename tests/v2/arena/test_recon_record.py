from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

import pytest


async def _read_websocket_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    header = await reader.readexactly(2)
    opcode = header[0] & 0x0F
    payload_length = header[1] & 0x7F
    masked = bool(header[1] & 0x80)

    if payload_length == 126:
        payload_length = int.from_bytes(await reader.readexactly(2), "big")
    elif payload_length == 127:
        payload_length = int.from_bytes(await reader.readexactly(8), "big")

    mask = await reader.readexactly(4) if masked else b""
    payload = await reader.readexactly(payload_length)
    if masked:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return opcode, payload


async def _write_websocket_frame(
    writer: asyncio.StreamWriter,
    opcode: int,
    payload: bytes,
) -> None:
    if len(payload) < 126:
        header = bytes((0x80 | opcode, len(payload)))
    elif len(payload) < 2**16:
        header = bytes((0x80 | opcode, 126)) + len(payload).to_bytes(2, "big")
    else:
        header = bytes((0x80 | opcode, 127)) + len(payload).to_bytes(8, "big")
    writer.write(header + payload)
    await writer.drain()


async def _serve_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    html: bytes,
) -> None:
    try:
        request = await reader.readuntil(b"\r\n\r\n")
        lines = request.decode("iso-8859-1").split("\r\n")
        request_target = lines[0].split()[1]
        headers = {
            name.strip().lower(): value.strip()
            for line in lines[1:]
            if ":" in line
            for name, value in [line.split(":", 1)]
        }

        if request_target != "/ws":
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(html)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + html
            )
            await writer.drain()
            return

        key = headers["sec-websocket-key"].encode("ascii")
        accept = base64.b64encode(
            hashlib.sha1(key + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest()
        )
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: "
            + accept
            + b"\r\n\r\n"
        )
        await writer.drain()

        while True:
            opcode, payload = await _read_websocket_frame(reader)
            if opcode in {0x1, 0x2}:
                await _write_websocket_frame(writer, opcode, payload)
            elif opcode == 0x8:
                await _write_websocket_frame(writer, 0x8, payload)
                return
            elif opcode == 0x9:
                await _write_websocket_frame(writer, 0xA, payload)
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    finally:
        writer.close()
        await writer.wait_closed()


async def _start_echo_server() -> tuple[asyncio.AbstractServer, str]:
    server: asyncio.AbstractServer
    html = b""

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _serve_connection(reader, writer, html=html)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    html = f"""<!doctype html>
<script>
const socket = new WebSocket("ws://127.0.0.1:{port}/ws");
let echoes = 0;
socket.addEventListener("open", () => {{
  socket.send("hello arena");
  socket.send(new Uint8Array([0, 1, 2, 255]));
}});
socket.onmessage = () => {{
  echoes += 1;
  if (echoes === 2) socket.close();
}};
socket.addEventListener("close", () => {{ window.__arenaDone = true; }});
</script>""".encode("utf-8")
    return server, f"http://127.0.0.1:{port}/"


async def _require_chromium() -> None:
    pytest.importorskip("playwright.async_api", reason="Playwright is not installed")
    from playwright.async_api import Error, async_playwright

    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except Error as error:
            if "Executable doesn't exist" in str(error):
                pytest.skip(f"Playwright Chromium is not installed: {error}")
            raise
        await browser.close()


async def _record_echo_exchange(tmp_path: Path) -> Path:
    from src.v2.arena.recon.record import ArenaRecorder

    server, url = await _start_echo_server()
    recorder = ArenaRecorder(
        tmp_path / "recordings",
        tmp_path / "profile",
        screenshot_interval=60,
        dom_interval=60,
        headless=True,
    )
    try:
        run_dir = await recorder.start()
        page = await recorder.new_page()
        await page.goto(url)
        await page.wait_for_function("window.__arenaDone === true")
        frames_path = run_dir / "frames.jsonl"
        for _ in range(40):
            if '"kind":"close"' in frames_path.read_text(encoding="utf-8"):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("recorder did not receive the close event")
        return run_dir
    finally:
        await recorder.stop()
        server.close()
        await server.wait_closed()


def test_recorder_captures_bidirectional_websocket_frames(tmp_path: Path) -> None:
    asyncio.run(_require_chromium())
    run_dir = asyncio.run(_record_echo_exchange(tmp_path))

    records = [
        json.loads(line)
        for line in (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    socket_records = [record for record in records if record.get("sock") == 1]

    assert [record["kind"] for record in socket_records] == [
        "open",
        "text",
        "binary",
        "text",
        "binary",
        "close",
    ]
    assert [record["dir"] for record in socket_records] == [
        "in",
        "out",
        "out",
        "in",
        "in",
        "in",
    ]
    assert socket_records[1]["data"] == "hello arena"
    assert socket_records[2]["b64"] is True
    assert socket_records[2]["data"] == "AAEC/w=="
    assert socket_records[3]["data"] == "hello arena"
    assert socket_records[4]["b64"] is True
    assert socket_records[4]["data"] == "AAEC/w=="
