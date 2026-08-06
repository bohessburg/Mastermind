(() => {
  "use strict";

  if (window.__arenaWebSocketHookInstalled) {
    return;
  }
  window.__arenaWebSocketHookInstalled = true;

  const NativeWebSocket = window.WebSocket;
  if (typeof NativeWebSocket !== "function") {
    return;
  }

  let nextSocketId = 1;
  const socketStates = new WeakMap();
  const liveSockets = new Set();
  let activeSocket = null;

  function epochMilliseconds() {
    return Date.now();
  }

  function bytesToBase64(bytes) {
    const chunkSize = 0x8000;
    let binary = "";

    for (let start = 0; start < bytes.length; start += chunkSize) {
      const chunk = bytes.subarray(start, start + chunkSize);
      binary += String.fromCharCode(...chunk);
    }
    return btoa(binary);
  }

  async function encodePayload(payload) {
    if (typeof payload === "string") {
      return { kind: "text", data: payload, b64: false };
    }

    if (payload instanceof ArrayBuffer) {
      return {
        kind: "binary",
        data: bytesToBase64(new Uint8Array(payload)),
        b64: true,
      };
    }

    if (ArrayBuffer.isView(payload)) {
      return {
        kind: "binary",
        data: bytesToBase64(
          new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength),
        ),
        b64: true,
      };
    }

    if (payload instanceof Blob) {
      const buffer = await payload.arrayBuffer();
      return {
        kind: "binary",
        data: bytesToBase64(new Uint8Array(buffer)),
        b64: true,
      };
    }

    // WebSocket's Web IDL conversion treats other values as strings.
    return { kind: "text", data: String(payload), b64: false };
  }

  function forward(record) {
    try {
      const receiver = window.__arenaFrame;
      if (typeof receiver !== "function") {
        return undefined;
      }

      const result = receiver(record);
      if (result && typeof result.then === "function") {
        return result.catch(() => undefined);
      }
      return result;
    } catch (_error) {
      return undefined;
    }
  }

  class SocketState {
    constructor(socket) {
      this.id = nextSocketId++;
      this.url = socket.url;
      this.tail = Promise.resolve();
    }

    enqueue(direction, payload) {
      const ts = epochMilliseconds();
      this.tail = this.tail
        .then(async () => {
          const encoded = await encodePayload(payload);
          return forward({
            ts,
            sock: this.id,
            dir: direction,
            kind: encoded.kind,
            url: this.url,
            data: encoded.data,
            b64: encoded.b64,
          });
        })
        .catch(() => undefined);
    }

    enqueueLifecycle(kind) {
      const ts = epochMilliseconds();
      this.tail = this.tail
        .then(() =>
          forward({
            ts,
            sock: this.id,
            dir: "in",
            kind,
            url: this.url,
            data: "",
            b64: false,
          }),
        )
        .catch(() => undefined);
    }
  }

  class ArenaWebSocket extends NativeWebSocket {
    constructor(...args) {
      super(...args);

      const state = new SocketState(this);
      socketStates.set(this, state);

      // addEventListener is deliberately used instead of onmessage so page
      // handlers, including later onmessage assignments, remain untouched.
      this.addEventListener("open", () => {
        liveSockets.add(this);
        activeSocket = this;
        state.enqueueLifecycle("open");
      });
      this.addEventListener("close", () => {
        liveSockets.delete(this);
        if (activeSocket === this) {
          activeSocket = null;
          for (const candidate of liveSockets) {
            if (candidate.readyState === NativeWebSocket.OPEN) {
              activeSocket = candidate;
            }
          }
        }
        state.enqueueLifecycle("close");
      });
      this.addEventListener("message", (event) => {
        activeSocket = this;
        state.enqueue("in", event.data);
      });
    }

    send(payload) {
      const result = super.send(payload);
      const state = socketStates.get(this);
      if (state) {
        try {
          state.enqueue("out", payload);
        } catch (_error) {
          // Capturing must never change WebSocket's normal behavior.
        }
      }
      return result;
    }
  }

  window.WebSocket = ArenaWebSocket;
  window.__arenaSend = async (base64Bytes) => {
    let socket = activeSocket;
    if (!socket || socket.readyState !== NativeWebSocket.OPEN) {
      socket = null;
      for (const candidate of liveSockets) {
        if (candidate.readyState === NativeWebSocket.OPEN) {
          socket = candidate;
        }
      }
      activeSocket = socket;
    }
    if (!socket) {
      throw new Error("arena game WebSocket is unavailable");
    }

    const binary = atob(base64Bytes);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    socket.send(bytes);
    const state = socketStates.get(socket);
    if (state) {
      await state.tail;
    }
    return state ? state.id : null;
  };
})();
