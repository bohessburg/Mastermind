import type { ClientMessage, ServerMessage } from './protocol';

export type MessageHandler = (message: ServerMessage) => void;
export type StatusHandler = (status: 'connecting' | 'open' | 'closed') => void;

export class GameSocket {
  private socket?: WebSocket;
  private reconnectTimer?: number;
  private closedByUser = false;

  constructor(
    private readonly sessionId: string,
    private readonly seatToken: string,
    private readonly onMessage: MessageHandler,
    private readonly onStatus: StatusHandler,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.onStatus('connecting');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/${this.sessionId}/${this.seatToken}`;
    this.socket = new WebSocket(url);
    this.socket.onopen = () => this.onStatus('open');
    this.socket.onmessage = (event) => {
      this.onMessage(JSON.parse(event.data) as ServerMessage);
    };
    this.socket.onclose = () => {
      this.onStatus('closed');
      if (!this.closedByUser) {
        this.reconnectTimer = window.setTimeout(() => this.connect(), 1000);
      }
    };
  }

  send(message: ClientMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  close(): void {
    this.closedByUser = true;
    if (this.reconnectTimer !== undefined) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
    this.socket?.close();
  }
}
