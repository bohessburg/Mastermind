import type {
  DecisionMessage,
  GameOverMessage,
  ServerMessage,
  StateMessage,
  TableMessage,
} from './protocol';

export interface ClientState {
  table?: TableMessage;
  state?: StateMessage;
  decision?: DecisionMessage;
  log: string[];
  gameover?: GameOverMessage;
  error?: string;
  connected: boolean;
}

export const initialClientState: ClientState = {
  log: [],
  connected: false,
};

export function reduceServerMessage(state: ClientState, message: ServerMessage): ClientState {
  switch (message.type) {
    case 'table':
      return { ...state, table: message, error: undefined };
    case 'state':
      return { ...state, state: message, error: undefined };
    case 'decision':
      return { ...state, decision: message, error: undefined };
    case 'log':
      return { ...state, log: [...state.log, ...message.lines], error: undefined };
    case 'gameover':
      return { ...state, gameover: message, error: undefined };
    case 'error':
      return { ...state, error: message.message };
    default:
      return state;
  }
}

export function canSendDone(decision: DecisionMessage | undefined): boolean {
  return Boolean(decision?.options.some((option) => option.label === 'Done' || option.label === 'Pass'));
}
