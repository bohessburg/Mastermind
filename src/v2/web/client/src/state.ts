import type {
  CardDef,
  CountedCard,
  DecisionMessage,
  DecisionOption,
  GameOverMessage,
  ServerMessage,
  StateMessage,
  TableMessage,
} from './protocol';
import type { SelectZone } from './protocol';

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

export function indexDecisionOptionsByDef(
  decision: DecisionMessage | undefined,
  verb: 'Play' | 'Buy',
): Map<number, DecisionOption> {
  const byDef = new Map<number, DecisionOption>();
  if (!decision?.options.length) {
    return byDef;
  }
  for (const option of decision.options) {
    if (option.def === undefined || !option.label.startsWith(`${verb} `)) {
      continue;
    }
    byDef.set(option.def, option);
  }
  return byDef;
}

/**
 * Supports older servers while preferring the additive wire hint.  A select
 * action can name the same definition as a card in another visible zone, so
 * callers must only use the returned map for the matching zone.
 */
export function selectZoneForDecision(decision: DecisionMessage | undefined): SelectZone | undefined {
  if (!decision) {
    return undefined;
  }
  if (decision.select_zone) {
    return decision.select_zone;
  }
  if (decision.kind === 'ChooseGain') {
    return 'supply';
  }
  if (decision.kind === 'ReactWindow') {
    return 'hand';
  }
  if (decision.kind === 'Choose') {
    if (decision.source.name === 'Harbinger') {
      return 'discard';
    }
    if (decision.source.name === 'Bandit') {
      return 'set_aside';
    }
    return 'hand';
  }
  return undefined;
}

export function indexSelectOptionsByDef(
  decision: DecisionMessage | undefined,
  zone: SelectZone,
): Map<number, DecisionOption> {
  const byDef = new Map<number, DecisionOption>();
  if (selectZoneForDecision(decision) !== zone) {
    return byDef;
  }
  for (const option of decision?.options ?? []) {
    if (option.def !== undefined) {
      byDef.set(option.def, option);
    }
  }
  return byDef;
}

export function findNextBasicTreasurePlay(
  decision: DecisionMessage | undefined,
  defsById: ReadonlyMap<number, Pick<CardDef, 'is_basic_treasure'>>,
): DecisionOption | undefined {
  if (!decision || decision.kind !== 'PhaseBuy' || decision.options.length === 0) {
    return undefined;
  }

  let best: DecisionOption | undefined;
  for (const option of decision.options) {
    if (option.def === undefined || !option.label.startsWith('Play ')) {
      continue;
    }
    if (!defsById.get(option.def)?.is_basic_treasure) {
      continue;
    }
    if (!best || option.def > (best.def ?? -1)) {
      best = option;
    }
  }
  return best;
}

export function formatTrashEntries(
  trash: CountedCard[],
  nameForDef: (def: number) => string | undefined,
): string[] {
  if (trash.length === 0) {
    return ['Trash is empty'];
  }
  return trash
    .filter((card) => card.count > 0)
    .map((card) => `${nameForDef(card.def) ?? `Card ${card.def}`} x${card.count}`);
}
