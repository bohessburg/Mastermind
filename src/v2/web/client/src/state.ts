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
  undoPendingSeat?: number;
  undoOfferSeat?: number;
  undoNotice?: string;
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
    case 'undo_pending':
      return {
        ...state,
        undoPendingSeat: message.seat,
        undoOfferSeat: undefined,
        undoNotice: undefined,
      };
    case 'undo_offer':
      return {
        ...state,
        undoPendingSeat: undefined,
        undoOfferSeat: message.seat,
        undoNotice: undefined,
      };
    case 'undo_result': {
      let undoNotice: string | undefined;
      if (!message.accepted) {
        switch (message.reason) {
          case 'denied':
            undoNotice = 'Opponent declined the undo request.';
            break;
          case 'game_advanced':
            undoNotice = 'Undo request expired because the game advanced.';
            break;
          case 'disconnect':
            undoNotice = 'Undo request was cancelled because a player disconnected.';
            break;
        }
      }
      return { ...state, undoPendingSeat: undefined, undoOfferSeat: undefined, undoNotice };
    }
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

const BASIC_SUPPLY_ORDER = ['Copper', 'Silver', 'Gold', 'Estate', 'Duchy', 'Province', 'Curse'];

/** Returns a display order without changing the server's pile array. */
export function orderSupplyPiles(
  piles: CountedCard[],
  defsById: ReadonlyMap<number, Pick<CardDef, 'name' | 'cost'>>,
): CountedCard[] {
  const basicOrder = new Map(BASIC_SUPPLY_ORDER.map((name, index) => [name, index]));

  return [...piles].sort((left, right) => {
    const leftDef = defsById.get(left.def);
    const rightDef = defsById.get(right.def);
    const leftBasicOrder = basicOrder.get(leftDef?.name ?? '');
    const rightBasicOrder = basicOrder.get(rightDef?.name ?? '');

    if (leftBasicOrder !== undefined || rightBasicOrder !== undefined) {
      if (leftBasicOrder === undefined) {
        return 1;
      }
      if (rightBasicOrder === undefined) {
        return -1;
      }
      return leftBasicOrder - rightBasicOrder;
    }

    const costDifference = (leftDef?.cost.coins ?? Infinity) - (rightDef?.cost.coins ?? Infinity);
    if (costDifference !== 0) {
      return costDifference;
    }

    const nameDifference = (leftDef?.name ?? `Card ${left.def}`).localeCompare(rightDef?.name ?? `Card ${right.def}`);
    return nameDifference !== 0 ? nameDifference : left.def - right.def;
  });
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
