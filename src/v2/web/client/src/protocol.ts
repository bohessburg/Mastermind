export type SeatKind =
  | 'human'
  | 'bot'
  | 'bot:bigmoney'
  | 'bot:random'
  | 'bot:nn'
  | 'bot:nnmcts'
  | (string & {});

export interface SeatInfo {
  index: number;
  kind: SeatKind;
}

export interface Cost {
  coins: number;
  potion: number;
  debt: number;
}

export interface CardDef {
  id: number;
  name: string;
  cost: Cost;
  types: string[];
  vp: number;
  coin_value: number;
  is_basic_treasure: boolean;
  text: string;
}

export interface DefTable {
  version: number;
  defs: CardDef[];
}

export interface PileView {
  def: number;
  count: number;
}

export interface CountedCard {
  def: number;
  count: number;
}

export interface OpponentView {
  seat: number;
  handCount: number;
  deckCount: number;
  discardCount: number;
  discardTop: number | null;
  inPlay: number[];
  resources?: ResourceView;
  vp?: number | null;
}

export interface ResourceView {
  actions: number;
  buys: number;
  coins: number;
  potion: number;
  debt: number;
  coffers: number;
  villagers: number;
  favors: number;
  vp_tokens: number;
}

export interface SeatStateView {
  piles: PileView[];
  myHand: CountedCard[];
  myPlayArea: number[];
  myDeckCount: number;
  myDiscardCount: number;
  myDiscardTop: number | null;
  mySetAside?: number[];
  opponents: OpponentView[];
  trash: CountedCard[];
  trashTop: number | null;
  resources: ResourceView;
  phase: number;
  turn: number;
}

export interface DecisionOption {
  action: number;
  label: string;
  def?: number;
  name?: string;
}

/** The visible card zone a card-select decision refers to. */
export type SelectZone = 'hand' | 'supply' | 'discard' | 'set_aside';

export interface TableMessage {
  type: 'table';
  seats: SeatInfo[];
  kingdom: number[];
  landscapes: number[];
}

export interface StateMessage {
  type: 'state';
  view: SeatStateView;
}

export interface DecisionMessage {
  type: 'decision';
  seat: number;
  kind: string;
  source: { def: number; name?: string };
  prompt: string;
  options: DecisionOption[];
  min: number;
  max: number;
  /** Additive server hint for A_SELECT card choices; absent for non-card decisions. */
  select_zone?: SelectZone;
}

export interface LogMessage {
  type: 'log';
  lines: string[];
}

export interface GameOverMessage {
  type: 'gameover';
  scores: number[];
  winner: number | null;
  truncated: boolean;
}

export interface ErrorMessage {
  type: 'error';
  message: string;
}

export type ServerMessage =
  | TableMessage
  | StateMessage
  | DecisionMessage
  | LogMessage
  | GameOverMessage
  | ErrorMessage;

export interface ActMessage {
  type: 'act';
  action: number;
}

export interface UndoRequestMessage {
  type: 'undo_request';
}

export type ClientMessage = ActMessage | UndoRequestMessage;

export interface CreateSessionRequest {
  seats: SeatKind[];
  kingdom?: Array<string | number>;
  seed?: number;
  thinking_delay_ms?: number;
}

export interface CreateSessionResponse {
  session_id: string;
  seat_tokens: string[];
}
