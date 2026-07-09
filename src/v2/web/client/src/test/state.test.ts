import { describe, expect, it } from 'vitest';
import type { CardDef, DecisionMessage, LogMessage, StateMessage, TableMessage } from '../protocol';
import {
  canSendDone,
  findNextBasicTreasurePlay,
  formatTrashEntries,
  indexDecisionOptionsByDef,
  indexSelectOptionsByDef,
  initialClientState,
  reduceServerMessage,
} from '../state';

describe('client state reducer', () => {
  it('stores table and state redraw messages', () => {
    const table: TableMessage = {
      type: 'table',
      seats: [{ index: 0, kind: 'human' }],
      kingdom: [10, 11],
      landscapes: [],
    };
    const state: StateMessage = {
      type: 'state',
      view: {
        piles: [{ def: 0, count: 46 }],
        myHand: [{ def: 0, count: 3 }],
        myPlayArea: [],
        myDeckCount: 5,
        myDiscardCount: 0,
        myDiscardTop: null,
        opponents: [],
        trash: [],
        trashTop: null,
        resources: {
          actions: 1,
          buys: 1,
          coins: 0,
          potion: 0,
          debt: 0,
          coffers: 0,
          villagers: 0,
          favors: 0,
          vp_tokens: 0,
        },
        phase: 0,
        turn: 0,
      },
    };

    const withTable = reduceServerMessage(initialClientState, table);
    const withState = reduceServerMessage(withTable, state);
    expect(withState.table?.kingdom).toEqual([10, 11]);
    expect(withState.state?.view.myHand[0].count).toBe(3);
  });

  it('appends log lines', () => {
    const first: LogMessage = { type: 'log', lines: ['P1 plays Copper'] };
    const second: LogMessage = { type: 'log', lines: ['P1 buys Silver'] };
    const state = reduceServerMessage(reduceServerMessage(initialClientState, first), second);
    expect(state.log).toEqual(['P1 plays Copper', 'P1 buys Silver']);
  });

  it('recognizes pass or done options for multi-select flow', () => {
    const decision: DecisionMessage = {
      type: 'decision',
      seat: 0,
      kind: 'Choose',
      source: { def: 10, name: 'Cellar' },
      prompt: 'Cellar: discard any number of cards',
      min: 0,
      max: 10,
      options: [
        { action: 0, label: 'Done' },
        { action: 400, label: 'Discard Estate', def: 5 },
      ],
    };
    expect(canSendDone(decision)).toBe(true);
    expect(canSendDone(undefined)).toBe(false);
  });

  it('indexes play and buy options by def for tile clicks', () => {
    const decision: DecisionMessage = {
      type: 'decision',
      seat: 0,
      kind: 'PhaseBuy',
      source: { def: 0, name: 'Copper' },
      prompt: 'Buy phase',
      min: 0,
      max: 0,
      options: [
        { action: 100, label: 'Play Copper', def: 0 },
        { action: 200, label: 'Buy Silver', def: 1 },
      ],
    };

    expect(indexDecisionOptionsByDef(decision, 'Play').get(0)?.action).toBe(100);
    expect(indexDecisionOptionsByDef(decision, 'Buy').get(1)?.action).toBe(200);
    expect(indexDecisionOptionsByDef(decision, 'Play').has(1)).toBe(false);
  });

  it('indexes A_SELECT card actions only in the decision zone', () => {
    const chooseFromHand: DecisionMessage = {
      type: 'decision',
      seat: 0,
      kind: 'Choose',
      source: { def: 10, name: 'Cellar' },
      prompt: 'Cellar: discard any number of cards',
      min: 0,
      max: 5,
      select_zone: 'hand',
      options: [
        { action: 400, label: 'Discard Copper', def: 0 },
        { action: 0, label: 'Done' },
      ],
    };
    const gainFromSupply: DecisionMessage = {
      ...chooseFromHand,
      kind: 'ChooseGain',
      source: { def: 14, name: 'Workshop' },
      prompt: 'Workshop: gain a card costing up to 4',
      select_zone: 'supply',
      options: [{ action: 400, label: 'Gain Copper', def: 0 }],
    };

    // Copper can be visible in hand and Supply at the same time. Only the
    // decision's actual source zone may receive the click affordance.
    expect(indexSelectOptionsByDef(chooseFromHand, 'hand').get(0)?.action).toBe(400);
    expect(indexSelectOptionsByDef(chooseFromHand, 'supply').has(0)).toBe(false);
    expect(indexSelectOptionsByDef(gainFromSupply, 'supply').get(0)?.action).toBe(400);
    expect(indexSelectOptionsByDef(gainFromSupply, 'hand').has(0)).toBe(false);
  });

  it('supports discard and set-aside select zones for their visible card tiles', () => {
    const harbinger: DecisionMessage = {
      type: 'decision',
      seat: 0,
      kind: 'Choose',
      source: { def: 34, name: 'Harbinger' },
      prompt: 'Harbinger: put a discard card onto your deck',
      min: 0,
      max: 1,
      options: [{ action: 401, label: 'Topdeck Silver', def: 1 }],
    };
    const bandit: DecisionMessage = {
      ...harbinger,
      source: { def: 38, name: 'Bandit' },
      select_zone: 'set_aside',
      options: [{ action: 402, label: 'Trash Gold', def: 2 }],
    };

    // Harbinger also exercises the older-message fallback when select_zone is
    // absent; Bandit receives the explicit server hint.
    expect(indexSelectOptionsByDef(harbinger, 'discard').get(1)?.action).toBe(401);
    expect(indexSelectOptionsByDef(harbinger, 'hand').has(1)).toBe(false);
    expect(indexSelectOptionsByDef(bandit, 'set_aside').get(2)?.action).toBe(402);
  });

  it('selects only legal basic treasure plays and stops when none remain or seat is inactive', () => {
    const defs = new Map<number, Pick<CardDef, 'is_basic_treasure'>>([
      [0, { is_basic_treasure: true }],
      [1, { is_basic_treasure: true }],
      [2, { is_basic_treasure: true }],
      [16, { is_basic_treasure: false }],
    ]);
    const buyDecision: DecisionMessage = {
      type: 'decision',
      seat: 0,
      kind: 'PhaseBuy',
      source: { def: 0, name: 'Copper' },
      prompt: 'Buy phase',
      min: 0,
      max: 0,
      options: [
        { action: 100, label: 'Play Copper', def: 0 },
        { action: 102, label: 'Play Gold', def: 2 },
        { action: 116, label: 'Play Mine', def: 16 },
        { action: 201, label: 'Buy Silver', def: 1 },
      ],
    };

    expect(findNextBasicTreasurePlay(buyDecision, defs)?.action).toBe(102);
    expect(
      findNextBasicTreasurePlay({ ...buyDecision, options: [{ action: 116, label: 'Play Mine', def: 16 }] }, defs),
    ).toBeUndefined();
    expect(findNextBasicTreasurePlay({ ...buyDecision, kind: 'PhaseAction' }, defs)).toBeUndefined();
    expect(findNextBasicTreasurePlay({ ...buyDecision, options: [] }, defs)).toBeUndefined();
  });

  it('formats the full trash multiset for the trash popover', () => {
    const names = new Map<number, string>([
      [0, 'Copper'],
      [1, 'Silver'],
    ]);
    expect(formatTrashEntries(
      [
        { def: 0, count: 3 },
        { def: 1, count: 1 },
      ],
      (def) => names.get(def),
    )).toEqual(['Copper x3', 'Silver x1']);
    expect(formatTrashEntries([], (def) => names.get(def))).toEqual(['Trash is empty']);
  });
});
