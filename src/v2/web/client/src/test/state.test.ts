import { describe, expect, it } from 'vitest';
import type { DecisionMessage, LogMessage, StateMessage, TableMessage } from '../protocol';
import { canSendDone, initialClientState, reduceServerMessage } from '../state';

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
});
