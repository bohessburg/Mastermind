import { FormEvent, KeyboardEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode, RefObject } from 'react';
import { createPortal } from 'react-dom';
import { GameSocket } from './connection';
import { cardDef, defTable, defsById, kingdomPreset } from './defs';
import type {
  CountedCard,
  CreateSessionResponse,
  DecisionMessage,
  DecisionOption,
  OpponentView,
  ResourceView,
  SeatKind,
  ServerMessage,
} from './protocol';
import {
  ClientState,
  findNextBasicTreasurePlay,
  formatTrashEntries,
  indexDecisionOptionsByDef,
  indexSelectOptionsByDef,
  initialClientState,
  orderSupplyPiles,
  reduceServerMessage,
} from './state';

interface Credentials {
  sessionId: string;
  seatToken: string;
}

interface CreatedSession {
  sessionId: string;
  seatTokens: string[];
}

const TYPE_PRIORITY = ['Attack', 'Reaction', 'Action', 'Treasure', 'Victory', 'Curse'];

function typeClass(types: string[]): string {
  const primary = TYPE_PRIORITY.find((type) => types.includes(type)) ?? 'Plain';
  return `type-${primary.toLowerCase()}`;
}

function CostBadge({ cost }: { cost: { coins: number; potion: number; debt: number } }) {
  return (
    <div className="cost-badges" aria-label="cost">
      <span className="coin-cost">{cost.coins}</span>
      {cost.potion > 0 && <span className="potion-cost">{cost.potion}P</span>}
      {cost.debt > 0 && <span className="debt-cost">{cost.debt}D</span>}
    </div>
  );
}

const POPOVER_GAP = 6;

interface PopoverPosition {
  left: number;
  top: number;
}

function CardPopover({
  open,
  targetRef,
  children,
}: {
  open: boolean;
  targetRef: RefObject<HTMLElement | null>;
  children: ReactNode;
}) {
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const [position, setPosition] = useState<PopoverPosition | undefined>();

  const updatePosition = useCallback(() => {
    const target = targetRef.current;
    const popover = popoverRef.current;
    if (!target || !popover) {
      return;
    }

    const tileRect = target.getBoundingClientRect();
    const popoverRect = popover.getBoundingClientRect();
    const left = Math.max(0, Math.min(tileRect.left, window.innerWidth - popoverRect.width));
    const top =
      tileRect.bottom + POPOVER_GAP + popoverRect.height <= window.innerHeight
        ? tileRect.bottom + POPOVER_GAP
        : Math.max(0, tileRect.top - POPOVER_GAP - popoverRect.height);
    setPosition({ left, top });
  }, [targetRef]);

  useLayoutEffect(() => {
    if (!open) {
      setPosition(undefined);
      return undefined;
    }

    updatePosition();
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
    };
  }, [open, updatePosition]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div
      className="card-popover"
      ref={popoverRef}
      role="tooltip"
      style={
        position
          ? { left: position.left, top: position.top }
          : { left: 0, top: 0, visibility: 'hidden' }
      }
    >
      {children}
    </div>,
    document.body,
  );
}

function CardTile({
  def,
  count,
  compact = false,
  empty = false,
  clickable = false,
  onActivate,
}: {
  def: number | null | undefined;
  count?: number;
  compact?: boolean;
  empty?: boolean;
  clickable?: boolean;
  onActivate?: () => void;
}) {
  const card = cardDef(def);
  const tileRef = useRef<HTMLDivElement | null>(null);
  const [popoverOpen, setPopoverOpen] = useState(false);
  if (!card) {
    return <div className={`card-tile unknown ${compact ? 'compact' : ''}`}>Empty</div>;
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!clickable || !onActivate) {
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onActivate();
    }
  }

  return (
    <>
      <div
        className={`card-tile ${typeClass(card.types)} ${compact ? 'compact' : ''} ${empty ? 'empty' : ''} ${clickable ? 'clickable' : ''}`}
        ref={tileRef}
        tabIndex={0}
        role={clickable ? 'button' : undefined}
        aria-label={clickable ? `Act with ${card.name}` : undefined}
        onClick={clickable ? onActivate : undefined}
        onKeyDown={onKeyDown}
        onMouseEnter={() => setPopoverOpen(true)}
        onMouseLeave={() => setPopoverOpen(false)}
        onFocus={() => setPopoverOpen(true)}
        onBlur={() => setPopoverOpen(false)}
      >
        <div className="card-title-row">
          {!compact && <CostBadge cost={card.cost} />}
          <strong>{card.name}</strong>
        </div>
        {!compact && <div className="card-types">{card.types.join(' - ')}</div>}
        {count !== undefined && <div className="card-count">×{count}</div>}
      </div>
      <CardPopover open={popoverOpen} targetRef={tileRef}>
        <div className="popover-title">{card.name}</div>
        <CostBadge cost={card.cost} />
        <div className="popover-types">{card.types.join(' - ') || 'Card'}</div>
        <p>{card.text}</p>
      </CardPopover>
    </>
  );
}

function ResourceBar({ resources }: { resources: ResourceView }) {
  const extras = (
    [
      ['Potion', resources.potion],
      ['Debt', resources.debt],
      ['Coffers', resources.coffers],
      ['Villagers', resources.villagers],
      ['Favors', resources.favors],
      ['VP', resources.vp_tokens],
    ] satisfies Array<[string, number]>
  ).filter(([, value]) => value !== 0);

  return (
    <div className="resource-bar">
      <span>
        Actions <b className="resource-value">{resources.actions}</b>
      </span>
      <span>
        Buys <b className="resource-value">{resources.buys}</b>
      </span>
      <span>
        Coins <b className="resource-value resource-coins">{resources.coins}</b>
      </span>
      {extras.map(([label, value]) => (
        <span key={label}>
          {label} <b className="resource-value">{value}</b>
        </span>
      ))}
    </div>
  );
}

function SupplyGrid({
  piles,
  actionsByDef,
  onAction,
}: {
  piles: CountedCard[];
  actionsByDef: Map<number, DecisionOption>;
  onAction: (option: DecisionOption) => void;
}) {
  return (
    <section className="supply-grid" aria-label="Supply">
      {piles.map((pile) => (
        <CardTile
          key={pile.def}
          def={pile.def}
          count={pile.count}
          empty={pile.count === 0}
          clickable={actionsByDef.has(pile.def)}
          onActivate={() => {
            const option = actionsByDef.get(pile.def);
            if (option) {
              onAction(option);
            }
          }}
        />
      ))}
    </section>
  );
}

function OpponentStrip({ opponents }: { opponents: OpponentView[] }) {
  return (
    <section className="opponent-strip" aria-label="Opponents">
      {opponents.map((opponent) => (
        <div className="opponent-panel" key={opponent.seat}>
          <div className="panel-heading">P{opponent.seat + 1}</div>
          <div className="public-counts">
            <span>Hand {opponent.handCount}</span>
            <span>Deck {opponent.deckCount}</span>
            <span>Discard {opponent.discardCount}</span>
          </div>
          <div className="mini-row">
            <span>Discard</span>
            <CardTile def={opponent.discardTop} compact />
          </div>
          <div className="mini-row">
            <span>In play</span>
            <div className="tile-row compact-row">
              {opponent.inPlay.length === 0 && <span className="muted">None</span>}
              {opponent.inPlay.map((def, index) => (
                <CardTile key={`${def}-${index}`} def={def} compact />
              ))}
            </div>
          </div>
          {opponent.resources && <ResourceBar resources={opponent.resources} />}
        </div>
      ))}
    </section>
  );
}

function TrashTile({ trash, trashTop }: { trash: CountedCard[]; trashTop: number | null }) {
  const topCard = cardDef(trashTop);
  const total = trash.reduce((sum, card) => sum + card.count, 0);
  const entries = formatTrashEntries(trash, (def) => cardDef(def)?.name);
  const tileClass = topCard ? typeClass(topCard.types) : 'unknown';
  const tileRef = useRef<HTMLDivElement | null>(null);
  const [popoverOpen, setPopoverOpen] = useState(false);

  return (
    <>
      <div
        className={`card-tile compact trash-tile ${tileClass}`}
        ref={tileRef}
        tabIndex={0}
        onMouseEnter={() => setPopoverOpen(true)}
        onMouseLeave={() => setPopoverOpen(false)}
        onFocus={() => setPopoverOpen(true)}
        onBlur={() => setPopoverOpen(false)}
      >
        <div className="card-title-row">
          <strong>{topCard ? topCard.name : 'Empty'}</strong>
        </div>
        {total > 0 && <div className="card-count">×{total}</div>}
      </div>
      <CardPopover open={popoverOpen} targetRef={tileRef}>
        <div className="popover-title">Trash</div>
        <div className="trash-popover-list">
          {entries.map((entry) => (
            <span key={entry}>{entry}</span>
          ))}
        </div>
      </CardPopover>
    </>
  );
}

function TrashAndResources({
  trash,
  trashTop,
  resources,
  turn,
  decision,
}: {
  trash: CountedCard[];
  trashTop: number | null;
  resources: ResourceView;
  turn: number;
  decision?: DecisionMessage;
}) {
  const turnLabel = decision ? `Turn ${turn} — P${decision.seat + 1}'s turn` : `Turn ${turn}`;

  return (
    <section className="table-middle">
      <div className="trash-panel">
        <span>Trash top</span>
        <TrashTile trash={trash} trashTop={trashTop} />
      </div>
      <div className="turn-panel">
        <span className="turn-label">{turnLabel}</span>
        <ResourceBar resources={resources} />
      </div>
    </section>
  );
}

function CountedCardRow({
  cards,
  actionsByDef,
  onAction,
}: {
  cards: CountedCard[];
  actionsByDef: Map<number, DecisionOption>;
  onAction: (option: DecisionOption) => void;
}) {
  if (cards.length === 0) {
    return <div className="empty-row">No cards</div>;
  }
  return (
    <div className="tile-row hand-row">
      {cards.map((card) => (
        <CardTile
          key={card.def}
          def={card.def}
          count={card.count}
          clickable={actionsByDef.has(card.def)}
          onActivate={() => {
            const option = actionsByDef.get(card.def);
            if (option) {
              onAction(option);
            }
          }}
        />
      ))}
    </div>
  );
}

function DefRow({
  defs,
  actionsByDef,
  onAction,
}: {
  defs: number[];
  actionsByDef?: Map<number, DecisionOption>;
  onAction?: (option: DecisionOption) => void;
}) {
  if (defs.length === 0) {
    return <div className="empty-row">No cards</div>;
  }
  return (
    <div className="tile-row">
      {defs.map((def, index) => (
        <CardTile
          key={`${def}-${index}`}
          def={def}
          compact
          clickable={actionsByDef?.has(def)}
          onActivate={() => {
            const option = actionsByDef?.get(def);
            if (option) {
              onAction?.(option);
            }
          }}
        />
      ))}
    </div>
  );
}

function PlayerArea({
  hand,
  inPlay,
  setAside,
  deckCount,
  discardCount,
  discardTop,
  handActionsByDef,
  discardActionsByDef,
  setAsideActionsByDef,
  onAction,
}: {
  hand: CountedCard[];
  inPlay: number[];
  setAside: number[];
  deckCount: number;
  discardCount: number;
  discardTop: number | null;
  handActionsByDef: Map<number, DecisionOption>;
  discardActionsByDef: Map<number, DecisionOption>;
  setAsideActionsByDef: Map<number, DecisionOption>;
  onAction: (option: DecisionOption) => void;
}) {
  return (
    <section className="player-area">
      <div className="zone-block in-play-block">
        <div className="zone-title">In Play</div>
        <DefRow defs={inPlay} />
      </div>
      <div className="zone-block hand-block">
        <div className="zone-title">Hand</div>
        <CountedCardRow cards={hand} actionsByDef={handActionsByDef} onAction={onAction} />
      </div>
      <details className="mats-tray" open={setAside.length > 0}>
        <summary>Mats and private counts</summary>
        <div className="mats-grid">
          <div>Deck {deckCount}</div>
          <div>Discard {discardCount}</div>
          <div className="mini-row">
            <span>Discard top</span>
            <CardTile
              def={discardTop}
              compact
              clickable={discardTop !== null && discardActionsByDef.has(discardTop)}
              onActivate={() => {
                if (discardTop === null) {
                  return;
                }
                const option = discardActionsByDef.get(discardTop);
                if (option) {
                  onAction(option);
                }
              }}
            />
          </div>
          <div>
            <span>Set aside</span>
            <DefRow defs={setAside} actionsByDef={setAsideActionsByDef} onAction={onAction} />
          </div>
        </div>
      </details>
    </section>
  );
}

function DecisionPanel({
  decision,
  picked,
  onAct,
  canPlayAllTreasures,
  autoPlayingTreasures,
  onPlayAllTreasures,
}: {
  decision?: DecisionMessage;
  picked: string[];
  onAct: (option: DecisionOption) => void;
  canPlayAllTreasures: boolean;
  autoPlayingTreasures: boolean;
  onPlayAllTreasures: () => void;
}) {
  if (!decision) {
    return (
      <section className="decision-panel waiting">
        <h2>Decision</h2>
        <p className="muted">Connecting…</p>
      </section>
    );
  }

  const active = decision.options.length > 0;
  const tileOptions = decision.options.filter((option) => option.def !== undefined);
  const controlOptions = decision.options.filter((option) => option.def === undefined);
  const pickGuidance = (() => {
    if (decision.max <= 1) {
      return undefined;
    }
    if (decision.min === decision.max) {
      return `Pick exactly ${decision.max}`;
    }
    if (decision.min === 0) {
      return `Pick up to ${decision.max}`;
    }
    return `Pick ${decision.min}–${decision.max}`;
  })();

  return (
    <section className={`decision-panel ${active ? 'active' : 'waiting'}`}>
      <h2>Decision</h2>
      {!active && (
        <div className="decision-meta">
          <span>P{decision.seat + 1}</span>
        </div>
      )}
      <div className="decision-prompt-row">
        <p className="decision-prompt">{active ? decision.prompt : 'Waiting for opponent…'}</p>
        {active && pickGuidance && <span className="selection-guidance">{pickGuidance}</span>}
      </div>
      {picked.length > 0 && (
        <div className="picked-list">
          <span>Picked</span>
          {picked.map((label, index) => (
            <em key={`${label}-${index}`}>{label}</em>
          ))}
        </div>
      )}
      {decision.kind === 'PhaseBuy' && (
        <button
          className="primary-option treasure-sequence"
          type="button"
          disabled={!canPlayAllTreasures || autoPlayingTreasures}
          onClick={onPlayAllTreasures}
        >
          {autoPlayingTreasures ? 'Playing treasures...' : 'Play all treasures'}
        </button>
      )}
      <div className="option-stack">
        {tileOptions.length > 0 && (
          <div className="tile-option-chips" aria-label="Card options">
            {tileOptions.map((option) => (
              <button
                className="tile-option-chip"
                key={option.action}
                type="button"
                onClick={() => onAct(option)}
              >
                {option.label}
              </button>
            ))}
          </div>
        )}
        {controlOptions.map((option) => (
          <button
            className={option.label === 'Done' ? 'primary-option' : ''}
            key={option.action}
            type="button"
            onClick={() => onAct(option)}
          >
            {option.label}
          </button>
        ))}
        {!active && <button disabled>Waiting for opponent…</button>}
      </div>
    </section>
  );
}

function LogRail({ lines }: { lines: string[] }) {
  const linesRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const element = linesRef.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }, [lines]);

  return (
    <section className="log-rail">
      <h2>Log</h2>
      <div className="log-lines" ref={linesRef}>
        {lines.map((line, index) => {
          const seatMatch = /^(P\d+)(\b.*)$/.exec(line);
          return (
            <div key={`${line}-${index}`}>
              {seatMatch ? (
                <>
                  <span className="log-seat">{seatMatch[1]}</span>
                  {seatMatch[2]}
                </>
              ) : (
                line
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function GameOverOverlay({ gameover, onPlayAgain }: { gameover?: ClientState['gameover']; onPlayAgain: () => void }) {
  if (!gameover) {
    return null;
  }
  return (
    <div className="gameover-overlay">
      <div className="gameover-dialog">
        <h2>Game Over</h2>
        <div className="score-list">
          {gameover.scores.map((score, index) => (
            <span key={index}>
              P{index + 1}: {score}
            </span>
          ))}
        </div>
        <p>{gameover.winner === null ? 'Tie game' : `Winner: P${gameover.winner + 1}`}</p>
        {gameover.truncated && <p>Truncated at turn cap.</p>}
        <button type="button" onClick={onPlayAgain}>Play again</button>
      </div>
    </div>
  );
}

function JoinScreen({ onJoin }: { onJoin: (credentials: Credentials) => void }) {
  const [sessionId, setSessionId] = useState('');
  const [seatToken, setSeatToken] = useState('');
  const [mode, setMode] = useState<'human-bot' | 'human-nn' | 'human-human'>('human-bot');
  const [kingdomMode, setKingdomMode] = useState<'preset' | 'random'>('preset');
  const [seed, setSeed] = useState('2026');
  const [created, setCreated] = useState<CreatedSession | undefined>();
  const [error, setError] = useState<string | undefined>();

  async function createSession() {
    setError(undefined);
    const seats: SeatKind[] =
      mode === 'human-bot' ? ['human', 'bot'] : mode === 'human-nn' ? ['human', 'bot:nn'] : ['human', 'human'];
    const response = await fetch('/api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        seats,
        kingdom: kingdomMode === 'random' ? 'random' : kingdomPreset(),
        seed: seed.trim() === '' ? undefined : Number(seed),
      }),
    });
    if (!response.ok) {
      setError(await response.text());
      return;
    }
    const body = (await response.json()) as CreateSessionResponse;
    const next = { sessionId: body.session_id, seatTokens: body.seat_tokens };
    setCreated(next);
    onJoin({ sessionId: next.sessionId, seatToken: next.seatTokens[0] });
  }

  function joinExisting(event: FormEvent) {
    event.preventDefault();
    onJoin({ sessionId: sessionId.trim(), seatToken: seatToken.trim() });
  }

  return (
    <main className="join-screen">
      <section className="join-panel">
        <h1>DominionZero v2</h1>
        <div className="create-row">
          <label>
            Seat config
            <select value={mode} onChange={(event) => setMode(event.target.value as 'human-bot' | 'human-nn' | 'human-human')}>
              <option value="human-bot">Human vs bot</option>
              <option value="human-nn">Human vs neural net</option>
              <option value="human-human">Human vs human</option>
            </select>
          </label>
          <label>
            Kingdom
            <select
              value={kingdomMode}
              onChange={(event) => setKingdomMode(event.target.value as 'preset' | 'random')}
            >
              <option value="preset">Preset</option>
              <option value="random">Random</option>
            </select>
          </label>
          <label>
            Seed
            <input value={seed} onChange={(event) => setSeed(event.target.value)} />
          </label>
          <button type="button" onClick={createSession}>Create session</button>
        </div>
        {created && (
          <div className="created-session">
            <strong>Session</strong> {created.sessionId}
            <div className="token-list">
              {created.seatTokens.map((token, index) => (
                <code key={token}>P{index + 1}: {token}</code>
              ))}
            </div>
          </div>
        )}
        <form className="join-form" onSubmit={joinExisting}>
          <label>
            Session id
            <input value={sessionId} onChange={(event) => setSessionId(event.target.value)} />
          </label>
          <label>
            Seat token
            <input value={seatToken} onChange={(event) => setSeatToken(event.target.value)} />
          </label>
          <button type="submit">Join</button>
        </form>
        {error && <p className="error-text">{error}</p>}
      </section>
    </main>
  );
}

export function App() {
  const [credentials, setCredentials] = useState<Credentials | undefined>();
  const [clientState, setClientState] = useState<ClientState>(initialClientState);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'open' | 'closed'>('idle');
  const [picked, setPicked] = useState<string[]>([]);
  const [autoPlayingTreasures, setAutoPlayingTreasures] = useState(false);
  const socketRef = useRef<GameSocket | null>(null);
  const lastAutoDecisionRef = useRef<DecisionMessage | undefined>(undefined);

  const decisionKey = useMemo(() => {
    const decision = clientState.decision;
    return decision ? `${decision.kind}:${decision.seat}:${decision.source.def}:${decision.prompt}` : '';
  }, [clientState.decision]);

  useEffect(() => {
    setPicked([]);
  }, [decisionKey]);

  useEffect(() => {
    if (!autoPlayingTreasures) {
      lastAutoDecisionRef.current = undefined;
      return;
    }
    const decision = clientState.decision;
    if (!decision || lastAutoDecisionRef.current === decision) {
      return;
    }
    lastAutoDecisionRef.current = decision;
    const option = findNextBasicTreasurePlay(decision, defsById);
    if (!option) {
      setAutoPlayingTreasures(false);
      return;
    }
    socketRef.current?.send({ type: 'act', action: option.action });
  }, [autoPlayingTreasures, clientState.decision]);

  useEffect(() => {
    if (!credentials) {
      return undefined;
    }
    setClientState(initialClientState);
    const socket = new GameSocket(
      credentials.sessionId,
      credentials.seatToken,
      (message: ServerMessage) => setClientState((state) => reduceServerMessage(state, message)),
      (nextStatus) => setStatus(nextStatus),
    );
    socketRef.current = socket;
    socket.connect();
    return () => socket.close();
  }, [credentials]);

  function act(option: DecisionOption) {
    socketRef.current?.send({ type: 'act', action: option.action });
    setAutoPlayingTreasures(false);
    if (option.label !== 'Done' && option.label !== 'Pass') {
      setPicked((current) => [...current, option.label]);
    }
  }

  function undo() {
    socketRef.current?.send({ type: 'undo_request' });
    setPicked([]);
    setAutoPlayingTreasures(false);
  }

  const playActionsByDef = useMemo(() => {
    const decision = clientState.decision;
    if (decision?.kind !== 'PhaseAction' && decision?.kind !== 'PhaseBuy') {
      return new Map<number, DecisionOption>();
    }
    return indexDecisionOptionsByDef(decision, 'Play');
  }, [clientState.decision]);
  const buyActionsByDef = useMemo(() => {
    const decision = clientState.decision;
    if (decision?.kind !== 'PhaseBuy') {
      return new Map<number, DecisionOption>();
    }
    return indexDecisionOptionsByDef(decision, 'Buy');
  }, [clientState.decision]);
  const selectHandActionsByDef = useMemo(
    () => indexSelectOptionsByDef(clientState.decision, 'hand'),
    [clientState.decision],
  );
  const selectSupplyActionsByDef = useMemo(
    () => indexSelectOptionsByDef(clientState.decision, 'supply'),
    [clientState.decision],
  );
  const selectDiscardActionsByDef = useMemo(
    () => indexSelectOptionsByDef(clientState.decision, 'discard'),
    [clientState.decision],
  );
  const selectSetAsideActionsByDef = useMemo(
    () => indexSelectOptionsByDef(clientState.decision, 'set_aside'),
    [clientState.decision],
  );
  const handActionsByDef = playActionsByDef.size > 0 ? playActionsByDef : selectHandActionsByDef;
  const supplyActionsByDef = buyActionsByDef.size > 0 ? buyActionsByDef : selectSupplyActionsByDef;
  const canPlayAllTreasures = Boolean(findNextBasicTreasurePlay(clientState.decision, defsById));

  if (!credentials) {
    return <JoinScreen onJoin={setCredentials} />;
  }

  const view = clientState.state?.view;
  const connectionStatus =
    status === 'open'
      ? { label: 'Connected', className: 'connected' }
      : status === 'closed'
        ? { label: 'Disconnected', className: 'disconnected' }
        : { label: 'Connecting', className: 'connecting' };

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>DominionZero v2</h1>
          <span className={`connection-status ${connectionStatus.className}`}>
            <i aria-hidden="true" />
            {connectionStatus.label}
          </span>
        </div>
        <div className="header-actions">
          <button type="button" onClick={undo}>Undo</button>
          <button type="button" onClick={() => setCredentials(undefined)}>Leave</button>
        </div>
      </header>
      {clientState.error && <div className="error-banner">{clientState.error}</div>}
      <div className="game-layout">
        <div className="main-table">
          {view && (
            <SupplyGrid
              piles={orderSupplyPiles(view.piles, defsById)}
              actionsByDef={supplyActionsByDef}
              onAction={act}
            />
          )}
          {view && <OpponentStrip opponents={view.opponents} />}
          {view && (
            <TrashAndResources
              trash={view.trash}
              trashTop={view.trashTop}
              resources={view.resources}
              turn={view.turn}
              decision={clientState.decision}
            />
          )}
          {view && (
            <PlayerArea
              hand={view.myHand}
              inPlay={view.myPlayArea}
              setAside={view.mySetAside ?? []}
              deckCount={view.myDeckCount}
              discardCount={view.myDiscardCount}
              discardTop={view.myDiscardTop}
              handActionsByDef={handActionsByDef}
              discardActionsByDef={selectDiscardActionsByDef}
              setAsideActionsByDef={selectSetAsideActionsByDef}
              onAction={act}
            />
          )}
        </div>
        <aside className="right-rail">
          <LogRail lines={clientState.log} />
          <DecisionPanel
            decision={clientState.decision}
            picked={picked}
            onAct={act}
            canPlayAllTreasures={canPlayAllTreasures}
            autoPlayingTreasures={autoPlayingTreasures}
            onPlayAllTreasures={() => setAutoPlayingTreasures(true)}
          />
        </aside>
      </div>
      <GameOverOverlay gameover={clientState.gameover} onPlayAgain={() => setCredentials(undefined)} />
      <footer className="def-count">{defTable.defs.length} defs loaded</footer>
    </main>
  );
}
