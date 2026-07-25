#pragma once

#include "v2/core/rng.h"
#include "v2/core/types.h"

#include <cstdint>
#include <type_traits>

constexpr int MAX_IN_PLAY = 24;
constexpr int MAX_TURN_QUEUE = 16;
constexpr int MAX_TRIGGER_SUBS = 64;

struct OrderedZone {
    std::uint8_t size = 0;
    Slot cards[MAX_DECK_CARDS]{};
};

struct InPlayEntry {
    Slot slot = 0;
    Slot behaves_as = 0;
    std::uint8_t flags = 0;
};

struct EffectFrame {
    std::uint16_t source = 0;
    std::uint8_t pc = 0;
    PlayerId player = 0;
    std::uint8_t flags = 0;
    std::uint8_t repeats_left = 1;
    std::int16_t data[8]{};
};

struct PlayerState {
    std::uint8_t hand[MAX_SLOTS]{};
    std::uint8_t exile[MAX_SLOTS]{};
    std::uint8_t tavern[MAX_SLOTS]{};
    std::uint8_t island_mat[MAX_SLOTS]{};

    OrderedZone deck;
    OrderedZone discard;
    OrderedZone set_aside;

    std::uint8_t in_play_size = 0;
    InPlayEntry in_play[MAX_IN_PLAY]{};

    std::uint8_t pending_size = 0;
    EffectFrame pending[MAX_PENDING]{};

    std::uint8_t coffers = 0;
    std::uint8_t villagers = 0;
    std::uint8_t favors = 0;
    std::uint8_t vp_tokens_lo = 0;
    std::uint8_t vp_tokens_hi = 0;
    std::uint8_t debt = 0;
    std::uint8_t journey_up : 1 = 0;
    std::uint8_t minus_card : 1 = 0;
    std::uint8_t minus_coin : 1 = 0;
    std::uint8_t flags_pad : 5 = 0;
};

struct Pile {
    Slot base = 0;
    std::uint8_t count = 0;
    std::uint8_t mixed_len = 0;
    Slot mixed[12]{};
    std::uint8_t trait = NO_LANDSCAPE;
    std::uint8_t embargo = 0;
    std::uint8_t gain_counter = 0;
    std::uint8_t adv_tokens[MAX_PLAYERS]{};
};

enum class TurnKind : std::uint8_t {
    Normal,
    Outpost,
    Mission,
    Voyage,
    Possession,
    FleetFinal,
};

struct TurnQueueEntry {
    PlayerId player = 0;
    TurnKind turn_kind = TurnKind::Normal;
};

struct TurnQueue {
    std::uint8_t head = 0;
    std::uint8_t size = 0;
    TurnQueueEntry entries[MAX_TURN_QUEUE]{};
};

enum class DecisionKind : std::uint8_t {
    None,
    PhaseAction,
    PhaseBuy,
    PhaseNight,
    Choose,
    ChooseGain,
    ChooseOption,
    ChooseOrder,
    ReactWindow,
    OrderTriggers,
};

enum class SelectSemantic : std::uint8_t {
    None = 0,
    Keep,
    Discard,
    Trash,
    Topdeck,
    Gain,
    Other,
};

struct PendingDecision {
    PlayerId player = 0;
    std::uint8_t kind = 0;
    std::uint16_t source = 0;
    std::uint8_t min_left = 0;
    std::uint8_t max_left = 0;
    std::uint8_t select_semantic = static_cast<std::uint8_t>(SelectSemantic::None);
};

struct Subscription {
    PlayerId owner = 0;
    std::uint16_t source = 0;
    std::uint8_t kind_of_source = 0;
};

struct TriggerTable {
    Subscription subs[MAX_TRIGGER_SUBS]{};
    std::uint8_t count = 0;
    std::uint8_t dirty : 1 = 1;
};

enum class Phase : std::uint8_t {
    Action,
    Buy,
    Night,
    Cleanup,
    Over,
};

struct GameState {
    std::uint8_t num_players = 0;
    DefId slot_to_def[MAX_SLOTS]{};
    std::uint8_t num_slots = 0;

    Pile piles[MAX_PILES]{};
    std::uint8_t num_piles = 0;
    Pile nonsupply[MAX_NONSUPPLY]{};
    std::uint8_t num_nonsupply = 0;
    std::uint8_t trash[MAX_SLOTS]{};
    PlayerState players[MAX_PLAYERS]{};

    std::uint8_t events[MAX_LANDSCAPES]{};
    std::uint8_t ways[MAX_LANDSCAPES]{};
    std::uint8_t landmarks[MAX_LANDSCAPES]{};
    std::uint8_t projects[MAX_LANDSCAPES]{};
    std::uint8_t project_bought[MAX_LANDSCAPES]{};
    std::uint8_t prophecy = NO_LANDSCAPE;
    std::uint8_t sun_tokens = 0;
    std::uint8_t artifact_holder[NUM_ARTIFACTS]{};
    OrderedZone boons;
    OrderedZone boons_discard;
    OrderedZone hexes;
    OrderedZone hexes_discard;

    std::uint8_t phase = 0;
    std::uint8_t actions = 0;
    std::uint8_t buys = 0;
    std::int16_t coins = 0;
    std::uint8_t potion_coins = 0;
    TurnQueue turn_queue;
    std::uint16_t turn_counter = 0;
    std::uint8_t truncated : 1 = 0;

    std::uint8_t effect_depth = 0;
    EffectFrame effect_stack[MAX_EFFECT_DEPTH]{};
    PendingDecision decision;
    TriggerTable trigger_table;

    Xoshiro256pp rng{};
};

static_assert(std::is_trivially_copyable_v<GameState>);
static_assert(sizeof(GameState) <= 16384U);
