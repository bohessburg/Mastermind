#include "spec_harness.h"

#include "v2/core/defs.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"

#include <cstring>

namespace {

[[nodiscard]] Setup full_phase2_setup() {
    Setup setup{};
    constexpr DefId kKingdom[] = {
        DEF_CELLAR,
        DEF_CHAPEL,
        DEF_VILLAGE,
        DEF_SMITHY,
        DEF_WORKSHOP,
        DEF_REMODEL,
        DEF_MINE,
        DEF_MERCHANT,
        DEF_MILITIA,
        DEF_WITCH,
        DEF_MOAT,
        DEF_BUREAUCRAT,
        DEF_MARKET,
        DEF_FESTIVAL,
        DEF_LABORATORY,
        DEF_GARDENS,
        DEF_MONEYLENDER,
        DEF_POACHER,
        DEF_VASSAL,
        DEF_HARBINGER,
        DEF_THRONE_ROOM,
        DEF_COUNCIL_ROOM,
        DEF_ARTISAN,
        DEF_BANDIT,
        DEF_EXACT_TWO_TEST,
        DEF_REPEAT_CHOOSE_TEST,
        DEF_ORDER_ALPHA_TEST,
        DEF_ORDER_BETA_TEST,
        DEF_ORDER_GAMMA_TEST,
    };
    setup.kingdom_count = static_cast<std::uint8_t>(sizeof(kKingdom) / sizeof(kKingdom[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kKingdom[i];
    }
    return setup;
}

[[nodiscard]] int count_ordered(const OrderedZone& zone) {
    return zone.size;
}

[[nodiscard]] int count_zone(const std::uint8_t (&zone)[MAX_SLOTS]) {
    int total = 0;
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        total += zone[slot];
    }
    return total;
}

[[nodiscard]] int count_player_cards(const PlayerState& player) {
    int total = 0;
    total += count_zone(player.hand);
    total += count_zone(player.exile);
    total += count_zone(player.tavern);
    total += count_zone(player.island_mat);
    total += count_ordered(player.deck);
    total += count_ordered(player.discard);
    total += player.in_play_size;
    return total;
}

[[nodiscard]] int count_pile_cards(const Pile& pile) {
    return pile.mixed_len > 0U ? pile.mixed_len : pile.count;
}

[[nodiscard]] int count_bandit_revealed(const GameState& state) {
    int total = 0;
    for (std::uint8_t i = 0; i < state.effect_depth; ++i) {
        const EffectFrame& frame = state.effect_stack[i];
        if (frame.source != DEF_BANDIT) {
            continue;
        }
        const std::uint8_t revealed_count = frame.data[3] < 0 ? 0U : static_cast<std::uint8_t>(frame.data[3]);
        for (std::uint8_t j = 0; j < revealed_count && j < 2U; ++j) {
            if (frame.data[1 + j] >= 0) {
                ++total;
            }
        }
    }
    return total;
}

[[nodiscard]] std::uint8_t ordered_count(const OrderedZone& zone, Slot slot) {
    std::uint8_t total = 0;
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        if (zone.cards[i] == slot) {
            ++total;
        }
    }
    return total;
}

} // namespace

SpecHarness::GivenBuilder::GivenBuilder(SpecHarness& harness)
    : harness_(harness) {}

SpecHarness::ExpectBuilder::ExpectBuilder(SpecHarness& harness)
    : harness_(harness) {}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::hand_has(const char* name) {
    REQUIRE(harness_.hand_count(0U, name) > 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::hand_lacks(const char* name) {
    REQUIRE(harness_.hand_count(0U, name) == 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::discard_has(const char* name) {
    REQUIRE(harness_.discard_count(0U, name) > 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::discard_lacks(const char* name) {
    REQUIRE(harness_.discard_count(0U, name) == 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::deck_top(const char* name) {
    const PlayerState& player = harness_.state_.players[0];
    REQUIRE(player.deck.size > 0U);
    REQUIRE(player.deck.cards[player.deck.size - 1U] == harness_.slot_named(name));
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::trash_has(const char* name) {
    REQUIRE(harness_.trash_count_for(name) > 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::trash_count(const char* name, std::uint8_t count) {
    REQUIRE(harness_.trash_count_for(name) == count);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::coins(std::int16_t coins) {
    REQUIRE(harness_.state_.coins == coins);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::actions(std::uint8_t actions) {
    REQUIRE(harness_.state_.actions == actions);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::buys(std::uint8_t buys) {
    REQUIRE(harness_.state_.buys == buys);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::score(PlayerId player, std::int16_t expected_score) {
    REQUIRE(::score(harness_.state_, player) == expected_score);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::player_hand_has(PlayerId player, const char* name) {
    REQUIRE(harness_.hand_count(player, name) > 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::player_discard_has(PlayerId player, const char* name) {
    REQUIRE(harness_.discard_count(player, name) > 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::player_trash_count(const char* name, std::uint8_t count) {
    REQUIRE(harness_.trash_count_for(name) == count);
    return *this;
}

SpecHarness::SpecHarness()
    : state_(Game::new_game(full_phase2_setup(), 0x5'EC5'0001ULL)),
      baseline_(0),
      given_(*this),
      expect_(*this) {
    refresh_baseline();
}

SpecHarness::GivenBuilder& SpecHarness::given() {
    clear_players();
    state_.phase = static_cast<std::uint8_t>(Phase::Action);
    state_.actions = 1;
    state_.buys = 1;
    state_.coins = 0;
    state_.potion_coins = 0;
    state_.effect_depth = 0;
    state_.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseAction), 0, 0, 0};
    state_.trigger_table.dirty = 1U;
    refresh_baseline();
    return given_;
}

SpecHarness::ExpectBuilder& SpecHarness::expect() {
    return expect_;
}

void SpecHarness::play(const char* name) {
    step_checked(play_action(def_named(name)));
}

void SpecHarness::choose(const char* name) {
    step_checked(select_action(def_named(name)));
}

void SpecHarness::gain(const char* name) {
    choose(name);
}

void SpecHarness::pass() {
    step_checked(A_PASS);
}

void SpecHarness::option(std::uint8_t option) {
    step_checked(option_action(option));
}

void SpecHarness::empty_supply_up_to(std::int8_t coins) {
    const Cost budget{coins, 127, 32767};
    for (std::uint8_t i = 0; i < state_.num_piles; ++i) {
        Pile& pile = state_.piles[i];
        if (pile.mixed_len == 0U && card_def(state_.slot_to_def[pile.base]).cost.fits_within(budget)) {
            pile.count = 0;
        }
    }
    refresh_baseline();
}

void SpecHarness::empty_supply(const char* name) {
    const Slot slot = slot_named(name);
    for (std::uint8_t i = 0; i < state_.num_piles; ++i) {
        Pile& pile = state_.piles[i];
        if (pile.mixed_len == 0U && pile.base == slot) {
            pile.count = 0;
            break;
        }
    }
    refresh_baseline();
}

void SpecHarness::expect_conservation() const {
    REQUIRE(total_cards() == baseline_);
}

GameState& SpecHarness::state() {
    return state_;
}

const GameState& SpecHarness::state() const {
    return state_;
}

DefId SpecHarness::def_named(const char* name) const {
    for (DefId def = 0; def < card_def_count(); ++def) {
        if (std::strcmp(card_def(def).name, name) == 0) {
            return def;
        }
    }
    FAIL("unknown v2 card name");
    return 0;
}

int SpecHarness::total_cards() const {
    int total = 0;
    for (PlayerId player = 0; player < state_.num_players; ++player) {
        total += count_player_cards(state_.players[player]);
    }
    for (std::uint8_t i = 0; i < state_.num_piles; ++i) {
        total += count_pile_cards(state_.piles[i]);
    }
    for (std::uint8_t i = 0; i < state_.num_nonsupply; ++i) {
        total += count_pile_cards(state_.nonsupply[i]);
    }
    total += count_zone(state_.trash);
    total += count_bandit_revealed(state_);
    return total;
}

void SpecHarness::add_hand(const char* name) {
    add_hand(0U, name);
}

void SpecHarness::add_hand(PlayerId player_id, const char* name) {
    REQUIRE(player_id < state_.num_players);
    const Slot slot = ensure_slot_named(name);
    ++state_.players[player_id].hand[slot];
}

void SpecHarness::add_deck(const char* name) {
    add_deck(0U, name);
}

void SpecHarness::add_deck(PlayerId player_id, const char* name) {
    REQUIRE(player_id < state_.num_players);
    const Slot slot = ensure_slot_named(name);
    PlayerState& player = state_.players[player_id];
    REQUIRE(player.deck.size < MAX_DECK_CARDS);
    player.deck.cards[player.deck.size] = slot;
    ++player.deck.size;
}

void SpecHarness::add_discard(const char* name) {
    add_discard(0U, name);
}

void SpecHarness::add_discard(PlayerId player_id, const char* name) {
    REQUIRE(player_id < state_.num_players);
    const Slot slot = ensure_slot_named(name);
    PlayerState& player = state_.players[player_id];
    REQUIRE(player.discard.size < MAX_DECK_CARDS);
    player.discard.cards[player.discard.size] = slot;
    ++player.discard.size;
}

void SpecHarness::clear_player(PlayerId player_id) {
    PlayerState& player = state_.players[player_id];
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        player.hand[slot] = 0;
        player.exile[slot] = 0;
        player.tavern[slot] = 0;
        player.island_mat[slot] = 0;
    }
    player.deck.size = 0;
    player.discard.size = 0;
    player.in_play_size = 0;
    player.pending_size = 0;
}

void SpecHarness::clear_players() {
    for (PlayerId player = 0; player < state_.num_players; ++player) {
        clear_player(player);
    }
}

void SpecHarness::refresh_baseline() {
    baseline_ = total_cards();
}

void SpecHarness::step_checked(Action action) {
    ActionMask legal{};
    (void)Game::legal_actions(state_, legal);
    REQUIRE(action < ACTION_SPACE_SIZE);
    REQUIRE(legal.test(action));
    (void)Game::step(state_, action);
}

Slot SpecHarness::ensure_slot_named(const char* name) {
    const DefId def = def_named(name);
    Slot slot = slot_of(state_, def);
    if (slot != NONE) {
        return slot;
    }
    REQUIRE(state_.num_slots < MAX_SLOTS);
    slot = state_.num_slots;
    state_.slot_to_def[slot] = def;
    ++state_.num_slots;
    return slot;
}

Slot SpecHarness::slot_named(const char* name) const {
    const Slot slot = slot_of(state_, def_named(name));
    REQUIRE(slot != NONE);
    return slot;
}

std::uint8_t SpecHarness::hand_count(PlayerId player, const char* name) const {
    REQUIRE(player < state_.num_players);
    return state_.players[player].hand[slot_named(name)];
}

std::uint8_t SpecHarness::discard_count(PlayerId player, const char* name) const {
    REQUIRE(player < state_.num_players);
    return ordered_count(state_.players[player].discard, slot_named(name));
}

std::uint8_t SpecHarness::trash_count_for(const char* name) const {
    return state_.trash[slot_named(name)];
}
