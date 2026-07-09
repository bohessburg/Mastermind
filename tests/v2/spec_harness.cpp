#include "spec_harness.h"

#include "v2/core/defs.h"
#include "v2/core/turns.h"

#include <cstring>

namespace {

[[nodiscard]] Setup full_phase2_setup() {
    Setup setup{};
    setup.kingdom_count = 7;
    setup.kingdom[0] = DEF_CELLAR;
    setup.kingdom[1] = DEF_CHAPEL;
    setup.kingdom[2] = DEF_VILLAGE;
    setup.kingdom[3] = DEF_SMITHY;
    setup.kingdom[4] = DEF_WORKSHOP;
    setup.kingdom[5] = DEF_REMODEL;
    setup.kingdom[6] = DEF_MINE;
    setup.kingdom[7] = DEF_EXACT_TWO_TEST;
    setup.kingdom[8] = DEF_REPEAT_CHOOSE_TEST;
    setup.kingdom_count = 9;
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
    REQUIRE(harness_.hand_count(name) > 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::hand_lacks(const char* name) {
    REQUIRE(harness_.hand_count(name) == 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::discard_has(const char* name) {
    REQUIRE(harness_.discard_count(name) > 0U);
    return *this;
}

SpecHarness::ExpectBuilder& SpecHarness::ExpectBuilder::discard_lacks(const char* name) {
    REQUIRE(harness_.discard_count(name) == 0U);
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

SpecHarness::SpecHarness()
    : state_(Game::new_game(full_phase2_setup(), 0x5'EC5'0001ULL)),
      baseline_(0),
      given_(*this),
      expect_(*this) {
    refresh_baseline();
}

SpecHarness::GivenBuilder& SpecHarness::given() {
    clear_player_zero();
    state_.phase = static_cast<std::uint8_t>(Phase::Action);
    state_.actions = 1;
    state_.buys = 1;
    state_.coins = 0;
    state_.potion_coins = 0;
    state_.effect_depth = 0;
    state_.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseAction), 0, 0, 0};
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
    return total;
}

void SpecHarness::add_hand(const char* name) {
    const Slot slot = ensure_slot_named(name);
    ++state_.players[0].hand[slot];
}

void SpecHarness::add_deck(const char* name) {
    const Slot slot = ensure_slot_named(name);
    PlayerState& player = state_.players[0];
    REQUIRE(player.deck.size < MAX_DECK_CARDS);
    player.deck.cards[player.deck.size] = slot;
    ++player.deck.size;
}

void SpecHarness::clear_player_zero() {
    PlayerState& player = state_.players[0];
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

std::uint8_t SpecHarness::hand_count(const char* name) const {
    return state_.players[0].hand[slot_named(name)];
}

std::uint8_t SpecHarness::discard_count(const char* name) const {
    return ordered_count(state_.players[0].discard, slot_named(name));
}

std::uint8_t SpecHarness::trash_count_for(const char* name) const {
    return state_.trash[slot_named(name)];
}
