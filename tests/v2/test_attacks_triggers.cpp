#include "v2/core/game.h"
#include "v2/core/triggers.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

namespace {

[[nodiscard]] Setup attack_setup(PlayerId players = 2U) {
    Setup setup{};
    setup.num_players = players;
    setup.kingdom_count = 5;
    setup.kingdom[0] = DEF_MERCHANT;
    setup.kingdom[1] = DEF_MILITIA;
    setup.kingdom[2] = DEF_WITCH;
    setup.kingdom[3] = DEF_MOAT;
    setup.kingdom[4] = DEF_BUREAUCRAT;
    setup.kingdom[5] = DEF_ORDER_ALPHA_TEST;
    setup.kingdom[6] = DEF_ORDER_BETA_TEST;
    setup.kingdom[7] = DEF_ORDER_GAMMA_TEST;
    setup.kingdom_count = 8;
    return setup;
}

[[nodiscard]] Pile* find_pile(GameState& state, DefId def) {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        Pile& pile = state.piles[i];
        if (pile.mixed_len == 0U && state.slot_to_def[pile.base] == def) {
            return &pile;
        }
    }
    return nullptr;
}

void clear_player(GameState& state, PlayerId player_id) {
    PlayerState& player = state.players[player_id];
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        player.hand[slot] = 0;
        player.exile[slot] = 0;
        player.tavern[slot] = 0;
        player.island_mat[slot] = 0;
    }
    player.deck.size = 0;
    player.discard.size = 0;
    player.set_aside.size = 0;
    player.in_play_size = 0;
    player.pending_size = 0;
}

void reset_for_action(GameState& state) {
    for (PlayerId player = 0; player < state.num_players; ++player) {
        clear_player(state, player);
    }
    state.phase = static_cast<std::uint8_t>(Phase::Action);
    state.actions = 1;
    state.buys = 1;
    state.coins = 0;
    state.potion_coins = 0;
    state.effect_depth = 0;
    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseAction), 0, 0, 0};
    mark_trigger_table_dirty(state);
}

void add_hand(GameState& state, PlayerId player_id, DefId def, std::uint8_t count = 1U) {
    const Slot slot = slot_of(state, def);
    REQUIRE(slot != NONE);
    state.players[player_id].hand[slot] = static_cast<std::uint8_t>(state.players[player_id].hand[slot] + count);
}

void add_in_play(GameState& state, PlayerId player_id, DefId def) {
    const Slot slot = slot_of(state, def);
    REQUIRE(slot != NONE);
    PlayerState& player = state.players[player_id];
    REQUIRE(player.in_play_size < MAX_IN_PLAY);
    player.in_play[player.in_play_size] = InPlayEntry{slot, slot, 0U};
    ++player.in_play_size;
}

[[nodiscard]] std::uint8_t hand_total(const GameState& state, PlayerId player_id) {
    std::uint8_t total = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        total = static_cast<std::uint8_t>(total + state.players[player_id].hand[slot]);
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
    total += player.deck.size;
    total += player.discard.size;
    total += player.set_aside.size;
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

[[nodiscard]] int total_cards(const GameState& state) {
    int total = 0;
    for (PlayerId player = 0; player < state.num_players; ++player) {
        total += count_player_cards(state.players[player]);
    }
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        total += count_pile_cards(state.piles[i]);
    }
    for (std::uint8_t i = 0; i < state.num_nonsupply; ++i) {
        total += count_pile_cards(state.nonsupply[i]);
    }
    total += count_zone(state.trash);
    total += count_bandit_revealed(state);
    return total;
}

void step_checked(GameState& state, Action action) {
    ActionMask legal{};
    const int count = Game::legal_actions(state, legal);
    REQUIRE(count > 0);
    REQUIRE(action < ACTION_SPACE_SIZE);
    REQUIRE(legal.test(action));
    (void)Game::step(state, action);
}

void require_order_options(const GameState& state, std::uint8_t count) {
    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::OrderTriggers));
    REQUIRE(legal_count == count);
    REQUIRE_FALSE(legal.test(A_PASS));
    for (std::uint8_t option = 0; option < count; ++option) {
        REQUIRE(legal.test(option_action(option)));
    }
    if (count < 16U) {
        REQUIRE_FALSE(legal.test(option_action(count)));
    }
}

[[nodiscard]] GameState order_trigger_state(DefId first, DefId second, DefId third = NONE) {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0BDEULL);
    reset_for_action(state);
    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseBuy), 0, 0, 0};

    add_in_play(state, 0U, first);
    add_in_play(state, 0U, second);
    if (third != NONE) {
        add_in_play(state, 0U, third);
    }
    add_hand(state, 0U, DEF_SILVER);
    mark_trigger_table_dirty(state);
    return state;
}

} // namespace

TEST_CASE("v2 attacks resolve opponents in turn order", "[v2][attack]") {
    GameState state = Game::new_game(attack_setup(4U), 0xA77A'0001ULL);
    reset_for_action(state);

    add_hand(state, 0U, DEF_MILITIA);
    add_hand(state, 1U, DEF_COPPER, 3U);
    add_hand(state, 1U, DEF_ESTATE);
    add_hand(state, 2U, DEF_COPPER, 3U);
    add_hand(state, 2U, DEF_SILVER);
    add_hand(state, 3U, DEF_COPPER, 3U);
    add_hand(state, 3U, DEF_GOLD);

    step_checked(state, play_action(DEF_MILITIA));
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state.decision.player == 1U);

    step_checked(state, select_action(DEF_COPPER));
    REQUIRE(state.decision.player == 1U);
    step_checked(state, select_action(DEF_COPPER));
    REQUIRE(state.decision.player == 1U);
    step_checked(state, select_action(DEF_COPPER));
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state.decision.player == 2U);

    step_checked(state, select_action(DEF_COPPER));
    step_checked(state, select_action(DEF_COPPER));
    step_checked(state, select_action(DEF_COPPER));
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));
    REQUIRE(state.decision.player == 3U);

    step_checked(state, select_action(DEF_GOLD));
    step_checked(state, select_action(DEF_COPPER));
    REQUIRE(state.decision.player == 3U);
    step_checked(state, select_action(DEF_COPPER));
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(hand_total(state, 1U) == 3U);
    REQUIRE(hand_total(state, 2U) == 3U);
    REQUIRE(hand_total(state, 3U) == 3U);
}

TEST_CASE("v2 Moat reveal grants attack immunity", "[v2][attack][reaction]") {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0002ULL);
    reset_for_action(state);

    add_hand(state, 0U, DEF_MILITIA);
    add_hand(state, 1U, DEF_MOAT);
    add_hand(state, 1U, DEF_COPPER, 4U);

    step_checked(state, play_action(DEF_MILITIA));
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::ReactWindow));
    REQUIRE(state.decision.player == 1U);

    step_checked(state, select_action(DEF_MOAT));
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(hand_total(state, 1U) == 5U);
    REQUIRE(state.players[1].discard.size == 0U);
}

TEST_CASE("v2 Militia with small hand auto-completes", "[v2][attack]") {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0003ULL);
    reset_for_action(state);

    add_hand(state, 0U, DEF_MILITIA);
    add_hand(state, 1U, DEF_COPPER, 3U);

    step_checked(state, play_action(DEF_MILITIA));
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::PhaseBuy));
    REQUIRE(hand_total(state, 1U) == 3U);
}

TEST_CASE("v2 Witch skips empty Curse pile", "[v2][attack]") {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0004ULL);
    reset_for_action(state);

    add_hand(state, 0U, DEF_WITCH);
    add_hand(state, 1U, DEF_COPPER, 3U);
    Pile* curses = find_pile(state, DEF_CURSE);
    REQUIRE(curses != nullptr);
    curses->count = 0;

    step_checked(state, play_action(DEF_WITCH));
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(ordered_count(state.players[1].discard, slot_of(state, DEF_CURSE)) == 0U);
}

TEST_CASE("v2 Bureaucrat topdecks Silver and skips opponents with no Victory", "[v2][attack]") {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0005ULL);
    reset_for_action(state);

    add_hand(state, 0U, DEF_BUREAUCRAT);
    add_hand(state, 1U, DEF_COPPER);
    add_hand(state, 1U, DEF_SILVER);

    step_checked(state, play_action(DEF_BUREAUCRAT));
    const Slot silver = slot_of(state, DEF_SILVER);
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state.players[0].deck.size == 1U);
    REQUIRE(state.players[0].deck.cards[0] == silver);
    REQUIRE(hand_total(state, 1U) == 2U);
}

TEST_CASE("v2 Merchant triggers only on first Silver", "[v2][trigger]") {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0006ULL);
    reset_for_action(state);

    add_hand(state, 0U, DEF_MERCHANT);
    add_hand(state, 0U, DEF_SILVER, 2U);

    step_checked(state, play_action(DEF_MERCHANT));
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));

    step_checked(state, play_action(DEF_SILVER));
    REQUIRE(state.coins == 3);

    step_checked(state, play_action(DEF_SILVER));
    REQUIRE(state.coins == 5);
}

TEST_CASE("v2 same-window triggers create an order window", "[v2][trigger]") {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0007ULL);
    reset_for_action(state);
    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.decision = PendingDecision{0, static_cast<std::uint8_t>(DecisionKind::PhaseBuy), 0, 0, 0};

    const Slot merchant = slot_of(state, DEF_MERCHANT);
    state.players[0].in_play[0] = InPlayEntry{merchant, merchant, 0U};
    state.players[0].in_play[1] = InPlayEntry{merchant, merchant, 0U};
    state.players[0].in_play_size = 2U;
    add_hand(state, 0U, DEF_SILVER);
    mark_trigger_table_dirty(state);

    step_checked(state, play_action(DEF_SILVER));
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::OrderTriggers));
    REQUIRE(state.decision.player == 0U);
    REQUIRE(state.decision.max_left == 2U);
    REQUIRE(state.coins == 2);

    step_checked(state, option_action(0U));
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state.coins == 4);
}

TEST_CASE("v2 trigger order option changes resolution for two different triggers", "[v2][trigger]") {
    GameState beta_first = order_trigger_state(DEF_ORDER_ALPHA_TEST, DEF_ORDER_BETA_TEST);
    step_checked(beta_first, play_action(DEF_SILVER));
    require_order_options(beta_first, 2U);
    step_checked(beta_first, option_action(0U));
    REQUIRE(beta_first.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(beta_first.coins == 5);

    GameState alpha_first = order_trigger_state(DEF_ORDER_ALPHA_TEST, DEF_ORDER_BETA_TEST);
    step_checked(alpha_first, play_action(DEF_SILVER));
    require_order_options(alpha_first, 2U);
    step_checked(alpha_first, option_action(1U));
    REQUIRE(alpha_first.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(alpha_first.coins == 14);
}

TEST_CASE("v2 trigger order preserves a three-trigger permutation", "[v2][trigger]") {
    GameState state = order_trigger_state(
        DEF_ORDER_ALPHA_TEST,
        DEF_ORDER_BETA_TEST,
        DEF_ORDER_GAMMA_TEST);

    step_checked(state, play_action(DEF_SILVER));
    require_order_options(state, 3U);

    step_checked(state, option_action(2U));
    require_order_options(state, 2U);
    REQUIRE(state.coins == 4);

    step_checked(state, option_action(1U));
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Buy));
    REQUIRE(state.coins == 114);
}

TEST_CASE("v2 clone-equivalence holds at a trigger-order window", "[v2][trigger][determinism]") {
    GameState state = order_trigger_state(
        DEF_ORDER_ALPHA_TEST,
        DEF_ORDER_BETA_TEST,
        DEF_ORDER_GAMMA_TEST);
    step_checked(state, play_action(DEF_SILVER));
    require_order_options(state, 3U);

    GameState clone = state;
    step_checked(state, option_action(2U));
    step_checked(clone, option_action(2U));
    REQUIRE(std::memcmp(&state, &clone, sizeof(GameState)) == 0);

    step_checked(state, option_action(1U));
    step_checked(clone, option_action(1U));
    REQUIRE(std::memcmp(&state, &clone, sizeof(GameState)) == 0);
}

TEST_CASE("v2 clone-equivalence holds at a mid-attack decision", "[v2][attack][determinism]") {
    GameState state = Game::new_game(attack_setup(), 0xA77A'0008ULL);
    reset_for_action(state);

    add_hand(state, 0U, DEF_MILITIA);
    add_hand(state, 1U, DEF_COPPER, 3U);
    add_hand(state, 1U, DEF_ESTATE);

    step_checked(state, play_action(DEF_MILITIA));
    REQUIRE(state.decision.kind == static_cast<std::uint8_t>(DecisionKind::Choose));

    GameState clone = state;
    step_checked(state, select_action(DEF_COPPER));
    step_checked(clone, select_action(DEF_COPPER));
    REQUIRE(std::memcmp(&state, &clone, sizeof(GameState)) == 0);
}

TEST_CASE("v2 conservation holds through random attack games", "[v2][attack][conservation]") {
    for (std::uint64_t seed = 0; seed < 5U; ++seed) {
        GameState state = Game::new_game(attack_setup(), 0xA77A'1000ULL + seed);
        RandomBot bot0{0xA77A'2000ULL + seed};
        RandomBot bot1{0xA77A'3000ULL + seed};
        const int baseline = total_cards(state);

        bool done = false;
        int steps = 0;
        while (!done) {
            REQUIRE(total_cards(state) == baseline);
            ActionMask legal{};
            const int legal_count = Game::legal_actions(state, legal);
            REQUIRE(legal_count > 0);
            const PlayerId player = Game::current_decision(state).player;
            const Action action = player == 0U
                ? bot0.choose_action(state, legal, legal_count)
                : bot1.choose_action(state, legal, legal_count);
            done = Game::step(state, action);
            ++steps;
            REQUIRE(steps < 10000);
        }
        REQUIRE(total_cards(state) == baseline);
    }
}
