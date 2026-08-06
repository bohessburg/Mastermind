#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/game.h"
#include "v2/core/moves.h"
#include "v2/core/setup.h"
#include "v2/core/state_builder.h"
#include "v2/core/turns.h"
#include "v2/drivers/bots.h"
#include "v2/encode/encoder.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

[[nodiscard]] Setup builder_setup() {
    Setup setup{};
    setup.num_players = 2;
    constexpr DefId kingdom[] = {
        DEF_CELLAR,
        DEF_CHAPEL,
        DEF_VILLAGE,
        DEF_SMITHY,
        DEF_WORKSHOP,
        DEF_MILITIA,
        DEF_MOAT,
        DEF_BUREAUCRAT,
        DEF_BANDIT,
        DEF_MARKET,
    };
    setup.kingdom_count =
        static_cast<std::uint8_t>(sizeof(kingdom) / sizeof(kingdom[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kingdom[i];
    }
    return setup;
}

[[nodiscard]] Setup arena_ordered_setup() {
    Setup setup{};
    setup.num_players = 2;
    constexpr DefId kingdom[] = {
        DEF_LABORATORY,
        DEF_MOAT,
        DEF_MERCHANT,
        DEF_WORKSHOP,
        DEF_HARBINGER,
        DEF_WITCH,
        DEF_VILLAGE,
        DEF_ARTISAN,
        DEF_MINE,
        DEF_VASSAL,
    };
    setup.kingdom_count =
        static_cast<std::uint8_t>(sizeof(kingdom) / sizeof(kingdom[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kingdom[i];
    }
    return setup;
}

[[nodiscard]] bool is_canonical_base_pile(DefId def) noexcept {
    switch (def) {
    case DEF_COPPER:
    case DEF_SILVER:
    case DEF_GOLD:
    case DEF_ESTATE:
    case DEF_DUCHY:
    case DEF_PROVINCE:
    case DEF_CURSE:
        return true;
    default:
        return false;
    }
}

void add_count(SnapshotCardCounts& counts, DefId def, std::uint16_t amount = 1U) {
    counts.by_def[def] =
        static_cast<std::uint16_t>(counts.by_def[def] + amount);
}

void add_ordered(
    const GameState& state,
    const OrderedZone& zone,
    SnapshotCardCounts& counts) {
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        add_count(counts, state.slot_to_def[zone.cards[i]]);
    }
}

[[nodiscard]] std::uint16_t hand_size(
    const GameState& state,
    PlayerId player) {
    std::uint16_t total = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        total = static_cast<std::uint16_t>(
            total + state.players[player].hand[slot]);
    }
    return total;
}

[[nodiscard]] std::uint16_t treasure_hand_size(
    const GameState& state,
    PlayerId player) {
    std::uint16_t total = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if ((card_def(state.slot_to_def[slot]).types & TYPE_TREASURE) != 0U) {
            total = static_cast<std::uint16_t>(
                total + state.players[player].hand[slot]);
        }
    }
    return total;
}

[[nodiscard]] Snapshot snapshot_of(
    const GameState& state,
    PlayerId perspective) {
    Snapshot snapshot{};
    snapshot.num_players = state.num_players;
    snapshot.our_player = perspective;
    snapshot.turn_number = state.turn_counter;
    snapshot.current_player = current_player(state);
    const Phase phase = static_cast<Phase>(state.phase);
    snapshot.phase = phase == Phase::Action ? SnapshotPhase::Action
        : phase == Phase::Buy             ? SnapshotPhase::Buy
                                          : SnapshotPhase::Cleanup;

    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        REQUIRE(pile.mixed_len == 0U);
        const DefId def = state.slot_to_def[pile.base];
        snapshot.supply_present[def] = 1U;
        snapshot.supply.by_def[def] = pile.count;
        if (!is_canonical_base_pile(def)) {
            REQUIRE(snapshot.kingdom_order_count < MAX_PILES);
            snapshot.kingdom_order[snapshot.kingdom_order_count] = def;
            ++snapshot.kingdom_order_count;
        }
    }
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        const DefId def = state.slot_to_def[slot];
        snapshot.trash.by_def[def] = state.trash[slot];
    }

    for (PlayerId player = 0; player < state.num_players; ++player) {
        const PlayerState& source = state.players[player];
        SnapshotPlayer& target = snapshot.players[player];
        for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
            const DefId def = state.slot_to_def[slot];
            target.hand.by_def[def] = source.hand[slot];
            target.hand_deck.by_def[def] = source.hand[slot];
        }
        target.hand_count = hand_size(state, player);
        add_ordered(state, source.deck, target.hand_deck);
        target.deck_count = source.deck.size;
        add_ordered(state, source.discard, target.discard);
        add_ordered(state, source.set_aside, target.set_aside);
        for (std::uint8_t i = 0; i < source.in_play_size; ++i) {
            add_count(
                target.in_play,
                state.slot_to_def[source.in_play[i].slot]);
        }
        if (player == snapshot.current_player) {
            target.actions = state.actions;
            target.buys = state.buys;
            target.coins = state.coins;
        }
    }
    for (DefId def = 0; def < card_def_count(); ++def) {
        std::uint32_t total =
            snapshot.supply.by_def[def] + snapshot.trash.by_def[def];
        for (PlayerId player = 0; player < state.num_players; ++player) {
            const SnapshotPlayer& zones = snapshot.players[player];
            total += zones.hand_deck.by_def[def];
            total += zones.discard.by_def[def];
            total += zones.in_play.by_def[def];
            total += zones.set_aside.by_def[def];
        }
        snapshot.card_totals.by_def[def] =
            static_cast<std::uint16_t>(total);
    }
    return snapshot;
}

[[nodiscard]] ActionMask mask_of(const GameState& state) {
    ActionMask mask{};
    REQUIRE(Game::legal_actions(state, mask) > 0);
    return mask;
}

void require_same_mask(const GameState& lhs, const GameState& rhs) {
    const ActionMask lhs_mask = mask_of(lhs);
    const ActionMask rhs_mask = mask_of(rhs);
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        INFO(
            "action=" << action << " lhs_phase=" << static_cast<int>(lhs.phase)
                      << " rhs_phase=" << static_cast<int>(rhs.phase)
                      << " lhs_depth=" << static_cast<int>(lhs.effect_depth)
                      << " rhs_depth=" << static_cast<int>(rhs.effect_depth));
        CHECK(lhs_mask.test(action) == rhs_mask.test(action));
    }
}

[[nodiscard]] Pile& pile_for(GameState& state, DefId def) {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (state.slot_to_def[state.piles[i].base] == def) {
            return state.piles[i];
        }
    }
    FAIL("missing requested pile");
    return state.piles[0];
}

void snapshot_move_supply_to(
    Snapshot& snapshot,
    PlayerId player,
    DefId def,
    SnapshotCardCounts& destination) {
    REQUIRE(snapshot.supply.by_def[def] > 0U);
    --snapshot.supply.by_def[def];
    add_count(destination, def);
    (void)player;
}

void snapshot_gain_to_hand(
    Snapshot& snapshot,
    PlayerId player,
    DefId def) {
    snapshot_move_supply_to(
        snapshot, player, def, snapshot.players[player].hand);
    add_count(snapshot.players[player].hand_deck, def);
    ++snapshot.players[player].hand_count;
}

[[nodiscard]] int mask_count(const ActionMask& mask) {
    int total = 0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        total += mask.test(action) ? 1 : 0;
    }
    return total;
}

} // namespace

TEST_CASE(
    "v2 snapshot rebuild matches organic clean decision states",
    "[v2][state-builder]") {
    int sampled = 0;
    int behavior_checks = 0;
    for (std::uint64_t seed = 0; seed < 8U; ++seed) {
        GameState organic =
            Game::new_game(builder_setup(), 0x57A7'E000ULL + seed);
        RandomBot bots[2] = {
            RandomBot(0xB011'D000ULL + seed),
            RandomBot(0xB011'E000ULL + seed),
        };
        for (int step = 0; step < 100 && organic.phase
             != static_cast<std::uint8_t>(Phase::Over); ++step) {
            if (organic.effect_depth == 0U
                && (organic.phase == static_cast<std::uint8_t>(Phase::Action)
                    || organic.phase == static_cast<std::uint8_t>(Phase::Buy))) {
                const PlayerId perspective = current_player(organic);
                GameState rebuilt =
                    build_game_from_snapshot(snapshot_of(organic, perspective));
                validate_game(rebuilt);
                require_same_mask(organic, rebuilt);
                ++sampled;

                const ActionMask legal = mask_of(organic);
                Action behavior_action = ACTION_SPACE_SIZE;
                if (organic.phase == static_cast<std::uint8_t>(Phase::Action)) {
                    GameState buy_probe = organic;
                    apply_action(buy_probe, A_PASS);
                    const ActionMask buy_legal = mask_of(buy_probe);
                    if (mask_count(buy_legal) > 1) {
                        behavior_action = A_PASS;
                    }
                } else {
                    if (organic.buys > 0U
                        || treasure_hand_size(organic, perspective) > 1U) {
                        for (DefId def = 0; def < card_def_count(); ++def) {
                            const Action play = play_action(def);
                            if (legal.test(play)) {
                                behavior_action = play;
                                break;
                            }
                        }
                    }
                }
                if (behavior_action < ACTION_SPACE_SIZE) {
                    GameState organic_after = organic;
                    GameState rebuilt_after = rebuilt;
                    (void)Game::step(organic_after, behavior_action);
                    (void)Game::step(rebuilt_after, behavior_action);
                    INFO(
                        "seed=" << seed << " step=" << step
                                << " behavior_action=" << behavior_action
                                << " before_phase="
                                << static_cast<int>(organic.phase)
                                << " before_actions="
                                << static_cast<int>(organic.actions)
                                << " before_buys="
                                << static_cast<int>(organic.buys)
                                << " before_coins=" << organic.coins
                                << " hand=" << hand_size(organic, perspective)
                                << " organic_after_phase="
                                << static_cast<int>(organic_after.phase)
                                << " rebuilt_after_phase="
                                << static_cast<int>(rebuilt_after.phase)
                                << " organic_after_buys="
                                << static_cast<int>(organic_after.buys)
                                << " rebuilt_after_buys="
                                << static_cast<int>(rebuilt_after.buys)
                                << " organic_treasures="
                                << treasure_hand_size(
                                       organic_after,
                                       current_player(organic_after))
                                << " rebuilt_treasures="
                                << treasure_hand_size(
                                       rebuilt_after,
                                       current_player(rebuilt_after)));
                    require_same_mask(organic_after, rebuilt_after);
                    ++behavior_checks;
                }
            }

            const ActionMask legal = mask_of(organic);
            const PlayerId player = Game::current_decision(organic).player;
            const Action action =
                bots[player].choose_action(organic, legal, mask_count(legal));
            (void)Game::step(organic, action);
        }
    }
    REQUIRE(sampled >= 100);
    REQUIRE(behavior_checks >= 80);
}

TEST_CASE(
    "v2 snapshot rebuild preserves dealt supply order and turn encoding",
    "[v2][state-builder][encoder]") {
    const GameState organic =
        Game::new_game(arena_ordered_setup(), 0xA11E'0A00ULL);
    const Snapshot snapshot = snapshot_of(organic, 0U);
    const GameState rebuilt = build_game_from_snapshot(snapshot);

    REQUIRE(organic.turn_counter == 0U);
    REQUIRE(snapshot.turn_number == 0U);
    REQUIRE(rebuilt.turn_counter == organic.turn_counter);
    REQUIRE(rebuilt.num_piles == organic.num_piles);
    REQUIRE(snapshot.kingdom_order_count == 10U);
    for (std::uint8_t pile = 0; pile < organic.num_piles; ++pile) {
        CHECK(
            rebuilt.slot_to_def[rebuilt.piles[pile].base]
            == organic.slot_to_def[organic.piles[pile].base]);
    }

    std::array<float, OBS_SIZE_V3> organic_obs{};
    std::array<float, OBS_SIZE_V3> rebuilt_obs{};
    encode(organic, 0U, organic_obs.data(), ObsVersion::V3);
    encode(rebuilt, 0U, rebuilt_obs.data(), ObsVersion::V3);
    REQUIRE(std::memcmp(
                organic_obs.data() + OBS_V2_SUPPLY_OFFSET,
                rebuilt_obs.data() + OBS_V2_SUPPLY_OFFSET,
                OBS_SUPPLY_SIZE * sizeof(float))
        == 0);
    REQUIRE(
        organic_obs[OBS_V2_TURN_OFFSET + OBS_PHASE_COUNT + 2U] == 0.0F);
    REQUIRE(
        rebuilt_obs[OBS_V2_TURN_OFFSET + OBS_PHASE_COUNT + 2U] == 0.0F);
}

TEST_CASE(
    "v2 deck order rigging controls draws and rejects non-permutations",
    "[v2][state-builder]") {
    GameState organic = Game::new_game(builder_setup(), 0xD3C0'0001ULL);
    const Slot smithy = slot_of(organic, DEF_SMITHY);
    REQUIRE(smithy != NONE);
    REQUIRE(do_gain(organic, 0U, smithy, GainDestination::Hand));

    GameState state = build_game_from_snapshot(snapshot_of(organic, 0U));
    std::vector<DefId> order;
    const OrderedZone& deck = state.players[0].deck;
    for (std::uint8_t i = deck.size; i > 0U; --i) {
        order.push_back(state.slot_to_def[deck.cards[i - 1U]]);
    }
    REQUIRE(order.size() >= 3U);
    set_deck_order(state, 0U, std::span<const DefId>(order));

    std::vector<DefId> invalid_order = order;
    invalid_order[0] = invalid_order[0] == DEF_COPPER ? DEF_ESTATE : DEF_COPPER;
    REQUIRE_THROWS_AS(
        set_deck_order(state, 0U, std::span<const DefId>(invalid_order)),
        std::invalid_argument);

    std::uint8_t before[MAX_SLOTS]{};
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        before[slot] = state.players[0].hand[slot];
    }
    REQUIRE(mask_of(state).test(play_action(DEF_SMITHY)));
    (void)Game::step(state, play_action(DEF_SMITHY));
    for (std::size_t i = 0; i < 3U; ++i) {
        const Slot slot = slot_of(state, order[i]);
        REQUIRE(slot != NONE);
        ++before[slot];
    }
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (state.slot_to_def[slot] == DEF_SMITHY) {
            --before[slot];
        }
        CHECK(state.players[0].hand[slot] == before[slot]);
    }
    validate_game(state);
}

TEST_CASE(
    "v2 validate catches deliberate card-conservation corruption",
    "[v2][state-builder]") {
    GameState state = Game::new_game(builder_setup(), 0xA11D'0001ULL);
    REQUIRE_NOTHROW(validate_game(state));
    REQUIRE(pile_for(state, DEF_SILVER).count > 0U);
    --pile_for(state, DEF_SILVER).count;
    REQUIRE_THROWS_AS(validate_game(state), std::logic_error);
}

TEST_CASE(
    "v2 seeded interrupt frames expose the native decisions",
    "[v2][state-builder][attack]") {
    const GameState base = Game::new_game(builder_setup(), 0x1A7E'0001ULL);

    SECTION("Moat reaction") {
        Snapshot snapshot = snapshot_of(base, 1U);
        snapshot_gain_to_hand(snapshot, 1U, DEF_MOAT);
        snapshot_move_supply_to(
            snapshot, 0U, DEF_MILITIA, snapshot.players[0].in_play);
        snapshot.interrupt = SeededInterrupt::MoatReaction;
        snapshot.attacker = 0U;
        snapshot.defender = 1U;

        const GameState state = build_game_from_snapshot(snapshot);
        const ActionMask legal = mask_of(state);
        REQUIRE(mask_count(legal) == 2);
        REQUIRE(legal.test(A_PASS));
        REQUIRE(legal.test(select_action(DEF_MOAT)));
    }

    SECTION("Militia discard-down-to-three") {
        Snapshot snapshot = snapshot_of(base, 1U);
        while (snapshot.players[1].hand_count <= 3U) {
            snapshot_gain_to_hand(snapshot, 1U, DEF_COPPER);
        }
        snapshot_move_supply_to(
            snapshot, 0U, DEF_MILITIA, snapshot.players[0].in_play);
        snapshot.interrupt = SeededInterrupt::MilitiaDiscard;
        snapshot.attacker = 0U;
        snapshot.defender = 1U;

        const GameState state = build_game_from_snapshot(snapshot);
        const ActionMask legal = mask_of(state);
        int distinct_hand_defs = 0;
        for (DefId def = 0; def < card_def_count(); ++def) {
            if (snapshot.players[1].hand.by_def[def] != 0U) {
                ++distinct_hand_defs;
                REQUIRE(legal.test(select_action(def)));
            }
        }
        REQUIRE(mask_count(legal) == distinct_hand_defs);
        REQUIRE_FALSE(legal.test(A_PASS));
    }

    SECTION("Bureaucrat Victory topdeck") {
        Snapshot snapshot = snapshot_of(base, 1U);
        if (snapshot.players[1].hand.by_def[DEF_ESTATE] == 0U) {
            snapshot_gain_to_hand(snapshot, 1U, DEF_ESTATE);
        }
        snapshot_move_supply_to(
            snapshot, 0U, DEF_BUREAUCRAT, snapshot.players[0].in_play);
        snapshot.interrupt = SeededInterrupt::BureaucratTopdeck;
        snapshot.attacker = 0U;
        snapshot.defender = 1U;

        const GameState state = build_game_from_snapshot(snapshot);
        const ActionMask legal = mask_of(state);
        REQUIRE(legal.test(select_action(DEF_ESTATE)));
        REQUIRE_FALSE(legal.test(A_PASS));
        for (DefId def = 0; def < card_def_count(); ++def) {
            if ((card_def(def).types & TYPE_VICTORY) == 0U) {
                REQUIRE_FALSE(legal.test(select_action(def)));
            }
        }
    }

    SECTION("Bandit revealed-Treasure trash") {
        Snapshot snapshot = snapshot_of(base, 1U);
        snapshot_move_supply_to(
            snapshot, 0U, DEF_BANDIT, snapshot.players[0].in_play);
        snapshot_move_supply_to(
            snapshot, 1U, DEF_SILVER, snapshot.players[1].set_aside);
        snapshot_move_supply_to(
            snapshot, 1U, DEF_GOLD, snapshot.players[1].set_aside);
        snapshot.interrupt = SeededInterrupt::BanditTrash;
        snapshot.attacker = 0U;
        snapshot.defender = 1U;

        const GameState state = build_game_from_snapshot(snapshot);
        const ActionMask legal = mask_of(state);
        REQUIRE(mask_count(legal) == 2);
        REQUIRE(legal.test(select_action(DEF_SILVER)));
        REQUIRE(legal.test(select_action(DEF_GOLD)));
    }
}

TEST_CASE(
    "v2 state builder rejects impossible snapshots",
    "[v2][state-builder]") {
    const GameState state = Game::new_game(builder_setup(), 0xBAD5'0001ULL);
    SECTION("inconsistent hidden counts") {
        Snapshot snapshot = snapshot_of(state, 0U);
        ++snapshot.players[1].deck_count;
        REQUIRE_THROWS_AS(
            build_game_from_snapshot(snapshot), std::invalid_argument);
    }
    SECTION("unknown phase") {
        Snapshot snapshot = snapshot_of(state, 0U);
        snapshot.phase = static_cast<SnapshotPhase>(255U);
        REQUIRE_THROWS_AS(
            build_game_from_snapshot(snapshot), std::invalid_argument);
    }
    SECTION("bad seats") {
        Snapshot snapshot = snapshot_of(state, 0U);
        snapshot.current_player = snapshot.num_players;
        REQUIRE_THROWS_AS(
            build_game_from_snapshot(snapshot), std::invalid_argument);
    }
}
