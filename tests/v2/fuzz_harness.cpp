#include "fuzz_harness.h"

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/turns.h"
#include "v2/drivers/bots.h"

#include <cstdint>
#include <cstdio>

namespace {

constexpr DefId kImplementedKingdoms[] = {
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
    DEF_LIBRARY,
    DEF_SENTRY,
};

constexpr std::uint8_t kImplementedKingdomCount =
    static_cast<std::uint8_t>(sizeof(kImplementedKingdoms) / sizeof(kImplementedKingdoms[0]));
constexpr std::uint64_t kSetupSeedSalt = 0x51A7'2D3C'9E01'0001ULL;
constexpr std::uint64_t kGameSeedSalt = 0x6A09'E667'F3BC'C909ULL;
constexpr std::uint64_t kBotSeedSalt = 0xB007'F00D'CAFE'0000ULL;

template <typename... Args>
[[nodiscard]] InvariantViolation fail(const char* format, Args... args) noexcept {
    InvariantViolation violation{};
    violation.ok = false;
    if constexpr (sizeof...(args) == 0U) {
        (void)std::snprintf(violation.message, sizeof(violation.message), "%s", format);
    } else {
        (void)std::snprintf(violation.message, sizeof(violation.message), format, args...);
    }
    return violation;
}

[[nodiscard]] int pile_card_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? pile.mixed_len : pile.count;
}

[[nodiscard]] InvariantViolation count_ordered_zone(
    const GameState& state,
    const OrderedZone& zone,
    const char* name,
    PlayerId player_id,
    int& total) noexcept {
    if (zone.size > MAX_DECK_CARDS) {
        return fail(
            "p%u %s size exceeds cap: %u",
            static_cast<unsigned>(player_id),
            name,
            static_cast<unsigned>(zone.size));
    }
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        if (zone.cards[i] >= state.num_slots) {
            return fail(
                "p%u %s has invalid slot %u",
                static_cast<unsigned>(player_id),
                name,
                static_cast<unsigned>(zone.cards[i]));
        }
        ++total;
    }
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation count_count_zone(
    const GameState& state,
    const std::uint8_t (&zone)[MAX_SLOTS],
    const char* name,
    PlayerId player_id,
    int& total) noexcept {
    int zone_total = 0;
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        if (slot >= state.num_slots && zone[slot] != 0U) {
            return fail(
                "p%u %s has count for invalid slot %u",
                static_cast<unsigned>(player_id),
                name,
                static_cast<unsigned>(slot));
        }
        zone_total += zone[slot];
    }
    if (zone_total > MAX_DECK_CARDS) {
        return fail(
            "p%u %s count exceeds cap: %d",
            static_cast<unsigned>(player_id),
            name,
            zone_total);
    }
    total += zone_total;
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation count_player_cards(
    const GameState& state,
    PlayerId player_id,
    int& total) noexcept {
    const PlayerState& player = state.players[player_id];
    InvariantViolation violation = count_count_zone(state, player.hand, "hand", player_id, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_count_zone(state, player.exile, "exile", player_id, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_count_zone(state, player.tavern, "tavern", player_id, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_count_zone(state, player.island_mat, "island", player_id, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_ordered_zone(state, player.deck, "deck", player_id, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_ordered_zone(state, player.discard, "discard", player_id, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_ordered_zone(state, player.set_aside, "set_aside", player_id, total);
    if (!violation.ok) {
        return violation;
    }
    if (player.in_play_size > MAX_IN_PLAY) {
        return fail(
            "p%u in_play exceeds cap: %u",
            static_cast<unsigned>(player_id),
            static_cast<unsigned>(player.in_play_size));
    }
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        if (player.in_play[i].slot >= state.num_slots || player.in_play[i].behaves_as >= state.num_slots) {
            return fail(
                "p%u in_play has invalid slot",
                static_cast<unsigned>(player_id));
        }
        ++total;
    }
    if (player.pending_size > MAX_PENDING) {
        return fail(
            "p%u pending exceeds cap: %u",
            static_cast<unsigned>(player_id),
            static_cast<unsigned>(player.pending_size));
    }
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation count_trash(
    const GameState& state,
    int& total) noexcept {
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        if (slot >= state.num_slots && state.trash[slot] != 0U) {
            return fail("trash has count for invalid slot %u", static_cast<unsigned>(slot));
        }
        total += state.trash[slot];
    }
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation count_frame_local_cards(
    const GameState& state,
    int& total) noexcept {
    for (std::uint8_t i = 0; i < state.effect_depth; ++i) {
        const EffectFrame& frame = state.effect_stack[i];
        if (frame.source != DEF_BANDIT) {
            continue;
        }
        if (frame.data[3] < 0 || frame.data[3] > 2) {
            return fail("Bandit frame has invalid revealed count %d", static_cast<int>(frame.data[3]));
        }
        const std::uint8_t revealed_count = static_cast<std::uint8_t>(frame.data[3]);
        for (std::uint8_t j = 0; j < revealed_count; ++j) {
            const std::int16_t raw_slot = frame.data[1 + j];
            if (raw_slot < 0) {
                continue;
            }
            if (raw_slot >= state.num_slots) {
                return fail("Bandit frame has invalid revealed slot %d", static_cast<int>(raw_slot));
            }
            ++total;
        }
    }
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation count_piles(
    const GameState& state,
    const InvariantBaseline& baseline,
    int& total) noexcept {
    if (state.num_piles != baseline.num_piles) {
        return fail(
            "num_piles changed: %u vs %u",
            static_cast<unsigned>(state.num_piles),
            static_cast<unsigned>(baseline.num_piles));
    }
    if (state.num_nonsupply != baseline.num_nonsupply) {
        return fail(
            "num_nonsupply changed: %u vs %u",
            static_cast<unsigned>(state.num_nonsupply),
            static_cast<unsigned>(baseline.num_nonsupply));
    }
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        const int count = pile_card_count(pile);
        if (count > baseline.pile_counts[i]) {
            return fail("pile %u grew past initial count", static_cast<unsigned>(i));
        }
        if (pile.base >= state.num_slots) {
            return fail("pile %u has invalid base slot", static_cast<unsigned>(i));
        }
        if (pile.mixed_len > 12U) {
            return fail("pile %u mixed_len exceeds cap", static_cast<unsigned>(i));
        }
        for (std::uint8_t mixed = 0; mixed < pile.mixed_len; ++mixed) {
            if (pile.mixed[mixed] >= state.num_slots) {
                return fail("pile %u has invalid mixed slot", static_cast<unsigned>(i));
            }
        }
        total += count;
    }
    for (std::uint8_t i = 0; i < state.num_nonsupply; ++i) {
        const Pile& pile = state.nonsupply[i];
        const int count = pile_card_count(pile);
        if (count > baseline.nonsupply_counts[i]) {
            return fail("nonsupply %u grew past initial count", static_cast<unsigned>(i));
        }
        if (pile.base >= state.num_slots) {
            return fail("nonsupply %u has invalid base slot", static_cast<unsigned>(i));
        }
        total += count;
    }
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation check_effect_stack(const GameState& state) noexcept {
    if (state.effect_depth > MAX_EFFECT_DEPTH) {
        return fail("effect_depth exceeds cap: %u", static_cast<unsigned>(state.effect_depth));
    }
    for (std::uint8_t i = 0; i < state.effect_depth; ++i) {
        const EffectFrame& frame = state.effect_stack[i];
        if (frame.player >= state.num_players) {
            return fail("effect frame %u has invalid player", static_cast<unsigned>(i));
        }
        if (frame.source >= card_def_count()) {
            return fail("effect frame %u has invalid source", static_cast<unsigned>(i));
        }
        if (frame.repeats_left == 0U) {
            return fail("effect frame %u has zero repeats_left", static_cast<unsigned>(i));
        }
    }
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation check_decision(const GameState& state) noexcept {
    const DecisionKind kind = static_cast<DecisionKind>(state.decision.kind);
    if (state.decision.kind > static_cast<std::uint8_t>(DecisionKind::OrderTriggers)) {
        return fail("invalid decision kind %u", static_cast<unsigned>(state.decision.kind));
    }
    if (static_cast<Phase>(state.phase) == Phase::Over) {
        return InvariantViolation{};
    }
    if (kind == DecisionKind::None) {
        return fail("non-over state has no pending decision");
    }
    if (state.decision.player >= state.num_players) {
        return fail("decision has invalid player %u", static_cast<unsigned>(state.decision.player));
    }
    if ((kind == DecisionKind::Choose
            || kind == DecisionKind::ChooseGain
            || kind == DecisionKind::ChooseOption
            || kind == DecisionKind::ChooseOrder
            || kind == DecisionKind::ReactWindow
            || kind == DecisionKind::OrderTriggers)
        && state.effect_depth == 0U) {
        return fail("effect decision has empty stack");
    }
    if (kind != DecisionKind::PhaseAction
        && kind != DecisionKind::PhaseBuy
        && kind != DecisionKind::PhaseNight
        && state.decision.source >= card_def_count()) {
        return fail("decision has invalid source %u", static_cast<unsigned>(state.decision.source));
    }

    ActionMask legal{};
    const int legal_count = Game::legal_actions(state, legal);
    if (legal_count <= 0) {
        return fail("non-over state has no legal actions");
    }
    return InvariantViolation{};
}

[[nodiscard]] InvariantViolation check_turn_queue(const GameState& state) noexcept {
    if (state.turn_queue.size > MAX_TURN_QUEUE) {
        return fail("turn queue exceeds cap");
    }
    for (std::uint8_t i = 0; i < state.turn_queue.size; ++i) {
        const std::uint8_t index = static_cast<std::uint8_t>((state.turn_queue.head + i) % MAX_TURN_QUEUE);
        if (state.turn_queue.entries[index].player >= state.num_players) {
            return fail("turn queue has invalid player");
        }
    }
    return InvariantViolation{};
}

[[nodiscard]] std::uint64_t game_seed(std::uint64_t setup_seed) noexcept {
    return setup_seed ^ kGameSeedSalt;
}

[[nodiscard]] std::uint64_t bot_seed(std::uint64_t setup_seed, PlayerId player) noexcept {
    return (setup_seed ^ kBotSeedSalt) + (static_cast<std::uint64_t>(player) * 0x9E37'79B9'7F4A'7C15ULL);
}

void record_failure(
    FuzzResult& result,
    std::uint64_t seed,
    std::uint64_t step,
    const Setup& setup,
    const GameState& state,
    InvariantViolation violation) noexcept {
    result.violations = 1;
    result.failing_seed = seed;
    result.failing_step = step;
    result.failing_setup = setup;
    result.failing_state = state;
    result.violation = violation;
}

} // namespace

InvariantBaseline capture_baseline(const GameState& state) noexcept {
    InvariantBaseline baseline{};
    baseline.num_piles = state.num_piles;
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        baseline.pile_counts[i] = static_cast<std::uint8_t>(pile_card_count(state.piles[i]));
    }
    baseline.num_nonsupply = state.num_nonsupply;
    for (std::uint8_t i = 0; i < state.num_nonsupply; ++i) {
        baseline.nonsupply_counts[i] = static_cast<std::uint8_t>(pile_card_count(state.nonsupply[i]));
    }

    int total = 0;
    for (PlayerId player = 0; player < state.num_players; ++player) {
        (void)count_player_cards(state, player, total);
    }
    (void)count_piles(state, baseline, total);
    (void)count_trash(state, total);
    (void)count_frame_local_cards(state, total);
    baseline.total_cards = total;
    return baseline;
}

InvariantViolation check_invariants(
    const GameState& state,
    const InvariantBaseline& baseline) noexcept {
    if (state.num_players < 2U || state.num_players > MAX_PLAYERS) {
        return fail("invalid player count %u", static_cast<unsigned>(state.num_players));
    }
    if (state.num_slots > MAX_SLOTS) {
        return fail("num_slots exceeds cap: %u", static_cast<unsigned>(state.num_slots));
    }
    if (state.phase > static_cast<std::uint8_t>(Phase::Over)) {
        return fail("invalid phase %u", static_cast<unsigned>(state.phase));
    }
    if (state.coins < 0) {
        return fail("negative coins: %d", static_cast<int>(state.coins));
    }
    if (state.turn_counter > MAX_TURNS) {
        return fail("turn counter exceeds MAX_TURNS: %u", static_cast<unsigned>(state.turn_counter));
    }
    if (state.truncated != 0U
        && (state.turn_counter != MAX_TURNS || static_cast<Phase>(state.phase) != Phase::Over)) {
        return fail("truncated set outside MAX_TURNS game over");
    }

    InvariantViolation violation = check_turn_queue(state);
    if (!violation.ok) {
        return violation;
    }
    violation = check_effect_stack(state);
    if (!violation.ok) {
        return violation;
    }

    int total = 0;
    for (PlayerId player = 0; player < state.num_players; ++player) {
        violation = count_player_cards(state, player, total);
        if (!violation.ok) {
            return violation;
        }
    }
    violation = count_piles(state, baseline, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_trash(state, total);
    if (!violation.ok) {
        return violation;
    }
    violation = count_frame_local_cards(state, total);
    if (!violation.ok) {
        return violation;
    }
    if (total != baseline.total_cards) {
        return fail("card conservation failed: %d vs %d", total, baseline.total_cards);
    }

    return check_decision(state);
}

Setup random_kingdom_setup(std::uint64_t seed, PlayerId players) noexcept {
    Xoshiro256pp rng = Xoshiro256pp::seeded(seed ^ kSetupSeedSalt);
    Setup setup{};
    setup.num_players = players;
    if (setup.num_players < 2U || setup.num_players > MAX_PLAYERS) {
        setup.num_players = static_cast<PlayerId>(2U + rng.uniform(3U));
    }
    setup.use_colony_platinum = rng.uniform(5U) == 0U;

    DefId pool[kImplementedKingdomCount]{};
    for (std::uint8_t i = 0; i < kImplementedKingdomCount; ++i) {
        pool[i] = kImplementedKingdoms[i];
    }
    setup.kingdom_count = 10;
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        const std::uint8_t remaining = static_cast<std::uint8_t>(kImplementedKingdomCount - i);
        const std::uint8_t offset = static_cast<std::uint8_t>(rng.uniform(remaining));
        const std::uint8_t index = static_cast<std::uint8_t>(i + offset);
        const DefId chosen = pool[index];
        pool[index] = pool[i];
        pool[i] = chosen;
        setup.kingdom[i] = chosen;
    }
    return setup;
}

FuzzResult run_fuzz(const FuzzConfig& config) noexcept {
    FuzzResult result{};
    for (std::uint64_t game_index = 0;
         game_index < config.seeds && result.steps < config.steps_budget;
         ++game_index) {
        const std::uint64_t seed = config.seed_start + game_index;
        const Setup setup = random_kingdom_setup(seed, config.players);
        GameState state = Game::new_game(setup, game_seed(seed));
        const InvariantBaseline baseline = capture_baseline(state);
        ++result.games;

        InvariantViolation violation = check_invariants(state, baseline);
        if (!violation.ok) {
            record_failure(result, seed, 0, setup, state, violation);
            return result;
        }

        RandomBot bots[MAX_PLAYERS] = {
            RandomBot(bot_seed(seed, 0U)),
            RandomBot(bot_seed(seed, 1U)),
            RandomBot(bot_seed(seed, 2U)),
            RandomBot(bot_seed(seed, 3U)),
        };

        bool done = static_cast<Phase>(state.phase) == Phase::Over;
        std::uint64_t game_step = 0;
        while (!done && result.steps < config.steps_budget) {
            ActionMask legal{};
            const int legal_count = Game::legal_actions(state, legal);
            if (legal_count <= 0) {
                record_failure(result, seed, game_step, setup, state, fail("no legal action before step"));
                return result;
            }
            const PlayerId player = Game::current_decision(state).player;
            if (player >= state.num_players) {
                record_failure(result, seed, game_step, setup, state, fail("invalid decision player"));
                return result;
            }
            const Action action = bots[player].choose_action(state, legal, legal_count);
            done = Game::step(state, action);
            ++game_step;
            ++result.steps;

            violation = check_invariants(state, baseline);
            if (!violation.ok) {
                record_failure(result, seed, game_step, setup, state, violation);
                return result;
            }
        }
    }
    return result;
}

void print_fuzz_summary(std::FILE* file, const FuzzResult& result) noexcept {
    (void)std::fprintf(
        file,
        "games=%llu steps=%llu violations=%llu\n",
        static_cast<unsigned long long>(result.games),
        static_cast<unsigned long long>(result.steps),
        static_cast<unsigned long long>(result.violations));
}

void print_fuzz_failure(std::FILE* file, const FuzzResult& result) noexcept {
    if (result.violations == 0U) {
        return;
    }
    const GameState& state = result.failing_state;
    (void)std::fprintf(
        file,
        "violation: %s\nseed=%llu game_step=%llu global_steps=%llu\n",
        result.violation.message,
        static_cast<unsigned long long>(result.failing_seed),
        static_cast<unsigned long long>(result.failing_step),
        static_cast<unsigned long long>(result.steps));
    (void)std::fprintf(
        file,
        "players=%u colony=%u kingdom=",
        static_cast<unsigned>(result.failing_setup.num_players),
        result.failing_setup.use_colony_platinum ? 1U : 0U);
    for (std::uint8_t i = 0; i < result.failing_setup.kingdom_count; ++i) {
        const DefId def = result.failing_setup.kingdom[i];
        (void)std::fprintf(file, "%s%s", i == 0U ? "" : ",", card_def(def).name);
    }
    (void)std::fprintf(
        file,
        "\nphase=%u turn=%u truncated=%u actions=%u buys=%u coins=%d decision=(kind=%u player=%u source=%u) effects=%u piles=%u\n",
        static_cast<unsigned>(state.phase),
        static_cast<unsigned>(state.turn_counter),
        static_cast<unsigned>(state.truncated),
        static_cast<unsigned>(state.actions),
        static_cast<unsigned>(state.buys),
        static_cast<int>(state.coins),
        static_cast<unsigned>(state.decision.kind),
        static_cast<unsigned>(state.decision.player),
        static_cast<unsigned>(state.decision.source),
        static_cast<unsigned>(state.effect_depth),
        static_cast<unsigned>(state.num_piles));
    for (PlayerId player = 0; player < state.num_players; ++player) {
        const PlayerState& p = state.players[player];
        (void)std::fprintf(
            file,
            "p%u deck=%u discard=%u set_aside=%u in_play=%u pending=%u debt=%u\n",
            static_cast<unsigned>(player),
            static_cast<unsigned>(p.deck.size),
            static_cast<unsigned>(p.discard.size),
            static_cast<unsigned>(p.set_aside.size),
            static_cast<unsigned>(p.in_play_size),
            static_cast<unsigned>(p.pending_size),
            static_cast<unsigned>(p.debt));
    }
}
