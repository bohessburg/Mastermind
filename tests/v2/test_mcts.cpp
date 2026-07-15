#include "v2/mcts/tree.h"
#include "v2/mcts/selfplay.h"

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/setup.h"
#include "v2/core/turns.h"
#include "v2/drivers/bots.h"

#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace {

[[nodiscard]] Pile* find_pile(GameState& state, DefId def) {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        Pile& pile = state.piles[i];
        if (pile.mixed_len == 0U && state.slot_to_def[pile.base] == def) {
            return &pile;
        }
    }
    return nullptr;
}

void clear_player_cards(GameState& state, PlayerId player_id) {
    PlayerState& player = state.players[player_id];
    for (Slot slot = 0; slot < MAX_SLOTS; ++slot) {
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

void add_to_discard(GameState& state, PlayerId player_id, DefId def, std::uint8_t count) {
    const Slot slot = slot_of(state, def);
    PlayerState& player = state.players[player_id];
    for (std::uint8_t i = 0; i < count; ++i) {
        player.discard.cards[player.discard.size] = slot;
        ++player.discard.size;
    }
}

[[nodiscard]] GameState buy_position(std::int16_t coins) {
    GameState state = Game::new_game(Setup{}, 101U);
    for (Slot slot = 0; slot < MAX_SLOTS; ++slot) {
        state.players[0].hand[slot] = 0;
    }
    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.actions = 0U;
    state.buys = 1U;
    state.coins = coins;
    refresh_current_decision(state);
    return state;
}

[[nodiscard]] GameState action_rich_position() {
    Setup setup{};
    setup.kingdom_count = 5U;
    setup.kingdom[0] = DEF_VILLAGE;
    setup.kingdom[1] = DEF_SMITHY;
    setup.kingdom[2] = DEF_MOAT;
    setup.kingdom[3] = DEF_MARKET;
    setup.kingdom[4] = DEF_FESTIVAL;
    GameState state = Game::new_game(setup, 0xAC710A01ULL);
    clear_player_cards(state, 0U);
    for (DefId def : {DEF_VILLAGE, DEF_SMITHY, DEF_MOAT, DEF_MARKET, DEF_FESTIVAL}) {
        state.players[0].hand[slot_of(state, def)] = 1U;
    }
    state.phase = static_cast<std::uint8_t>(Phase::Action);
    state.actions = 4U;
    state.buys = 1U;
    refresh_current_decision(state);
    return state;
}

[[nodiscard]] GameState pile_clock_buy_position(bool player_ahead) {
    Setup setup{};
    setup.kingdom_count = 1U;
    setup.kingdom[0] = DEF_VILLAGE;
    GameState state = Game::new_game(setup, 0xC10C'0001ULL);
    clear_player_cards(state, 0U);
    clear_player_cards(state, 1U);
    add_to_discard(state, player_ahead ? 0U : 1U, DEF_PROVINCE, 1U);
    add_to_discard(state, player_ahead ? 1U : 0U, DEF_ESTATE, 1U);

    Pile* copper = find_pile(state, DEF_COPPER);
    Pile* gold = find_pile(state, DEF_GOLD);
    Pile* village = find_pile(state, DEF_VILLAGE);
    REQUIRE(copper != nullptr);
    REQUIRE(gold != nullptr);
    REQUIRE(village != nullptr);
    copper->count = 0U;
    gold->count = 0U;
    village->count = 1U;

    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.actions = 0U;
    state.buys = 1U;
    state.coins = 3;
    refresh_current_decision(state);
    return state;
}

[[nodiscard]] bool legal_in_state(const GameState& state, Action action) {
    ActionMask legal{};
    (void)Game::legal_actions(state, legal);
    return legal.test(action);
}

[[nodiscard]] std::int16_t terminal_score_after_buy(DefId def) {
    GameState state = buy_position(8);
    Pile* provinces = find_pile(state, DEF_PROVINCE);
    REQUIRE(provinces != nullptr);
    provinces->count = 1U;
    (void)Game::step(state, buy_action(def));
    return score(state, 0U);
}

[[nodiscard]] std::uint32_t root_child_count(const Mcts& search) {
    std::uint32_t count = 0U;
    for (std::uint32_t child = search.node(0U).first_child; child != MCTS_NULL;
         child = search.node(child).next_sibling) {
        ++count;
    }
    return count;
}

[[nodiscard]] std::uint32_t node_child_count(const Mcts& search, std::uint32_t node) {
    std::uint32_t count = 0U;
    for (std::uint32_t child = search.node(node).first_child; child != MCTS_NULL;
         child = search.node(child).next_sibling) {
        ++count;
    }
    return count;
}

[[nodiscard]] ActionMask root_child_actions(const Mcts& search) {
    ActionMask actions{};
    for (std::uint32_t child = search.node(0U).first_child; child != MCTS_NULL;
         child = search.node(child).next_sibling) {
        actions.set(search.node(child).action_from_parent);
    }
    return actions;
}

[[nodiscard]] ActionMask node_child_actions(const Mcts& search, std::uint32_t node) {
    ActionMask actions{};
    for (std::uint32_t child = search.node(node).first_child; child != MCTS_NULL;
         child = search.node(child).next_sibling) {
        actions.set(search.node(child).action_from_parent);
    }
    return actions;
}

[[nodiscard]] bool same_action_mask(const ActionMask& lhs, const ActionMask& rhs) {
    for (std::uint16_t word = 0U; word < ACTION_MASK_WORDS; ++word) {
        if (lhs.words[word] != rhs.words[word]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] ActionMask highest_prior_actions(
    const ActionMask& legal,
    const std::array<float, ACTION_SPACE_SIZE>& priors,
    int count) {
    ActionMask selected{};
    for (int rank = 0; rank < count; ++rank) {
        Action best = A_PASS;
        float best_prior = -1.0F;
        bool found = false;
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (!legal.test(action) || selected.test(action)) {
                continue;
            }
            if (!found || priors[action] > best_prior
                || (priors[action] == best_prior && action < best)) {
                best = action;
                best_prior = priors[action];
                found = true;
            }
        }
        REQUIRE(found);
        selected.set(best);
    }
    return selected;
}

struct MctsPlayer {
    Mcts search;
    MctsConfig config;

    explicit MctsPlayer(const MctsConfig& cfg) : search(cfg), config(cfg) {}

    [[nodiscard]] Action choose(const GameState& state, const ActionMask&, int) noexcept {
        return search.choose(state, current_player(state));
    }
};

[[nodiscard]] GameResult run_mcts_vs_random(std::uint64_t seed, const MctsConfig& config) {
    GameState state = Game::new_game(Setup{}, seed);
    MctsPlayer mcts(config);
    RandomBot random(seed ^ 0xA53A'9D15ULL);
    std::uint16_t guard = 0;
    while (state.phase != static_cast<std::uint8_t>(Phase::Over) && guard < 6000U) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(state, legal);
        REQUIRE(legal_count > 0);
        const PlayerId player = current_player(state);
        const Action action = player == 0U
            ? mcts.choose(state, legal, legal_count)
            : random.choose_action(state, legal, legal_count);
        REQUIRE(legal.test(action));
        (void)Game::step(state, action);
        ++guard;
    }
    REQUIRE(state.phase == static_cast<std::uint8_t>(Phase::Over));

    GameResult result{};
    result.scores[0] = score(state, 0U);
    result.scores[1] = score(state, 1U);
    result.turns = state.turn_counter;
    result.truncated = state.truncated != 0U;
    if (result.scores[0] > result.scores[1]) {
        result.winner = 0U;
    } else if (result.scores[1] > result.scores[0]) {
        result.winner = 1U;
    } else {
        result.winner = NONE;
    }
    return result;
}

} // namespace

TEST_CASE("v2 MCTS visit-scaled PUCT has exact reference values", "[v2][mcts][puct]") {
    MctsConfig config{};
    config.c_puct_schedule = MctsCPuctSchedule::VisitScaled;
    config.c_puct_init = 1.25F;

    config.c_puct_base = 19652.0F;
    REQUIRE(mcts_effective_c_puct(config, 0U) == 1.2500509F);
    REQUIRE(mcts_effective_c_puct(config, 1600U) == 1.3283190F);

    config.c_puct_base = 500.0F;
    REQUIRE(mcts_effective_c_puct(config, 0U) == 1.2519980F);
    REQUIRE(mcts_effective_c_puct(config, 1600U) == 2.6855607F);
}

TEST_CASE("v2 MCTS fixed PUCT preserves seeded tree selection", "[v2][mcts][puct]") {
    const GameState state = buy_position(3);
    MctsConfig legacy{};
    legacy.sims_per_move = 48U;
    legacy.c_puct = 1.75F;
    legacy.determinizations = 1U;
    legacy.max_tree_nodes = 256U;
    legacy.rollout_step_cap = 0U;
    legacy.rollout_seed = 0xC0FF'EE01ULL;

    MctsConfig fixed = legacy;
    fixed.c_puct_schedule = MctsCPuctSchedule::Fixed;
    // Fixed selection must ignore both visit-scaled parameters completely.
    fixed.c_puct_init = 9.0F;
    fixed.c_puct_base = 0.5F;
    REQUIRE(mcts_effective_c_puct(fixed, 1600U) == fixed.c_puct);

    Mcts legacy_search(legacy);
    Mcts fixed_search(fixed);
    legacy_search.reset(state, 0U);
    fixed_search.reset(state, 0U);
    Xoshiro256pp legacy_rng = Xoshiro256pp::seeded(legacy.rollout_seed);
    Xoshiro256pp fixed_rng = Xoshiro256pp::seeded(legacy.rollout_seed);
    legacy_search.run_simulations(legacy.sims_per_move, legacy_rng);
    fixed_search.run_simulations(fixed.sims_per_move, fixed_rng);

    REQUIRE(fixed_search.best_root_action() == legacy_search.best_root_action());
    REQUIRE(fixed_search.node_count() == legacy_search.node_count());
    for (std::uint32_t index = 0U; index < legacy_search.node_count(); ++index) {
        const MctsNode& expected = legacy_search.node(index);
        const MctsNode& actual = fixed_search.node(index);
        REQUIRE(actual.parent == expected.parent);
        REQUIRE(actual.first_child == expected.first_child);
        REQUIRE(actual.next_sibling == expected.next_sibling);
        REQUIRE(actual.action_from_parent == expected.action_from_parent);
        REQUIRE(actual.visits == expected.visits);
        REQUIRE(actual.value_sum == expected.value_sum);
        REQUIRE(actual.prior == expected.prior);
    }
}

TEST_CASE("v2 MCTS prefers Province over Estate with eight coins", "[v2][mcts]") {
    GameState state = buy_position(8);
    Pile* provinces = find_pile(state, DEF_PROVINCE);
    REQUIRE(provinces != nullptr);
    provinces->count = 1U;

    MctsConfig config{};
    config.sims_per_move = 160U;
    config.determinizations = 1U;
    config.max_tree_nodes = 4096U;
    config.rollout_seed = 0xBEEF'1001ULL;

    Mcts search(config);
    search.reset(state, 0U);
    Xoshiro256pp rng = Xoshiro256pp::seeded(config.rollout_seed);
    search.run_simulations(config.sims_per_move, rng);

    REQUIRE(legal_in_state(state, buy_action(DEF_PROVINCE)));
    REQUIRE(legal_in_state(state, buy_action(DEF_ESTATE)));
    REQUIRE(search.best_root_action() == buy_action(DEF_PROVINCE));
    REQUIRE(terminal_score_after_buy(DEF_PROVINCE) > terminal_score_after_buy(DEF_ESTATE));
}

TEST_CASE("v2 MCTS is deterministic for the same seed and config", "[v2][mcts]") {
    GameState state = buy_position(6);
    MctsConfig config{};
    config.sims_per_move = 80U;
    config.determinizations = 2U;
    config.max_tree_nodes = 2048U;
    config.rollout_seed = 0xD37E'0011ULL;

    const Action first = mcts_choose(state, 0U, config);
    const Action second = mcts_choose(state, 0U, config);
    REQUIRE(first == second);
}

TEST_CASE("v2 MCTS slab exhaustion degrades to a legal action", "[v2][mcts]") {
    GameState state = buy_position(8);
    MctsConfig config{};
    config.sims_per_move = 100U;
    config.determinizations = 1U;
    config.max_tree_nodes = 2U;
    config.rollout_seed = 0x510B'0001ULL;

    Mcts search(config);
    const Action action = search.choose(state, 0U);

    REQUIRE(search.exhausted());
    REQUIRE(legal_in_state(state, action));
}

TEST_CASE("v2 MCTS root expansion and noise remain full-width with top-k", "[v2][mcts][topk]") {
    GameState state = buy_position(8);
    MctsConfig config{};
    config.determinizations = 1U;
    config.max_tree_nodes = 128U;
    config.rollout_policy = MctsRolloutPolicy::External;
    config.expand_top_k = 2U;

    Mcts search(config);
    search.reset(state, 0U);
    MctsPendingLeaf leaf{};
    REQUIRE(search.collect_external_leaf(leaf));
    REQUIRE(leaf.legal_count > static_cast<int>(config.expand_top_k));

    std::array<float, ACTION_SPACE_SIZE> priors{};
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (leaf.legal.test(action)) {
            priors[action] = static_cast<float>(action + 1U);
        }
    }
    std::array<float, ACTION_SPACE_SIZE> root_noised{};
    float prior_sum = 0.0F;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (leaf.legal.test(action)) {
            root_noised[action] = priors[action];
            prior_sum += root_noised[action];
        }
    }
    REQUIRE(prior_sum > 0.0F);
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        root_noised[action] = leaf.legal.test(action)
            ? root_noised[action] / prior_sum
            : 0.0F;
    }
    Xoshiro256pp noise_rng = Xoshiro256pp::seeded(0xD1A1'C4E7ULL);
    mcts_add_dirichlet_noise(
        root_noised.data(),
        leaf.legal,
        leaf.legal_count,
        0.30F,
        1.0F,
        noise_rng);
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (!leaf.legal.test(action)) {
            REQUIRE(root_noised[action] == 0.0F);
        } else {
            REQUIRE(root_noised[action] > 0.0F);
        }
    }
    search.provide_external_evaluation(leaf, 0.0F, root_noised.data());
    REQUIRE(root_child_count(search) == static_cast<std::uint32_t>(leaf.legal_count));
    REQUIRE(same_action_mask(root_child_actions(search), leaf.legal));
}

TEST_CASE("v2 MCTS non-root top-k keeps a deterministic off-prior wildcard", "[v2][mcts][topk]") {
    GameState state = action_rich_position();
    MctsConfig config{};
    config.determinizations = 1U;
    config.max_tree_nodes = 128U;
    config.rollout_policy = MctsRolloutPolicy::External;
    config.rollout_seed = 0x70B0'B001ULL;
    config.expand_top_k = 3U;

    const auto prepare_interior_leaf = [&state, &config](Mcts& search, Xoshiro256pp& rng) {
        search.reset(state, 0U);
        MctsPendingLeaf root{};
        REQUIRE(search.collect_external_leaf(root));
        std::array<float, ACTION_SPACE_SIZE> root_priors{};
        root_priors[play_action(DEF_VILLAGE)] = 1.0F;
        search.provide_external_evaluation(root, 0.0F, root_priors.data(), rng);
        REQUIRE(root_child_count(search) == static_cast<std::uint32_t>(root.legal_count));

        MctsPendingLeaf interior{};
        REQUIRE(search.collect_external_leaf(interior));
        REQUIRE(interior.node != 0U);
        REQUIRE(interior.legal_count > static_cast<int>(config.expand_top_k));
        return interior;
    };

    Mcts first(config);
    Xoshiro256pp first_rng = Xoshiro256pp::seeded(config.rollout_seed);
    const MctsPendingLeaf first_leaf = prepare_interior_leaf(first, first_rng);
    std::array<float, ACTION_SPACE_SIZE> ranked{};
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (first_leaf.legal.test(action)) {
            ranked[action] = static_cast<float>(action + 1U);
        }
    }
    const ActionMask ranked_children = highest_prior_actions(
        first_leaf.legal,
        ranked,
        static_cast<int>(config.expand_top_k) - 1);
    first.provide_external_evaluation(first_leaf, 0.0F, ranked.data(), first_rng);
    const ActionMask first_children = node_child_actions(first, first_leaf.node);
    REQUIRE(node_child_count(first, first_leaf.node) == config.expand_top_k);
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (ranked_children.test(action)) {
            REQUIRE(first_children.test(action));
        }
    }
    int wildcard_count = 0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (first_children.test(action) && !ranked_children.test(action)) {
            REQUIRE(first_leaf.legal.test(action));
            ++wildcard_count;
        }
    }
    REQUIRE(wildcard_count == 1);

    Mcts second(config);
    Xoshiro256pp second_rng = Xoshiro256pp::seeded(config.rollout_seed);
    const MctsPendingLeaf second_leaf = prepare_interior_leaf(second, second_rng);
    REQUIRE(same_action_mask(second_leaf.legal, first_leaf.legal));
    second.provide_external_evaluation(second_leaf, 0.0F, ranked.data(), second_rng);
    REQUIRE(same_action_mask(
        node_child_actions(second, second_leaf.node),
        first_children));
}

TEST_CASE("v2 MCTS reuses only a hash-matching selected subtree", "[v2][mcts][reuse]") {
    GameState state = buy_position(3);
    MctsConfig config{};
    config.determinizations = 1U;
    config.max_tree_nodes = 256U;
    config.rollout_step_cap = 0U;
    config.tree_reuse = true;

    Mcts matching_search(config);
    matching_search.reset(state, 0U);
    Xoshiro256pp rng = Xoshiro256pp::seeded(0x7EEE'0001ULL);
    matching_search.run_simulations(8U, rng);
    const Action matched_action = matching_search.best_root_action();
    REQUIRE(legal_in_state(state, matched_action));
    const std::uint32_t child_visits = matching_search.root_visits_for(matched_action);
    REQUIRE(child_visits > 0U);

    GameState matching_state = state;
    (void)Game::step(matching_state, matched_action);
    REQUIRE(matching_search.retain_root_child(
        matched_action,
        mcts_state_hash(matching_state)));
    REQUIRE(matching_search.adopt_retained_root(
        matching_state,
        current_player(matching_state)));
    REQUIRE(matching_search.node(0U).visits == child_visits);
    REQUIRE(mcts_state_hash(matching_search.state_for(
        matching_search.node(0U).state_index)) == mcts_state_hash(matching_state));

    Mcts mismatching_search(config);
    mismatching_search.reset(state, 0U);
    rng = Xoshiro256pp::seeded(0x7EEE'0001ULL);
    mismatching_search.run_simulations(8U, rng);
    const Action stale_action = mismatching_search.best_root_action();
    GameState stale_child_state = state;
    (void)Game::step(stale_child_state, stale_action);
    REQUIRE(mismatching_search.retain_root_child(
        stale_action,
        mcts_state_hash(stale_child_state)));

    ActionMask intervening_legal{};
    REQUIRE(Game::legal_actions(stale_child_state, intervening_legal) > 0);
    (void)Game::step(stale_child_state, intervening_legal.nth_set(0U));
    REQUIRE(mcts_state_hash(stale_child_state) != mcts_state_hash(
        matching_search.state_for(0U)));
    REQUIRE_FALSE(mismatching_search.adopt_retained_root(
        stale_child_state,
        current_player(stale_child_state)));
    // This mirrors SelfPlayRunner::start_search's mismatch path.
    mismatching_search.reset(stale_child_state, current_player(stale_child_state));
    REQUIRE(mismatching_search.node(0U).visits == 0U);
}

TEST_CASE("v2 MCTS reuse treats adopted root visits as a visit target", "[v2][mcts][reuse]") {
    MctsConfig config{};
    config.sims_per_move = 40U;
    config.min_new_sims = 3U;
    config.determinizations = 1U;
    config.max_tree_nodes = 256U;
    config.rollout_step_cap = 0U;
    config.tree_reuse = true;

    GameState state = buy_position(3);
    Mcts search(config);
    Xoshiro256pp rng = Xoshiro256pp::seeded(0x7EEE'1001ULL);
    search.reset(state, 0U);
    search.run_simulations(8U, rng);

    const Action action = search.best_root_action();
    const std::uint32_t inherited_visits = search.root_visits_for(action);
    REQUIRE(inherited_visits > 0U);
    REQUIRE(inherited_visits < config.sims_per_move);

    GameState next_state = state;
    (void)Game::step(next_state, action);
    REQUIRE(search.retain_root_child(action, mcts_state_hash(next_state)));
    REQUIRE(search.adopt_retained_root(next_state, current_player(next_state)));
    REQUIRE(search.node(0U).visits == inherited_visits);

    const std::uint32_t new_sims = search.new_simulation_target(true);
    REQUIRE(new_sims == config.sims_per_move - inherited_visits);
    search.run_simulations(new_sims, rng);
    REQUIRE(search.node(0U).visits == inherited_visits + new_sims);
    REQUIRE(search.node(0U).visits == config.sims_per_move);
}

TEST_CASE("v2 MCTS reuse honors the minimum new-simulation floor", "[v2][mcts][reuse]") {
    MctsConfig config{};
    config.sims_per_move = 1U;
    config.min_new_sims = 5U;
    config.determinizations = 1U;
    config.max_tree_nodes = 256U;
    config.rollout_step_cap = 0U;
    config.tree_reuse = true;

    GameState state = buy_position(3);
    Mcts search(config);
    Xoshiro256pp rng = Xoshiro256pp::seeded(0x7EEE'1002ULL);
    search.reset(state, 0U);
    search.run_simulations(8U, rng);

    const Action action = search.best_root_action();
    const std::uint32_t inherited_visits = search.root_visits_for(action);
    REQUIRE(inherited_visits >= config.sims_per_move);

    GameState next_state = state;
    (void)Game::step(next_state, action);
    REQUIRE(search.retain_root_child(action, mcts_state_hash(next_state)));
    REQUIRE(search.adopt_retained_root(next_state, current_player(next_state)));

    const std::uint32_t new_sims = search.new_simulation_target(true);
    REQUIRE(new_sims == config.min_new_sims);
    search.run_simulations(new_sims, rng);
    REQUIRE(search.node(0U).visits == inherited_visits + new_sims);
}

TEST_CASE("v2 MCTS reuse keeps fresh-tree simulation budgets unchanged", "[v2][mcts][reuse]") {
    MctsConfig config{};
    config.sims_per_move = 7U;
    config.min_new_sims = 64U;
    config.determinizations = 1U;
    config.max_tree_nodes = 128U;
    config.rollout_step_cap = 0U;
    config.tree_reuse = true;

    Mcts search(config);
    const GameState state = buy_position(3);
    search.reset(state, 0U);
    REQUIRE(search.new_simulation_target(false) == config.sims_per_move);

    Xoshiro256pp rng = Xoshiro256pp::seeded(0x7EEE'1003ULL);
    search.run_simulations(search.new_simulation_target(false), rng);
    REQUIRE(search.node(0U).visits == config.sims_per_move);

    const SelfPlayConfig selfplay_config{};
    REQUIRE(config.min_new_sims == 64U);
    REQUIRE(selfplay_config.min_new_sims == 64U);
}

TEST_CASE("v2 selfplay applies the adopted-root visit target to slot counters", "[v2][mcts][reuse]") {
    SelfPlayConfig config{};
    config.n_games = 1U;
    config.sims_per_move = 12U;
    config.min_new_sims = 1U;
    config.max_batch = 1U;
    config.seed = 0x7EEE'1004ULL;
    config.kingdom_mode = SelfPlayKingdomMode::Fixed;
    config.dirichlet_frac = 0.0F;
    config.temp_moves = 0U;
    config.max_tree_nodes = 256U;
    config.tree_reuse = true;

    SelfPlayRunner runner(config);
    std::array<float, 1> values{};
    std::array<float, ACTION_SPACE_SIZE> policies{};
    bool saw_adopted_root = false;
    bool saw_adopted_completion = false;
    std::uint32_t inherited_visits = 0U;
    std::uint32_t expected_target = 0U;

    for (std::uint32_t guard = 0U; guard < 512U && !saw_adopted_completion; ++guard) {
        const std::uint32_t count = runner.collect_leaves(1U);
        const SelfPlaySearchStats before = runner.search_stats(0U);
        if (!saw_adopted_root && before.search_active
            && before.sims_started == 1U && before.sims_completed == 0U
            && before.root_visits > 0U) {
            inherited_visits = before.root_visits;
            const std::uint32_t remaining = inherited_visits >= config.sims_per_move
                ? 0U
                : config.sims_per_move - inherited_visits;
            expected_target = remaining > config.min_new_sims ? remaining : config.min_new_sims;
            REQUIRE(before.sims_target == expected_target);
            REQUIRE(expected_target < config.sims_per_move);
            saw_adopted_root = true;
        }

        if (count > 0U) {
            REQUIRE(count == 1U);
            runner.provide_evaluations(values.data(), policies.data(), count);
        }

        const SelfPlaySearchStats after = runner.search_stats(0U);
        if (saw_adopted_root && !after.search_active && after.sims_target == expected_target) {
            REQUIRE(after.sims_started == expected_target);
            REQUIRE(after.sims_completed == expected_target);
            REQUIRE(after.root_visits == inherited_visits + expected_target);
            saw_adopted_completion = true;
        }
    }

    REQUIRE(saw_adopted_root);
    REQUIRE(saw_adopted_completion);
}

TEST_CASE("v2 MCTS tree reuse rejects multiple determinizations", "[v2][mcts][reuse]") {
    MctsConfig config{};
    config.tree_reuse = true;
    config.determinizations = 2U;
    REQUIRE_THROWS_AS(Mcts(config), std::invalid_argument);
}

TEST_CASE("v2 MCTS treasure pruning forces ascending treasure plays", "[v2][mcts]") {
    GameState state = buy_position(6);
    state.players[0].hand[slot_of(state, DEF_COPPER)] = 1U;
    state.players[0].hand[slot_of(state, DEF_SILVER)] = 1U;
    refresh_current_decision(state);

    ActionMask legal{};
    REQUIRE(Game::legal_actions(state, legal) > 0);
    REQUIRE(legal.test(A_PASS));
    REQUIRE(legal.test(play_action(DEF_COPPER)));
    REQUIRE(legal.test(play_action(DEF_SILVER)));

    ActionMask filtered = mcts_filter_treasure_plays(state, legal);
    REQUIRE(filtered.test(play_action(DEF_COPPER)));
    REQUIRE(filtered.test(play_action(DEF_SILVER)));
    REQUIRE_FALSE(filtered.test(A_PASS));

    (void)Game::step(state, mcts_canonical_treasure_play(filtered));
    REQUIRE(state.players[0].in_play_size == 1U);
    REQUIRE(state.slot_to_def[state.players[0].in_play[0].slot] == DEF_COPPER);

    REQUIRE(Game::legal_actions(state, legal) > 0);
    filtered = mcts_filter_treasure_plays(state, legal);
    REQUIRE(filtered.test(play_action(DEF_SILVER)));
    REQUIRE_FALSE(filtered.test(A_PASS));
    REQUIRE(mcts_canonical_treasure_play(filtered) == play_action(DEF_SILVER));
}

TEST_CASE("v2 MCTS virtual loss bookkeeping is reversible", "[v2][mcts]") {
    GameState state = buy_position(3);
    MctsConfig config{};
    config.sims_per_move = 1U;
    config.determinizations = 1U;
    config.max_tree_nodes = 32U;

    Mcts search(config);
    search.reset(state, 0U);
    REQUIRE(search.node_count() == 1U);
    REQUIRE(search.node(0U).virtual_loss == 0.0F);

    search.add_virtual_loss(0U, 2.0F);
    REQUIRE(search.node(0U).virtual_loss == 2.0F);
    search.revert_virtual_loss(0U, 0.75F);
    REQUIRE(search.node(0U).virtual_loss == 1.25F);
    search.revert_virtual_loss(0U, 3.0F);
    REQUIRE(search.node(0U).virtual_loss == 0.0F);
}

TEST_CASE("v2 MCTS terminal values are perspective correct", "[v2][mcts]") {
    GameState state = Game::new_game(Setup{}, 303U);
    clear_player_cards(state, 0U);
    clear_player_cards(state, 1U);
    add_to_discard(state, 0U, DEF_ESTATE, 1U);
    add_to_discard(state, 1U, DEF_PROVINCE, 1U);
    state.phase = static_cast<std::uint8_t>(Phase::Over);

    REQUIRE(mcts_terminal_value(state, 0U) == -1.0F);
    REQUIRE(mcts_terminal_value(state, 1U) == 1.0F);
}

TEST_CASE("v2 MCTS rollout cutoff uses the current point margin for each perspective", "[v2][mcts][rollout]") {
    GameState state = Game::new_game(Setup{}, 304U);
    clear_player_cards(state, 0U);
    add_to_discard(state, 0U, DEF_PROVINCE, 1U);
    state.phase = static_cast<std::uint8_t>(Phase::Buy);
    state.actions = 0U;
    state.buys = 0U;
    state.coins = 0;
    refresh_current_decision(state);

    ActionMask legal{};
    REQUIRE(Game::legal_actions(state, legal) == 1);
    REQUIRE(legal.test(A_PASS));
    REQUIRE(score(state, 0U) > score(state, 1U));

    MctsConfig config{};
    config.determinizations = 1U;
    config.max_tree_nodes = 8U;
    config.rollout_step_cap = 0U;

    Mcts search(config);
    search.reset(state, 0U);
    Xoshiro256pp rng = Xoshiro256pp::seeded(config.rollout_seed);
    search.run_simulations(1U, rng);

    REQUIRE(search.node_count() == 2U);
    REQUIRE_FALSE(search.node(1U).terminal);
    REQUIRE(search.node(0U).player == 0U);
    REQUIRE(search.node(1U).player == 1U);
    REQUIRE(search.node(0U).value_sum == 1.0F);
    REQUIRE(search.node(1U).value_sum == -1.0F);

    GameState tied = state;
    clear_player_cards(tied, 0U);
    add_to_discard(tied, 0U, DEF_ESTATE, 3U);
    REQUIRE(score(tied, 0U) == score(tied, 1U));

    Mcts tied_search(config);
    tied_search.reset(tied, 0U);
    tied_search.run_simulations(1U, rng);

    REQUIRE(tied_search.node_count() == 2U);
    REQUIRE_FALSE(tied_search.node(1U).terminal);
    REQUIRE(tied_search.node(0U).value_sum == 0.0F);
    REQUIRE(tied_search.node(1U).value_sum == 0.0F);
}

TEST_CASE("v2 EngineLike rollout respects the third-pile clock", "[v2][mcts][rollout]") {
    GameState behind = pile_clock_buy_position(false);
    ActionMask legal{};
    REQUIRE(Game::legal_actions(behind, legal) > 0);
    REQUIRE(legal.test(buy_action(DEF_VILLAGE)));
    REQUIRE(score(behind, 0U) < score(behind, 1U));

    const Action behind_action = mcts_engine_like_rollout_buy(behind, legal);
    REQUIRE(behind_action != buy_action(DEF_VILLAGE));
    REQUIRE(legal.test(behind_action));

    GameState ahead = pile_clock_buy_position(true);
    REQUIRE(Game::legal_actions(ahead, legal) > 0);
    REQUIRE(legal.test(buy_action(DEF_VILLAGE)));
    REQUIRE(score(ahead, 0U) > score(ahead, 1U));
    REQUIRE(mcts_engine_like_rollout_buy(ahead, legal) == buy_action(DEF_VILLAGE));
}

TEST_CASE("v2 MCTS beats RandomBot in base games", "[v2][mcts][integration]") {
    MctsConfig config{};
    config.sims_per_move = 100U;
    config.determinizations = 1U;
    config.max_tree_nodes = 4096U;
    config.rollout_seed = 0xA17A'600DULL;

    int wins = 0;
    for (std::uint64_t seed = 0; seed < 20U; ++seed) {
        const GameResult result = run_mcts_vs_random(0x6060'0000ULL + seed, config);
        INFO("seed=" << seed << " winner=" << static_cast<int>(result.winner)
                     << " score0=" << result.scores[0] << " score1=" << result.scores[1]
                     << " turns=" << result.turns);
        if (result.winner == 0U) {
            ++wins;
        }
    }

    REQUIRE(wins >= 18);
}
