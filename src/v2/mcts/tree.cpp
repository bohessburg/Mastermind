#include "v2/mcts/tree.h"

#include "v2/core/determinize.h"
#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

[[nodiscard]] bool terminal_state(const GameState& state) noexcept {
    return state.phase == static_cast<std::uint8_t>(Phase::Over);
}

[[nodiscard]] PlayerId other_player(PlayerId player) noexcept {
    return static_cast<PlayerId>(player == 0U ? 1U : 0U);
}

[[nodiscard]] std::uint8_t safe_determinizations(const MctsConfig& config) noexcept {
    return config.determinizations == 0U ? 1U : config.determinizations;
}

[[nodiscard]] std::uint32_t safe_capacity(const MctsConfig& config) noexcept {
    return config.max_tree_nodes == 0U ? 1U : config.max_tree_nodes;
}

[[nodiscard]] float positive_prior(
    const MctsConfig& config,
    const GameState& state,
    PlayerId player,
    Action action) noexcept {
    if (config.prior_fn == nullptr) {
        return 1.0F;
    }
    const float prior = config.prior_fn(state, player, action, config.prior_user);
    return prior > 0.0F ? prior : 0.0F;
}

[[nodiscard]] DefId def_for_slot(const GameState& state, Slot slot) noexcept {
    return slot < state.num_slots ? state.slot_to_def[slot] : DEF_COPPER;
}

[[nodiscard]] bool is_action_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_ACTION) != 0U;
}

[[nodiscard]] bool is_treasure_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_TREASURE) != 0U;
}

[[nodiscard]] bool is_victory_def(DefId def) noexcept {
    return def < card_def_count() && (card_def(def).types & TYPE_VICTORY) != 0U;
}

[[nodiscard]] int pile_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? static_cast<int>(pile.mixed_len) : static_cast<int>(pile.count);
}

[[nodiscard]] DefId pile_top_def(const GameState& state, const Pile& pile) noexcept {
    const Slot slot = pile.mixed_len > 0U ? pile.mixed[pile.mixed_len - 1U] : pile.base;
    return def_for_slot(state, slot);
}

[[nodiscard]] int supply_count(const GameState& state, DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        if (pile_top_def(state, state.piles[i]) == def) {
            return pile_count(state.piles[i]);
        }
    }
    return 0;
}

[[nodiscard]] Action first_legal(const ActionMask& legal) noexcept {
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action first_legal_option(const ActionMask& legal) noexcept {
    for (Action action = A_OPTION_BASE; action < A_CALL_BASE; ++action) {
        if (legal.test(action)) {
            return action;
        }
    }
    return first_legal(legal);
}

[[nodiscard]] Action first_legal_play_treasure(const ActionMask& legal) noexcept {
    constexpr DefId kTreasures[] = {
        DEF_PLATINUM,
        DEF_GOLD,
        DEF_SILVER,
        DEF_COPPER,
        DEF_POTION,
    };
    for (const DefId def : kTreasures) {
        const Action action = play_action(def);
        if (legal.test(action)) {
            return action;
        }
    }
    return A_PASS;
}

[[nodiscard]] Action legal_buy(const ActionMask& legal, DefId def) noexcept {
    const Action action = buy_action(def);
    return legal.test(action) ? action : A_PASS;
}

[[nodiscard]] int discard_priority(DefId def) noexcept {
    if (def == DEF_CURSE) {
        return 0;
    }
    if (def == DEF_ESTATE) {
        return 1;
    }
    if (def == DEF_DUCHY) {
        return 2;
    }
    if (def == DEF_PROVINCE || def == DEF_COLONY) {
        return 3;
    }
    if (def == DEF_COPPER) {
        return 10;
    }
    if (is_victory_def(def)) {
        return 5;
    }
    if (is_action_def(def)) {
        return 15;
    }
    if (is_treasure_def(def)) {
        return 20;
    }
    return 25;
}

[[nodiscard]] int keep_priority(DefId def) noexcept {
    if (is_treasure_def(def)) {
        return 100 + card_def(def).coin_value;
    }
    if (is_action_def(def)) {
        return 50;
    }
    if (def == DEF_PROVINCE || def == DEF_COLONY) {
        return 30;
    }
    if (def == DEF_DUCHY) {
        return 20;
    }
    if (def == DEF_ESTATE || def == DEF_CURSE) {
        return 0;
    }
    return 10;
}

[[nodiscard]] int trash_priority(DefId def) noexcept {
    if (def == DEF_CURSE) {
        return 0;
    }
    if (def == DEF_ESTATE) {
        return 1;
    }
    if (def == DEF_COPPER) {
        return 2;
    }
    return 50;
}

[[nodiscard]] int gain_priority(DefId def) noexcept {
    switch (def) {
    case DEF_PROVINCE:
        return 0;
    case DEF_GOLD:
        return 10;
    case DEF_DUCHY:
        return 20;
    case DEF_SILVER:
        return 30;
    case DEF_ESTATE:
        return 40;
    case DEF_COPPER:
    case DEF_CURSE:
        return 900;
    default:
        return 100;
    }
}

[[nodiscard]] Action choose_select_min_priority(
    const ActionMask& legal,
    int (*priority)(DefId),
    bool allow_pass) noexcept {
    Action best = A_PASS;
    int best_priority = 9999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = select_action(def);
        if (!legal.test(action)) {
            continue;
        }
        const int value = priority(def);
        if (best == A_PASS || value < best_priority) {
            best = action;
            best_priority = value;
        }
    }
    if (best != A_PASS && (!allow_pass || best_priority < 50)) {
        return best;
    }
    return allow_pass && legal.test(A_PASS) ? A_PASS : (best != A_PASS ? best : first_legal(legal));
}

[[nodiscard]] Action choose_select_max_priority(
    const ActionMask& legal,
    int (*priority)(DefId),
    bool allow_pass) noexcept {
    Action best = A_PASS;
    int best_priority = -9999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = select_action(def);
        if (!legal.test(action)) {
            continue;
        }
        const int value = priority(def);
        if (best == A_PASS || value > best_priority) {
            best = action;
            best_priority = value;
        }
    }
    if (best != A_PASS) {
        return best;
    }
    return allow_pass && legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action choose_gain_most_expensive(const ActionMask& legal) noexcept {
    Action best = A_PASS;
    int best_cost = -1;
    int best_priority = 9999;
    for (DefId def = 0; def < ACTION_DEF_COUNT; ++def) {
        const Action action = select_action(def);
        if (!legal.test(action)) {
            continue;
        }
        const CardDef& card = card_def(def);
        const int cost = static_cast<int>(card.cost.coins)
            + (static_cast<int>(card.cost.potion) * 10)
            + (static_cast<int>(card.cost.debt) / 2);
        const int priority = gain_priority(def);
        if (best == A_PASS || cost > best_cost || (cost == best_cost && priority < best_priority)) {
            best = action;
            best_cost = cost;
            best_priority = priority;
        }
    }
    return best != A_PASS ? best : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
}

[[nodiscard]] Action big_money_buy(const GameState& state, const ActionMask& legal) noexcept {
    const int provinces_left = supply_count(state, DEF_PROVINCE);
    const int coins = state.coins;
    if (coins >= 8) {
        const Action province = legal_buy(legal, DEF_PROVINCE);
        if (province != A_PASS) {
            return province;
        }
    }
    if (coins >= 6) {
        if (provinces_left <= 4) {
            const Action duchy = legal_buy(legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action gold = legal_buy(legal, DEF_GOLD);
        if (gold != A_PASS) {
            return gold;
        }
    }
    if (coins == 5) {
        if (provinces_left <= 5) {
            const Action duchy = legal_buy(legal, DEF_DUCHY);
            if (duchy != A_PASS) {
                return duchy;
            }
        }
        const Action silver = legal_buy(legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins >= 3) {
        if (provinces_left <= 2) {
            const Action estate = legal_buy(legal, DEF_ESTATE);
            if (estate != A_PASS) {
                return estate;
            }
        }
        const Action silver = legal_buy(legal, DEF_SILVER);
        if (silver != A_PASS) {
            return silver;
        }
    }
    if (coins == 2 && provinces_left <= 3) {
        const Action estate = legal_buy(legal, DEF_ESTATE);
        if (estate != A_PASS) {
            return estate;
        }
    }
    return legal.test(A_PASS) ? A_PASS : first_legal(legal);
}

[[nodiscard]] Action sentry_option(const GameState& state, const ActionMask& legal) noexcept {
    DefId def = DEF_COPPER;
    if (state.effect_depth > 0U) {
        const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
        if (frame.source == DEF_SENTRY && frame.data[3] > 0 && frame.data[4] < frame.data[3]) {
            const std::uint8_t index = static_cast<std::uint8_t>(frame.data[4]);
            const Slot slot = static_cast<Slot>(frame.data[1U + index]);
            def = def_for_slot(state, slot);
        }
    }

    Action desired = option_action(2U);
    if (def == DEF_CURSE || def == DEF_ESTATE || def == DEF_COPPER) {
        desired = option_action(0U);
    } else if (def == DEF_DUCHY || def == DEF_PROVINCE || def == DEF_COLONY) {
        desired = option_action(1U);
    }
    return legal.test(desired) ? desired : first_legal_option(legal);
}

[[nodiscard]] Action rollout_random_action(
    const ActionMask& legal,
    int legal_count,
    Xoshiro256pp& rng) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    return legal.nth_set(rng.uniform(static_cast<std::uint32_t>(legal_count)));
}

[[nodiscard]] bool epsilon_explore(float epsilon, Xoshiro256pp& rng) noexcept {
    if (epsilon <= 0.0F) {
        return false;
    }
    if (epsilon > 1.0F) {
        epsilon = 1.0F;
    }
    const std::uint32_t threshold = static_cast<std::uint32_t>(epsilon * 10000.0F);
    return rng.uniform(10000U) < threshold;
}

[[nodiscard]] Action rollout_heuristic_action(
    const GameState& state,
    const ActionMask& legal,
    int legal_count,
    Xoshiro256pp& rng,
    float epsilon) noexcept {
    if (legal_count <= 0) {
        return A_PASS;
    }
    if (legal_count == 1 || epsilon_explore(epsilon, rng)) {
        return legal_count == 1 ? legal.nth_set(0U) : rollout_random_action(legal, legal_count, rng);
    }

    /*
     * Local BigMoney-class rollout policy. This intentionally duplicates the
     * driver bot's core choices so MCTS keeps no dependency on drivers.
     */
    const DecisionKind decision = static_cast<DecisionKind>(state.decision.kind);
    if (decision == DecisionKind::PhaseBuy) {
        const Action treasure = first_legal_play_treasure(legal);
        if (treasure != A_PASS) {
            return treasure;
        }
        return big_money_buy(state, legal);
    }
    if (decision == DecisionKind::PhaseAction || decision == DecisionKind::PhaseNight) {
        return legal.test(A_PASS) ? A_PASS : first_legal(legal);
    }
    if (decision == DecisionKind::ReactWindow) {
        const Action moat = select_action(DEF_MOAT);
        return legal.test(moat) ? moat : (legal.test(A_PASS) ? A_PASS : first_legal(legal));
    }
    if (decision == DecisionKind::OrderTriggers || decision == DecisionKind::ChooseOrder) {
        return first_legal_option(legal);
    }
    if (decision == DecisionKind::ChooseOption) {
        if (state.decision.source == DEF_SENTRY) {
            return sentry_option(state, legal);
        }
        const Action decline = option_action(0U);
        return legal.test(decline) ? decline : first_legal_option(legal);
    }
    if (decision == DecisionKind::ChooseGain) {
        return choose_gain_most_expensive(legal);
    }
    if (decision == DecisionKind::Choose) {
        const bool pass_allowed = legal.test(A_PASS) && state.decision.min_left == 0U;
        const DefId source = static_cast<DefId>(state.decision.source);
        if (source == DEF_MILITIA) {
            return choose_select_max_priority(legal, keep_priority, false);
        }
        if (source == DEF_CELLAR || source == DEF_POACHER) {
            return choose_select_min_priority(legal, discard_priority, pass_allowed);
        }
        if (source == DEF_CHAPEL || source == DEF_REMODEL || source == DEF_MINE
            || source == DEF_MONEYLENDER || source == DEF_BANDIT) {
            return choose_select_min_priority(legal, trash_priority, pass_allowed);
        }
        if (source == DEF_BUREAUCRAT || source == DEF_ARTISAN) {
            return choose_select_min_priority(legal, discard_priority, pass_allowed);
        }
        if (source == DEF_HARBINGER) {
            return choose_select_max_priority(legal, keep_priority, pass_allowed);
        }
        return pass_allowed ? A_PASS : first_legal(legal);
    }

    return legal.test(A_PASS) && state.decision.min_left == 0U ? A_PASS : first_legal(legal);
}

} // namespace

Mcts::Mcts(const MctsConfig& config)
    : config_(config),
      nodes_(new MctsNode[safe_capacity(config)]),
      states_(new GameState[safe_capacity(config)]),
      capacity_(safe_capacity(config)) {}

Action Mcts::choose(const GameState& root, PlayerId perspective) noexcept {
    ActionMask root_legal{};
    const int root_count = Game::legal_actions(root, root_legal);
    if (root_count <= 0) {
        return A_PASS;
    }

    float action_value[ACTION_SPACE_SIZE]{};
    std::uint32_t action_visits[ACTION_SPACE_SIZE]{};
    std::uint32_t action_seen[ACTION_SPACE_SIZE]{};

    const std::uint8_t dets = safe_determinizations(config_);
    const std::uint32_t total_sims = config_.sims_per_move == 0U ? 1U : config_.sims_per_move;
    const std::uint32_t base_sims = total_sims / dets;
    const std::uint32_t extra_sims = total_sims % dets;

    /*
     * Root sampling: each determinization gets its own tree, then root child
     * statistics are averaged. This keeps hidden-information worlds internally
     * consistent and avoids mixing incompatible private card orders below root.
     */
    for (std::uint8_t det = 0; det < dets; ++det) {
        GameState sampled = root;
        determinize(sampled, perspective, config_.rollout_seed + (0x9E37'79B9U * static_cast<std::uint64_t>(det + 1U)));
        reset(sampled, perspective);

        Xoshiro256pp rng = Xoshiro256pp::seeded(
            config_.rollout_seed ^ (0xD1B5'4A32'D192'ED03ULL * static_cast<std::uint64_t>(det + 1U)));
        const std::uint32_t sims = base_sims + (det < extra_sims ? 1U : 0U);
        run_simulations(sims == 0U ? 1U : sims, rng);

        const MctsNode& root_node = nodes_[0];
        for (std::uint32_t child_index = root_node.first_child; child_index != MCTS_NULL;
             child_index = nodes_[child_index].next_sibling) {
            const MctsNode& child = nodes_[child_index];
            const Action action = child.action_from_parent;
            if (action >= ACTION_SPACE_SIZE) {
                continue;
            }
            const std::uint32_t visits = child.visits;
            const float value = child_value_for_parent(root_node, child);
            action_value[action] += value * static_cast<float>(visits);
            action_visits[action] += visits;
            action_seen[action] += 1U;
        }
    }

    Action best = root_legal.nth_set(0U);
    std::uint32_t best_visits = 0;
    float best_value = -2.0F;
    for (int i = 0; i < root_count; ++i) {
        const Action action = root_legal.nth_set(static_cast<std::uint32_t>(i));
        const std::uint32_t visits = action_visits[action];
        const float value = visits == 0U
            ? -2.0F
            : action_value[action] / static_cast<float>(visits);
        if (visits > best_visits || (visits == best_visits && value > best_value)) {
            best = action;
            best_visits = visits;
            best_value = value;
        } else if (visits == 0U && best_visits == 0U && action_seen[action] > action_seen[best]) {
            best = action;
        }
    }
    return best;
}

void Mcts::reset(const GameState& root, PlayerId perspective) noexcept {
    node_count_ = 0;
    exhausted_ = false;
    root_perspective_ = perspective;
    const std::uint32_t root_index = allocate_node();
    assert(root_index == 0U);
    (void)root_index;
    states_[0] = root;
    MctsNode& root_node = nodes_[0];
    root_node.state_index = 0U;
    root_node.player = terminal_state(root) ? perspective : current_player(root);
    root_node.terminal = terminal_state(root);
}

void Mcts::run_simulations(std::uint32_t simulations, Xoshiro256pp& rng) noexcept {
    if (node_count_ == 0U) {
        return;
    }

    for (std::uint32_t sim = 0; sim < simulations; ++sim) {
        std::uint32_t path[MAX_EFFECT_DEPTH * 2U]{};
        std::uint8_t depth = 0;
        std::uint32_t node_index = 0U;
        path[depth++] = node_index;

        while (nodes_[node_index].expanded && nodes_[node_index].first_child != MCTS_NULL
               && !nodes_[node_index].terminal) {
            node_index = select_child(node_index);
            path[depth++] = node_index;
            if (depth >= static_cast<std::uint8_t>(MAX_EFFECT_DEPTH * 2U)) {
                break;
            }
        }

        MctsNode& leaf = nodes_[node_index];
        if (!leaf.terminal && !leaf.expanded) {
            const bool expanded = expand(node_index);
            if (expanded && leaf.first_child != MCTS_NULL) {
                node_index = select_child(node_index);
                if (depth < static_cast<std::uint8_t>(MAX_EFFECT_DEPTH * 2U)) {
                    path[depth++] = node_index;
                }
            }
        }

        GameState terminal = states_[nodes_[node_index].state_index];
        if (!terminal_state(terminal)) {
            rollout(terminal, rng);
        }
        backpropagate(path, depth, terminal);
    }
}

void Mcts::add_virtual_loss(std::uint32_t node, float amount) noexcept {
    if (node >= node_count_) {
        return;
    }
    nodes_[node].virtual_loss += amount;
}

void Mcts::revert_virtual_loss(std::uint32_t node, float amount) noexcept {
    if (node >= node_count_) {
        return;
    }
    nodes_[node].virtual_loss -= amount;
    if (nodes_[node].virtual_loss < 0.0F) {
        nodes_[node].virtual_loss = 0.0F;
    }
}

const MctsNode& Mcts::node(std::uint32_t index) const noexcept {
    assert(index < node_count_);
    return nodes_[index];
}

std::uint32_t Mcts::node_count() const noexcept {
    return node_count_;
}

bool Mcts::exhausted() const noexcept {
    return exhausted_;
}

Action Mcts::best_root_action() const noexcept {
    if (node_count_ == 0U) {
        return A_PASS;
    }

    const MctsNode& root = nodes_[0];
    Action best = A_PASS;
    std::uint32_t best_visits = 0;
    float best_value = -2.0F;
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        const float value = child_value_for_parent(root, child);
        if (child.visits > best_visits || (child.visits == best_visits && value > best_value)) {
            best = child.action_from_parent;
            best_visits = child.visits;
            best_value = value;
        }
    }
    return best;
}

std::uint32_t Mcts::root_visits_for(Action action) const noexcept {
    if (node_count_ == 0U) {
        return 0U;
    }
    for (std::uint32_t child_index = nodes_[0].first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        if (nodes_[child_index].action_from_parent == action) {
            return nodes_[child_index].visits;
        }
    }
    return 0U;
}

float Mcts::root_value_for(Action action) const noexcept {
    if (node_count_ == 0U) {
        return 0.0F;
    }
    const MctsNode& root = nodes_[0];
    for (std::uint32_t child_index = root.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        if (child.action_from_parent == action) {
            return child_value_for_parent(root, child);
        }
    }
    return 0.0F;
}

std::uint32_t Mcts::allocate_node() noexcept {
    if (node_count_ >= capacity_) {
        assert(false && "MCTS node slab exhausted");
        exhausted_ = true;
        return MCTS_NULL;
    }
    const std::uint32_t index = node_count_;
    nodes_[index] = MctsNode{};
    states_[index] = GameState{};
    ++node_count_;
    return index;
}

bool Mcts::expand(std::uint32_t node_index) noexcept {
    if (node_index >= node_count_) {
        return false;
    }

    MctsNode& node = nodes_[node_index];
    if (node.terminal) {
        node.expanded = true;
        return true;
    }

    ActionMask legal{};
    const int legal_count = Game::legal_actions(states_[node.state_index], legal);
    if (legal_count <= 0) {
        node.terminal = true;
        node.expanded = true;
        return true;
    }

    const bool use_heuristic_prior = config_.prior_fn == nullptr
        && config_.rollout_policy == MctsRolloutPolicy::Heuristic
        && legal_count > 1;
    Action heuristic_prior_action = A_PASS;
    if (use_heuristic_prior) {
        Xoshiro256pp prior_rng = Xoshiro256pp::seeded(0xC0FF'EE01ULL);
        heuristic_prior_action = rollout_heuristic_action(
            states_[node.state_index],
            legal,
            legal_count,
            prior_rng,
            0.0F);
    }

    float prior_sum = 0.0F;
    for (int i = 0; i < legal_count; ++i) {
        const Action action = legal.nth_set(static_cast<std::uint32_t>(i));
        if (use_heuristic_prior) {
            prior_sum += action == heuristic_prior_action
                ? 0.80F
                : 0.20F / static_cast<float>(legal_count - 1);
        } else {
            prior_sum += positive_prior(config_, states_[node.state_index], node.player, action);
        }
    }
    if (prior_sum <= 0.0F) {
        prior_sum = static_cast<float>(legal_count);
    }

    std::uint32_t previous_child = MCTS_NULL;
    for (int i = 0; i < legal_count; ++i) {
        const Action action = legal.nth_set(static_cast<std::uint32_t>(i));
        const std::uint32_t child_index = allocate_node();
        if (child_index == MCTS_NULL) {
            break;
        }

        GameState child_state = states_[node.state_index];
        const bool done = Game::step(child_state, action);
        states_[child_index] = child_state;

        MctsNode& child = nodes_[child_index];
        child.parent = node_index;
        child.state_index = child_index;
        child.action_from_parent = action;
        child.terminal = done || terminal_state(child_state);
        child.player = next_player_for_child(child_state, node.player);
        const float prior = use_heuristic_prior
            ? (action == heuristic_prior_action
                ? 0.80F
                : 0.20F / static_cast<float>(legal_count - 1))
            : positive_prior(config_, states_[node.state_index], node.player, action);
        child.prior = prior > 0.0F ? prior / prior_sum : 1.0F / static_cast<float>(legal_count);

        if (previous_child == MCTS_NULL) {
            node.first_child = child_index;
        } else {
            nodes_[previous_child].next_sibling = child_index;
        }
        previous_child = child_index;
    }

    node.expanded = true;
    return node.first_child != MCTS_NULL;
}

std::uint32_t Mcts::select_child(std::uint32_t node_index) const noexcept {
    const MctsNode& node = nodes_[node_index];
    std::uint32_t best_child = node.first_child;
    float best_score = -1.0e30F;
    const float parent_visits = static_cast<float>(node.visits + 1U);
    const float exploration_base = std::sqrt(parent_visits);

    for (std::uint32_t child_index = node.first_child; child_index != MCTS_NULL;
         child_index = nodes_[child_index].next_sibling) {
        const MctsNode& child = nodes_[child_index];
        const float q = child_value_for_parent(node, child);
        const float denom = 1.0F + static_cast<float>(child.visits) + child.virtual_loss;
        const float u = config_.c_puct * child.prior * exploration_base / denom;
        const float score_value = q + u - child.virtual_loss;
        if (score_value > best_score) {
            best_score = score_value;
            best_child = child_index;
        }
    }
    return best_child;
}

void Mcts::rollout(GameState& state, Xoshiro256pp& rng) const noexcept {
    std::uint16_t guard = 0;
    while (!terminal_state(state) && guard < 4096U) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(state, legal);
        if (legal_count <= 0) {
            break;
        }
        const Action action = config_.rollout_policy == MctsRolloutPolicy::Random
            ? rollout_random_action(legal, legal_count, rng)
            : rollout_heuristic_action(state, legal, legal_count, rng, config_.rollout_epsilon);
        (void)Game::step(state, action);
        ++guard;
    }
}

void Mcts::backpropagate(const std::uint32_t* path, std::uint8_t depth, const GameState& terminal) noexcept {
    for (std::uint8_t i = 0; i < depth; ++i) {
        MctsNode& path_node = nodes_[path[i]];
        ++path_node.visits;
        path_node.value_sum += terminal_value_for(path_node.player, terminal);
    }
}

float Mcts::terminal_value_for(PlayerId player, const GameState& terminal) const noexcept {
    return mcts_terminal_value(terminal, player);
}

float Mcts::child_value_for_parent(const MctsNode& parent, const MctsNode& child) const noexcept {
    if (child.visits == 0U) {
        return 0.0F;
    }
    float value = child.value_sum / static_cast<float>(child.visits);
    if (parent.player != child.player) {
        value = -value;
    }
    return value;
}

PlayerId Mcts::next_player_for_child(const GameState& child_state, PlayerId parent_player) const noexcept {
    if (terminal_state(child_state)) {
        return other_player(parent_player);
    }
    const PlayerId player = current_player(child_state);
    return player < child_state.num_players ? player : root_perspective_;
}

float mcts_terminal_value(const GameState& state, PlayerId player) noexcept {
    if (state.num_players != 2U || player >= state.num_players) {
        return 0.0F;
    }
    const PlayerId opponent = other_player(player);
    const std::int16_t player_score = score(state, player);
    const std::int16_t opponent_score = score(state, opponent);
    if (player_score > opponent_score) {
        return 1.0F;
    }
    if (player_score < opponent_score) {
        return -1.0F;
    }
    return 0.0F;
}

Action mcts_choose(const GameState& state, PlayerId perspective, const MctsConfig& config) noexcept {
    Mcts search(config);
    return search.choose(state, perspective);
}
