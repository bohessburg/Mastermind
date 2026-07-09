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
        if (value > best_value || (value == best_value && visits > best_visits)) {
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
        if (value > best_value || (value == best_value && child.visits > best_visits)) {
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

    float prior_sum = 0.0F;
    for (int i = 0; i < legal_count; ++i) {
        const Action action = legal.nth_set(static_cast<std::uint32_t>(i));
        prior_sum += positive_prior(config_, states_[node.state_index], node.player, action);
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
        const float prior = positive_prior(config_, states_[node.state_index], node.player, action);
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
        const std::uint32_t pick = rng.uniform(static_cast<std::uint32_t>(legal_count));
        const Action action = legal.nth_set(pick);
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
