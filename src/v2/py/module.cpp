#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/determinize.h"
#include "v2/core/game.h"
#include "v2/core/interp.h"
#include "v2/core/score.h"
#include "v2/encode/encoder.h"
#include "v2/mcts/eval_runner.h"
#include "v2/mcts/selfplay.h"
#include "v2/mcts/tree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

[[nodiscard]] DefId parse_def(py::handle object);

struct PySetup {
    Setup setup{};

    PySetup(int players, const py::object& kingdom, bool colony) {
        if (players < 2 || players > MAX_PLAYERS) {
            throw std::invalid_argument("players must be between 2 and MAX_PLAYERS");
        }
        setup = Setup{};
        setup.num_players = static_cast<PlayerId>(players);
        setup.use_colony_platinum = colony;
        fill_kingdom(kingdom);
    }

    void fill_kingdom(const py::object& kingdom) {
        if (kingdom.is_none()) {
            return;
        }
        if (py::isinstance<py::str>(kingdom)) {
            throw std::invalid_argument("kingdom must be a sequence");
        }

        const py::sequence sequence = py::reinterpret_borrow<py::sequence>(kingdom);
        const std::size_t size = py::len(sequence);
        if (size > MAX_KINGDOM_DEFS) {
            throw std::invalid_argument("kingdom has too many cards");
        }

        setup.kingdom_count = static_cast<std::uint8_t>(size);
        for (std::size_t i = 0; i < size; ++i) {
            setup.kingdom[static_cast<std::uint8_t>(i)] = parse_def(sequence[i]);
        }
    }
};

struct PyGame {
    GameState state{};
};

[[nodiscard]] bool valid_player(const GameState& state, int player) noexcept {
    return player >= 0 && player < static_cast<int>(state.num_players);
}

[[nodiscard]] ObsVersion parse_obs_version(int version) {
    if (version == static_cast<int>(ObsVersion::V1)) {
        return ObsVersion::V1;
    }
    if (version == static_cast<int>(ObsVersion::V2)) {
        return ObsVersion::V2;
    }
    throw std::invalid_argument("obs_version must be 1 or 2");
}

[[nodiscard]] int obs_version_value(ObsVersion version) noexcept {
    return static_cast<int>(version);
}

[[nodiscard]] PlayerId turn_player_id(const GameState& state) noexcept {
    if (state.turn_queue.size == 0U) {
        return 0U;
    }
    return state.turn_queue.entries[state.turn_queue.head].player;
}

[[nodiscard]] PlayerId decision_player_id(const GameState& state) noexcept {
    if (state.decision.player < state.num_players) {
        return state.decision.player;
    }
    return turn_player_id(state);
}

[[nodiscard]] int action_mask_count(const ActionMask& legal) noexcept {
    int count = 0;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        count += legal.test(action) ? 1 : 0;
    }
    return count;
}

[[nodiscard]] int pile_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? static_cast<int>(pile.mixed_len) : static_cast<int>(pile.count);
}

[[nodiscard]] Slot pile_top_slot(const Pile& pile) noexcept {
    if (pile.mixed_len > 0U) {
        return pile.mixed[pile.mixed_len - 1U];
    }
    return pile.base;
}

[[nodiscard]] std::int16_t vp_tokens(const PlayerState& player) noexcept {
    return static_cast<std::int16_t>(
        static_cast<std::uint16_t>(player.vp_tokens_lo)
        | (static_cast<std::uint16_t>(player.vp_tokens_hi) << 8U));
}

[[nodiscard]] bool def_from_name(const std::string& name, DefId& out) noexcept {
    const std::uint16_t count = card_def_count();
    for (DefId def = 0; def < count; ++def) {
        if (std::strcmp(card_def(def).name, name.c_str()) == 0) {
            out = def;
            return true;
        }
    }
    return false;
}

[[nodiscard]] DefId parse_def(const py::handle object) {
    if (py::isinstance<py::str>(object)) {
        const std::string name = py::cast<std::string>(object);
        DefId def = 0;
        if (!def_from_name(name, def)) {
            throw std::invalid_argument("unknown card name: " + name);
        }
        return def;
    }

    const auto value = py::cast<std::uint64_t>(object);
    if (value >= card_def_count()) {
        throw std::invalid_argument("def id out of range");
    }
    return static_cast<DefId>(value);
}

[[nodiscard]] py::dict decision_dict(const PendingDecision& decision) {
    py::dict dict;
    dict["player"] = decision.player;
    dict["kind"] = decision.kind;
    dict["source"] = decision.source;
    dict["min"] = decision.min_left;
    dict["max"] = decision.max_left;
    return dict;
}

[[nodiscard]] py::dict empty_decision_context() {
    py::dict dict;
    dict["source_def"] = py::none();
    dict["subject_defs"] = py::list();
    dict["subject_index"] = py::none();
    return dict;
}

void append_subject_def(py::list& subjects, const GameState& state, std::int16_t slot_value) {
    if (slot_value < 0 || slot_value >= static_cast<std::int16_t>(state.num_slots)) {
        return;
    }
    subjects.append(py::int_(state.slot_to_def[static_cast<Slot>(slot_value)]));
}

[[nodiscard]] py::object bandit_attacker_object(const GameState& state, const EffectFrame& attack_frame) {
    for (std::uint8_t depth = state.effect_depth; depth > 0U; --depth) {
        const EffectFrame& frame = state.effect_stack[depth - 1U];
        if (frame.source == DEF_BANDIT
            && (frame.flags & FRAME_ATTACK) == 0U
            && frame.player != attack_frame.player) {
            return py::int_(frame.player);
        }
    }
    return py::none();
}

[[nodiscard]] py::dict decision_context_dict(const GameState& state) {
    if (state.decision.kind == static_cast<std::uint8_t>(DecisionKind::None)
        || state.effect_depth == 0U) {
        return empty_decision_context();
    }

    const EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
    py::dict dict = empty_decision_context();
    dict["source_def"] = py::int_(frame.source);

    py::list subjects;
    py::object subject_index = py::none();
    const auto kind = static_cast<DecisionKind>(state.decision.kind);

    if (frame.source == DEF_LIBRARY && kind == DecisionKind::ChooseOption) {
        // Library custom frame layout: data[1] is the currently drawn Action
        // slot being kept or set aside.
        append_subject_def(subjects, state, frame.data[1]);
        if (py::len(subjects) > 0) {
            subject_index = py::int_(0);
        }
    } else if (frame.source == DEF_SENTRY) {
        // Sentry custom frame layout:
        //   data[1], data[2]: looked-at set-aside slots in reveal order
        //   data[3]: looked-at count
        //   data[4]: current looked-at index for ChooseOption
        const std::uint8_t count = frame.data[3] <= 0
            ? 0U
            : static_cast<std::uint8_t>(frame.data[3]);
        if (kind == DecisionKind::ChooseOrder) {
            const PlayerState& player = state.players[frame.player];
            for (std::uint8_t i = 0; i < player.set_aside.size; ++i) {
                append_subject_def(subjects, state, player.set_aside.cards[i]);
            }
        } else {
            for (std::uint8_t i = 0; i < count && i < 2U; ++i) {
                append_subject_def(subjects, state, frame.data[1 + i]);
            }
            if (kind == DecisionKind::ChooseOption
                && frame.data[4] >= 0
                && frame.data[4] < static_cast<std::int16_t>(py::len(subjects))) {
                subject_index = py::int_(frame.data[4]);
            }
        }
    } else if (frame.source == DEF_VASSAL && kind == DecisionKind::ChooseOption) {
        // Vassal uses the shared DSL result slots. DiscardDeckTop stores the
        // discarded def id in data[4] so the later yes/no option can play it.
        const std::int16_t def_value = frame.data[4];
        if (def_value >= 0 && def_value < static_cast<std::int16_t>(card_def_count())) {
            subjects.append(py::int_(def_value));
            subject_index = py::int_(0);
        }
    } else if (frame.source == DEF_BANDIT && kind == DecisionKind::Choose) {
        // Bandit attack frame overlay mirrors interp.cpp:
        //   data[1], data[2]: revealed set-aside slots
        //   data[3]: revealed count
        // The pending player is the victim. The attacker is the nearest
        // non-attack Bandit frame below this attack frame, if still present.
        const std::uint8_t count = frame.data[3] <= 0
            ? 0U
            : static_cast<std::uint8_t>(frame.data[3]);
        for (std::uint8_t i = 0; i < count && i < 2U; ++i) {
            append_subject_def(subjects, state, frame.data[1 + i]);
        }
        dict["victim_player"] = py::int_(frame.player);
        dict["attacker_player"] = bandit_attacker_object(state, frame);
    }

    dict["subject_defs"] = subjects;
    dict["subject_index"] = subject_index;
    return dict;
}

[[nodiscard]] PyGame make_game(const GameState& state) noexcept {
    PyGame game{};
    game.state = state;
    return game;
}

[[nodiscard]] PyGame py_new_game(const PySetup& setup, std::uint64_t seed) {
    GameState state{};
    {
        py::gil_scoped_release release;
        state = Game::new_game(setup.setup, seed);
    }
    return make_game(state);
}

[[nodiscard]] py::array_t<bool> legal_mask_array(const GameState& state) {
    py::array_t<bool> array(static_cast<py::ssize_t>(ACTION_SPACE_SIZE));
    bool* data = array.mutable_data();

    ActionMask legal{};
    {
        py::gil_scoped_release release;
        (void)Game::legal_actions(state, legal);
    }

    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        data[action] = legal.test(action);
    }
    return array;
}

[[nodiscard]] py::array_t<float> encode_array(
    const GameState& state,
    PlayerId player,
    ObsVersion version) {
    py::array_t<float> array(static_cast<py::ssize_t>(obs_size_for(version)));
    float* data = array.mutable_data();
    {
        py::gil_scoped_release release;
        encode(state, player, data, version);
    }
    return array;
}

[[nodiscard]] py::object winner_object(const GameState& state) {
    if (state.phase != static_cast<std::uint8_t>(Phase::Over)) {
        return py::none();
    }

    PlayerId winner = 0;
    bool tied = false;
    for (PlayerId player = 1; player < state.num_players; ++player) {
        const std::int16_t player_score = score(state, player);
        const std::int16_t winner_score = score(state, winner);
        if (player_score > winner_score) {
            winner = player;
            tied = false;
        } else if (player_score == winner_score) {
            tied = true;
        }
    }

    if (tied) {
        return py::none();
    }
    return py::int_(winner);
}

[[nodiscard]] py::object discard_top_object(const GameState& state, PlayerId player) {
    const OrderedZone& discard = state.players[player].discard;
    if (discard.size == 0U) {
        return py::none();
    }
    const Slot slot = discard.cards[discard.size - 1U];
    return py::int_(slot < state.num_slots ? state.slot_to_def[slot] : 0U);
}

[[nodiscard]] float terminal_reward(const GameState& state, PlayerId perspective) noexcept {
    if (perspective >= state.num_players) {
        return 0.0F;
    }

    const std::int16_t perspective_score = score(state, perspective);
    std::int16_t best_score = perspective_score;
    PlayerId best_player = perspective;
    bool tied_best = false;
    for (PlayerId player = 0; player < state.num_players; ++player) {
        const std::int16_t value = score(state, player);
        if (value > best_score) {
            best_score = value;
            best_player = player;
            tied_best = false;
        } else if (player != best_player && value == best_score) {
            tied_best = true;
        }
    }

    if (perspective_score == best_score && tied_best) {
        return 0.0F;
    }
    return perspective_score == best_score ? 1.0F : -1.0F;
}

[[nodiscard]] std::uint64_t fnv1a_state(const GameState& state) noexcept {
    constexpr std::uint64_t kOffset = 14695981039346656037ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&state);
    std::uint64_t hash = kOffset;
    for (std::size_t i = 0; i < sizeof(GameState); ++i) {
        hash ^= bytes[i];
        hash *= kPrime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t batch_seed(
    std::uint64_t seed_base,
    std::uint64_t generation,
    std::uint64_t index) noexcept {
    return seed_base
        + (generation * 0x9E37'79B9'7F4A'7C15ULL)
        + (index * 0xD1B5'4A32'D192'ED03ULL);
}

[[nodiscard]] std::vector<Setup> parse_batch_setups(const py::object& object) {
    std::vector<Setup> setups;
    if (py::isinstance<PySetup>(object)) {
        setups.push_back(py::cast<const PySetup&>(object).setup);
        return setups;
    }

    if (!object.is_none() && !py::isinstance<py::str>(object)) {
        const py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
        const std::size_t size = py::len(sequence);
        if (size > 0 && py::isinstance<PySetup>(sequence[0])) {
            setups.reserve(size);
            for (std::size_t i = 0; i < size; ++i) {
                if (!py::isinstance<PySetup>(sequence[i])) {
                    throw std::invalid_argument("setup sequence must contain only Setup objects");
                }
                setups.push_back(py::cast<const PySetup&>(sequence[i]).setup);
            }
            return setups;
        }
    }

    PySetup setup(2, object, false);
    setups.push_back(setup.setup);
    return setups;
}

struct PyBatchRunner {
    std::vector<Setup> setups;
    std::vector<GameState> states;
    std::vector<std::uint64_t> generations;
    std::uint64_t seed_base = 0;
    std::uint64_t completed = 0;

    PyBatchRunner(const py::object& setups_or_kingdom, std::size_t n_games, std::uint64_t seed)
        : setups(parse_batch_setups(setups_or_kingdom)),
          states(n_games),
          generations(n_games),
          seed_base(seed) {
        if (n_games == 0U) {
            throw std::invalid_argument("n_games must be positive");
        }
        reset();
    }

    void reset() {
        completed = 0;
        for (std::size_t i = 0; i < states.size(); ++i) {
            generations[i] = 0;
        }
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < states.size(); ++i) {
                states[i] = Game::new_game(setup_for(i), batch_seed(seed_base, generations[i], i));
            }
        }
    }

    [[nodiscard]] py::array_t<float> observations() const {
        py::array_t<float> array({
            static_cast<py::ssize_t>(states.size()),
            static_cast<py::ssize_t>(OBS_SIZE),
        });
        float* data = array.mutable_data();
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < states.size(); ++i) {
                encode(states[i], decision_player_id(states[i]), data + (i * OBS_SIZE));
            }
        }
        return array;
    }

    [[nodiscard]] py::array_t<bool> legal_masks() const {
        py::array_t<bool> array({
            static_cast<py::ssize_t>(states.size()),
            static_cast<py::ssize_t>(ACTION_SPACE_SIZE),
        });
        bool* data = array.mutable_data();
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < states.size(); ++i) {
                ActionMask legal{};
                (void)Game::legal_actions(states[i], legal);
                bool* row = data + (i * ACTION_SPACE_SIZE);
                for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
                    row[action] = legal.test(action);
                }
            }
        }
        return array;
    }

    [[nodiscard]] py::array_t<std::int32_t> current_players() const {
        py::array_t<std::int32_t> array(static_cast<py::ssize_t>(states.size()));
        std::int32_t* data = array.mutable_data();
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < states.size(); ++i) {
                data[i] = static_cast<std::int32_t>(decision_player_id(states[i]));
            }
        }
        return array;
    }

    [[nodiscard]] py::tuple step(py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> actions) {
        if (actions.ndim() != 1 || actions.shape(0) != static_cast<py::ssize_t>(states.size())) {
            throw std::invalid_argument("actions must have shape [N]");
        }

        const std::int32_t* action_data = actions.data();
        for (std::size_t i = 0; i < states.size(); ++i) {
            if (action_data[i] < 0 || action_data[i] >= static_cast<std::int32_t>(ACTION_SPACE_SIZE)) {
                throw std::invalid_argument("action out of range");
            }
            ActionMask legal{};
            const int legal_count = Game::legal_actions(states[i], legal);
            const Action action = static_cast<Action>(action_data[i]);
            if (legal_count <= 0 || !legal.test(action)) {
                throw std::invalid_argument("illegal action");
            }
        }

        py::array_t<bool> dones(static_cast<py::ssize_t>(states.size()));
        py::array_t<float> rewards(static_cast<py::ssize_t>(states.size()));
        bool* done_data = dones.mutable_data();
        float* reward_data = rewards.mutable_data();

        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < states.size(); ++i) {
                GameState& state = states[i];
                const PlayerId perspective = decision_player_id(state);
                const bool done = Game::step(state, static_cast<Action>(action_data[i]));
                done_data[i] = done;
                reward_data[i] = done ? terminal_reward(state, perspective) : 0.0F;
                if (done) {
                    ++completed;
                    ++generations[i];
                    state = Game::new_game(setup_for(i), batch_seed(seed_base, generations[i], i));
                }
            }
        }

        return py::make_tuple(dones, rewards);
    }

    [[nodiscard]] std::uint64_t games_completed() const noexcept {
        return completed;
    }

private:
    [[nodiscard]] const Setup& setup_for(std::size_t index) const noexcept {
        return setups[index % setups.size()];
    }
};

/*
 * A one-decision, externally evaluated information-set MCTS.  It lives in
 * the binding layer so callers that need an interactive decision (the web
 * server and evaluation helpers) use exactly the same leaf parking/resume
 * mechanism as SelfPlayRunner without changing the core Mcts behavior.
 */
struct PyDecisionSearcher {
    struct World {
        explicit World(const MctsConfig& config) : mcts(config) {}

        Mcts mcts;
        MctsPendingLeaf pending_leaf{};
        std::uint32_t sims_target = 0;
        std::uint32_t sims_started = 0;
        std::uint32_t sims_completed = 0;
        bool waiting_for_evaluation = false;
    };

    PyDecisionSearcher(const PyGame& game, int perspective_player, const py::dict& config)
        : root_(game.state) {
        if (!valid_player(root_, perspective_player)) {
            throw std::invalid_argument("perspective_player is invalid for this game");
        }
        if (!config.contains("sims") || !config.contains("c_puct")
            || !config.contains("determinizations") || !config.contains("seed")) {
            throw std::invalid_argument(
                "DecisionSearcher config requires sims, c_puct, determinizations, and seed");
        }

        const std::uint32_t sims = py::cast<std::uint32_t>(config["sims"]);
        const float c_puct = py::cast<float>(config["c_puct"]);
        const std::uint32_t determinizations = py::cast<std::uint32_t>(config["determinizations"]);
        const std::uint64_t seed = py::cast<std::uint64_t>(config["seed"]);
        const ObsVersion obs_version = config.contains("obs_version")
            ? parse_obs_version(py::cast<int>(config["obs_version"]))
            : ObsVersion::V1;
        const bool auto_play_treasures = config.contains("auto_play_treasures")
            && py::cast<bool>(config["auto_play_treasures"]);
        const bool prune_treasure_plays = config.contains("prune_treasure_plays")
            && py::cast<bool>(config["prune_treasure_plays"]);
        if (sims == 0U) {
            throw std::invalid_argument("DecisionSearcher config sims must be positive");
        }
        if (c_puct < 0.0F) {
            throw std::invalid_argument("DecisionSearcher config c_puct must be non-negative");
        }
        if (determinizations == 0U || determinizations > 255U) {
            throw std::invalid_argument("DecisionSearcher config determinizations must be between 1 and 255");
        }

        perspective_ = static_cast<PlayerId>(perspective_player);
        obs_version_ = obs_version;
        root_legal_count_ = Game::legal_actions(root_, root_legal_);
        forced_action_ = auto_play_treasures
            ? mcts_canonical_treasure_play(mcts_filter_treasure_plays(root_, root_legal_))
            : A_PASS;
        if (forced_action_ != A_PASS) {
            return;
        }
        if (prune_treasure_plays) {
            root_legal_ = mcts_filter_treasure_plays(root_, root_legal_);
            root_legal_count_ = action_mask_count(root_legal_);
        }

        MctsConfig mcts_config{};
        mcts_config.sims_per_move = sims;
        mcts_config.c_puct = c_puct;
        mcts_config.determinizations = 1U;
        mcts_config.max_tree_nodes = 4096U;
        mcts_config.rollout_policy = MctsRolloutPolicy::External;
        mcts_config.rollout_seed = seed;
        mcts_config.prune_treasure_plays = prune_treasure_plays;

        const std::uint32_t base_sims = sims / determinizations;
        const std::uint32_t extra_sims = sims % determinizations;
        worlds_.reserve(determinizations);
        for (std::uint32_t det = 0; det < determinizations; ++det) {
            auto world = std::make_unique<World>(mcts_config);
            GameState sampled = root_;
            determinize(
                sampled,
                perspective_,
                seed + (0x9E37'79B9ULL * static_cast<std::uint64_t>(det + 1U)));
            world->mcts.reset(sampled, perspective_);
            const std::uint32_t allocated = base_sims + (det < extra_sims ? 1U : 0U);
            // Keep the same low-simulation behavior as mcts_choose: every
            // determinization gets at least one simulation.
            world->sims_target = allocated == 0U ? 1U : allocated;
            worlds_.push_back(std::move(world));
        }
    }

    [[nodiscard]] std::uint32_t collect() noexcept {
        if (!pending_worlds_.empty() || done()) {
            return static_cast<std::uint32_t>(pending_worlds_.size());
        }

        // A terminal leaf does not need a network round trip.  Keep advancing
        // until we have leaves for Python or every world has its quota.
        while (pending_worlds_.empty() && !done()) {
            bool progressed = false;
            for (std::uint32_t index = 0; index < worlds_.size(); ++index) {
                World& world = *worlds_[index];
                if (world.waiting_for_evaluation || world.sims_started >= world.sims_target) {
                    continue;
                }
                MctsPendingLeaf leaf{};
                const bool need_evaluation = world.mcts.collect_external_leaf(leaf);
                ++world.sims_started;
                progressed = true;
                if (need_evaluation) {
                    world.pending_leaf = leaf;
                    world.waiting_for_evaluation = true;
                    pending_worlds_.push_back(index);
                } else {
                    ++world.sims_completed;
                }
            }
            if (!progressed) {
                break;
            }
        }
        return static_cast<std::uint32_t>(pending_worlds_.size());
    }

    void provide(
        const float* values,
        const float* policies,
        std::uint32_t count) noexcept {
        if (count != pending_worlds_.size()) {
            return;
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            World& world = *worlds_[pending_worlds_[i]];
            float normalized[ACTION_SPACE_SIZE]{};
            normalize_policy(
                policies == nullptr ? nullptr : policies + (static_cast<std::size_t>(i) * ACTION_SPACE_SIZE),
                world.pending_leaf.legal,
                world.pending_leaf.legal_count,
                normalized);
            world.mcts.provide_external_evaluation(
                world.pending_leaf,
                values == nullptr ? 0.0F : values[i],
                normalized);
            world.waiting_for_evaluation = false;
            ++world.sims_completed;
        }
        pending_worlds_.clear();
    }

    [[nodiscard]] bool done() const noexcept {
        if (forced_action_ != A_PASS) {
            return true;
        }
        if (!pending_worlds_.empty()) {
            return false;
        }
        for (const auto& world : worlds_) {
            if (world->waiting_for_evaluation || world->sims_completed < world->sims_target) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] Action best_action() const noexcept {
        if (!done()) {
            return A_PASS;
        }
        if (forced_action_ != A_PASS) {
            return forced_action_;
        }

        const int root_count = root_legal_count_;
        if (root_count <= 0) {
            return A_PASS;
        }

        float action_value[ACTION_SPACE_SIZE]{};
        std::uint32_t action_visits[ACTION_SPACE_SIZE]{};
        std::uint32_t action_seen[ACTION_SPACE_SIZE]{};
        for (const auto& world : worlds_) {
            for (int i = 0; i < root_count; ++i) {
                const Action action = root_legal_.nth_set(static_cast<std::uint32_t>(i));
                const std::uint32_t visits = world->mcts.root_visits_for(action);
                action_value[action] += world->mcts.root_value_for(action) * static_cast<float>(visits);
                action_visits[action] += visits;
                // After the initial external evaluation Mcts expands every
                // legal root action, matching mcts_choose's action_seen tie
                // handling.  A terminal root has no legal actions above.
                if (world->sims_completed != 0U) {
                    ++action_seen[action];
                }
            }
        }

        Action best = root_legal_.nth_set(0U);
        std::uint32_t best_visits = 0U;
        float best_value = -2.0F;
        for (int i = 0; i < root_count; ++i) {
            const Action action = root_legal_.nth_set(static_cast<std::uint32_t>(i));
            const std::uint32_t visits = action_visits[action];
            const float value = visits == 0U ? -2.0F : action_value[action] / static_cast<float>(visits);
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

    [[nodiscard]] const MctsPendingLeaf& pending_leaf(std::uint32_t index) const noexcept {
        return worlds_[pending_worlds_[index]]->pending_leaf;
    }

    [[nodiscard]] const GameState& pending_state(std::uint32_t index) const noexcept {
        const World& world = *worlds_[pending_worlds_[index]];
        return world.mcts.state_for(world.pending_leaf.state_index);
    }

    [[nodiscard]] ObsVersion obs_version() const noexcept {
        return obs_version_;
    }

    [[nodiscard]] std::size_t observation_size() const noexcept {
        return obs_size_for(obs_version_);
    }

private:
    static void normalize_policy(
        const float* logits,
        const ActionMask& legal,
        int legal_count,
        float* out) noexcept {
        if (legal_count <= 0) {
            out[A_PASS] = 1.0F;
            return;
        }
        float max_logit = -3.4e38F;
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (legal.test(action)) {
                const float value = logits == nullptr ? 0.0F : logits[action];
                if (value > max_logit) {
                    max_logit = value;
                }
            }
        }
        double sum = 0.0;
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            if (!legal.test(action)) {
                continue;
            }
            const float value = logits == nullptr ? 0.0F : logits[action];
            const double weight = std::exp(static_cast<double>(value - max_logit));
            out[action] = static_cast<float>(weight);
            sum += weight;
        }
        if (sum <= 0.0) {
            const float uniform = 1.0F / static_cast<float>(legal_count);
            for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
                out[action] = legal.test(action) ? uniform : 0.0F;
            }
            return;
        }
        const float inverse_sum = static_cast<float>(1.0 / sum);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            out[action] *= inverse_sum;
        }
    }

    GameState root_{};
    PlayerId perspective_ = 0U;
    ObsVersion obs_version_ = ObsVersion::V1;
    ActionMask root_legal_{};
    int root_legal_count_ = 0;
    Action forced_action_ = A_PASS;
    std::vector<std::unique_ptr<World>> worlds_;
    std::vector<std::uint32_t> pending_worlds_;
};

[[nodiscard]] py::tuple decision_search_collect(PyDecisionSearcher& searcher) {
    std::uint32_t count = 0U;
    {
        py::gil_scoped_release release;
        count = searcher.collect();
    }
    py::array_t<float> obs({
        static_cast<py::ssize_t>(count),
        static_cast<py::ssize_t>(searcher.observation_size()),
    });
    py::array_t<bool> masks({
        static_cast<py::ssize_t>(count),
        static_cast<py::ssize_t>(ACTION_SPACE_SIZE),
    });
    for (std::uint32_t i = 0; i < count; ++i) {
        const MctsPendingLeaf& leaf = searcher.pending_leaf(i);
        encode(searcher.pending_state(i), leaf.player, obs.mutable_data(i, 0), searcher.obs_version());
        bool* row = masks.mutable_data(i, 0);
        for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
            row[action] = leaf.legal.test(action);
        }
    }
    return py::make_tuple(obs, masks);
}

void decision_search_provide(
    PyDecisionSearcher& searcher,
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    py::array_t<float, py::array::c_style | py::array::forcecast> policies) {
    if (values.ndim() != 1) {
        throw std::invalid_argument("values must have shape [B]");
    }
    if (policies.ndim() != 2 || policies.shape(1) != static_cast<py::ssize_t>(ACTION_SPACE_SIZE)) {
        throw std::invalid_argument("policies must have shape [B, ACTION_SPACE]");
    }
    if (policies.shape(0) != values.shape(0)) {
        throw std::invalid_argument("values and policies batch sizes differ");
    }
    const auto count = static_cast<std::uint32_t>(values.shape(0));
    if (count != searcher.collect()) {
        throw std::invalid_argument("evaluation batch size does not match pending leaves");
    }
    {
        py::gil_scoped_release release;
        searcher.provide(values.data(), policies.data(), count);
    }
}

[[nodiscard]] py::tuple selfplay_collect(SelfPlayRunner& runner, std::uint32_t max_batch) {
    std::uint32_t count = 0;
    {
        py::gil_scoped_release release;
        count = runner.collect_leaves(max_batch);
    }

    py::array_t<float> obs({
        static_cast<py::ssize_t>(count),
        static_cast<py::ssize_t>(runner.observation_size()),
    });
    py::array_t<bool> masks({
        static_cast<py::ssize_t>(count),
        static_cast<py::ssize_t>(ACTION_SPACE_SIZE),
    });
    if (count != 0U) {
        std::memcpy(
            obs.mutable_data(),
            runner.leaf_observations(),
            static_cast<std::size_t>(count) * runner.observation_size() * sizeof(float));
        std::memcpy(
            masks.mutable_data(),
            runner.leaf_legal_masks(),
            static_cast<std::size_t>(count) * ACTION_SPACE_SIZE * sizeof(bool));
    }
    return py::make_tuple(obs, masks);
}

[[nodiscard]] py::array_t<std::uint8_t> selfplay_leaf_players(const SelfPlayRunner& runner) {
    const auto count = runner.leaf_count();
    py::array_t<std::uint8_t> players(static_cast<py::ssize_t>(count));
    if (count != 0U) {
        std::memcpy(
            players.mutable_data(),
            runner.leaf_players(),
            static_cast<std::size_t>(count) * sizeof(PlayerId));
    }
    return players;
}

void selfplay_provide(
    SelfPlayRunner& runner,
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    py::array_t<float, py::array::c_style | py::array::forcecast> policies) {
    if (values.ndim() != 1) {
        throw std::invalid_argument("values must have shape [B]");
    }
    if (policies.ndim() != 2 || policies.shape(1) != static_cast<py::ssize_t>(ACTION_SPACE_SIZE)) {
        throw std::invalid_argument("policies must have shape [B, ACTION_SPACE]");
    }
    if (policies.shape(0) != values.shape(0)) {
        throw std::invalid_argument("values and policies batch sizes differ");
    }
    const auto count = static_cast<std::uint32_t>(values.shape(0));
    {
        py::gil_scoped_release release;
        runner.provide_evaluations(values.data(), policies.data(), count);
    }
}

[[nodiscard]] py::list selfplay_finished(SelfPlayRunner& runner) {
    std::vector<SelfPlayRecord> records = runner.take_finished_games();
    const std::size_t obs_size = runner.observation_size();
    py::list out;
    for (const SelfPlayRecord& record : records) {
        py::dict dict;
        py::array_t<float> obs({
            static_cast<py::ssize_t>(record.moves),
            static_cast<py::ssize_t>(obs_size),
        });
        py::array_t<float> policies({
            static_cast<py::ssize_t>(record.moves),
            static_cast<py::ssize_t>(ACTION_SPACE_SIZE),
        });
        py::array_t<float> values(static_cast<py::ssize_t>(record.moves));
        py::array_t<std::uint8_t> players(static_cast<py::ssize_t>(record.moves));
        if (record.moves != 0U) {
            std::memcpy(
                obs.mutable_data(),
                record.observations.data(),
                static_cast<std::size_t>(record.moves) * obs_size * sizeof(float));
            std::memcpy(
                policies.mutable_data(),
                record.policy_targets.data(),
                static_cast<std::size_t>(record.moves) * ACTION_SPACE_SIZE * sizeof(float));
            std::memcpy(
                values.mutable_data(),
                record.values.data(),
                static_cast<std::size_t>(record.moves) * sizeof(float));
            std::memcpy(
                players.mutable_data(),
                record.players.data(),
                static_cast<std::size_t>(record.moves) * sizeof(PlayerId));
        }
        py::list kingdom;
        for (std::uint8_t i = 0; i < record.kingdom_count; ++i) {
            kingdom.append(py::int_(record.kingdom[i]));
        }
        dict["observations"] = obs;
        dict["policy_targets"] = policies;
        dict["values"] = values;
        dict["players"] = players;
        py::array_t<std::int16_t> scores(MAX_PLAYERS);
        std::memcpy(
            scores.mutable_data(),
            record.scores,
            static_cast<std::size_t>(MAX_PLAYERS) * sizeof(std::int16_t));
        dict["scores"] = scores;
        dict["kingdom"] = kingdom;
        dict["seed"] = py::int_(record.seed);
        dict["winner"] = record.winner == NONE
            ? py::object(py::none())
            : py::object(py::int_(record.winner));
        dict["scripted_nn_player"] = record.scripted_nn_player == NONE
            ? py::object(py::none())
            : py::object(py::int_(record.scripted_nn_player));
        out.append(dict);
    }
    return out;
}

[[nodiscard]] py::tuple eval_collect(EvalRunner& runner, std::uint32_t max_batch) {
    std::uint32_t count = 0;
    {
        py::gil_scoped_release release;
        count = runner.collect_leaves(max_batch);
    }

    py::array_t<float> obs({
        static_cast<py::ssize_t>(count),
        static_cast<py::ssize_t>(runner.observation_size()),
    });
    py::array_t<bool> masks({
        static_cast<py::ssize_t>(count),
        static_cast<py::ssize_t>(ACTION_SPACE_SIZE),
    });
    if (count != 0U) {
        std::memcpy(
            obs.mutable_data(),
            runner.leaf_observations(),
            static_cast<std::size_t>(count) * runner.observation_size() * sizeof(float));
        std::memcpy(
            masks.mutable_data(),
            runner.leaf_legal_masks(),
            static_cast<std::size_t>(count) * ACTION_SPACE_SIZE * sizeof(bool));
    }
    return py::make_tuple(obs, masks);
}

void eval_provide(
    EvalRunner& runner,
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    py::array_t<float, py::array::c_style | py::array::forcecast> policies) {
    if (values.ndim() != 1) {
        throw std::invalid_argument("values must have shape [B]");
    }
    if (policies.ndim() != 2 || policies.shape(1) != static_cast<py::ssize_t>(ACTION_SPACE_SIZE)) {
        throw std::invalid_argument("policies must have shape [B, ACTION_SPACE]");
    }
    if (policies.shape(0) != values.shape(0)) {
        throw std::invalid_argument("values and policies batch sizes differ");
    }
    const auto count = static_cast<std::uint32_t>(values.shape(0));
    {
        py::gil_scoped_release release;
        runner.provide_evaluations(values.data(), policies.data(), count);
    }
}

[[nodiscard]] py::dict eval_result_dict(const EvalRunner& runner) {
    const EvalRunnerResult result = runner.result();
    py::dict dict;
    dict["games"] = py::int_(result.games);
    dict["nn_wins"] = py::int_(result.nn_wins);
    dict["scripted_wins"] = py::int_(result.scripted_wins);
    dict["ties"] = py::int_(result.ties);
    dict["truncated"] = py::int_(result.truncated);
    return dict;
}

[[nodiscard]] py::list eval_finished_games(EvalRunner& runner) {
    std::vector<GameState> states = runner.take_finished_games();
    py::list out;
    for (const GameState& state : states) {
        out.append(make_game(state));
    }
    return out;
}

[[nodiscard]] std::string def_constant_name(const char* name) {
    std::string constant = "DEF_";
    if (name != nullptr) {
        constant += name;
    }
    for (char& ch : constant) {
        if (ch == ' ') {
            ch = '_';
        } else if (ch >= 'a' && ch <= 'z') {
            ch = static_cast<char>(ch - ('a' - 'A'));
        }
    }
    return constant;
}

void add_def_constants(py::module_& module) {
    for (DefId def = 0; def < card_def_count(); ++def) {
        const char* name = card_def(def).name;
        if (name == nullptr || name[0] == '\0') {
            continue;
        }
        module.attr(def_constant_name(name).c_str()) = py::int_(def);
    }
}

} // namespace

PYBIND11_MODULE(dominion_v2_py, module) {
    module.doc() = "DominionZero v2 Python bindings.";

    py::class_<PySetup>(module, "Setup")
        .def(
            py::init<int, py::object, bool>(),
            py::arg("players") = 2,
            py::arg("kingdom") = py::none(),
            py::arg("use_colony_platinum") = false)
        .def("__repr__", [](const PySetup& self) {
            return "Setup(players=" + std::to_string(self.setup.num_players)
                + ", kingdom_count=" + std::to_string(self.setup.kingdom_count) + ")";
        });

    py::class_<PyGame>(module, "Game")
        .def("step", [](PyGame& self, std::uint32_t action_value) {
            if (action_value >= ACTION_SPACE_SIZE) {
                throw std::invalid_argument("action out of range");
            }
            ActionMask legal{};
            const int legal_count = Game::legal_actions(self.state, legal);
            const Action action = static_cast<Action>(action_value);
            if (legal_count <= 0 || !legal.test(action)) {
                throw std::invalid_argument("illegal action");
            }

            bool done = false;
            {
                py::gil_scoped_release release;
                done = Game::step(self.state, action);
            }
            return done;
        })
        .def("legal_mask", [](const PyGame& self) {
            return legal_mask_array(self.state);
        })
        .def("current_decision", [](const PyGame& self) {
            return decision_dict(Game::current_decision(self.state));
        })
        .def("decision_context", [](const PyGame& self) {
            return decision_context_dict(self.state);
        })
        .def("encode", [](const PyGame& self, int player, int version) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            return encode_array(self.state, static_cast<PlayerId>(player), parse_obs_version(version));
        }, py::arg("player"), py::arg("version") = static_cast<int>(ObsVersion::V1))
        .def("clone", [](const PyGame& self) {
            return make_game(self.state);
        })
        .def("determinize", [](PyGame& self, std::uint64_t seed) {
            const PlayerId player = decision_player_id(self.state);
            py::gil_scoped_release release;
            determinize(self.state, player, seed);
        })
        .def("score", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            return score(self.state, static_cast<PlayerId>(player));
        })
        .def("num_players", [](const PyGame& self) {
            return self.state.num_players;
        })
        .def("hand_count", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            int total = 0;
            const PlayerState& player_state = self.state.players[player];
            for (std::uint8_t slot = 0; slot < self.state.num_slots; ++slot) {
                total += player_state.hand[slot];
            }
            return total;
        })
        .def("deck_count", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            return self.state.players[player].deck.size;
        })
        .def("discard_count", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            return self.state.players[player].discard.size;
        })
        .def("discard", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            py::list list;
            const OrderedZone& discard = self.state.players[player].discard;
            for (std::uint8_t i = 0; i < discard.size; ++i) {
                const Slot slot = discard.cards[i];
                const DefId def = slot < self.state.num_slots ? self.state.slot_to_def[slot] : 0U;
                list.append(py::int_(def));
            }
            return list;
        })
        .def("discard_top", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            return discard_top_object(self.state, static_cast<PlayerId>(player));
        })
        .def("hand", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            py::dict dict;
            const PlayerState& player_state = self.state.players[player];
            for (std::uint8_t slot = 0; slot < self.state.num_slots; ++slot) {
                const std::uint8_t count = player_state.hand[slot];
                if (count != 0U) {
                    dict[py::int_(self.state.slot_to_def[slot])] = py::int_(count);
                }
            }
            return dict;
        })
        .def("trash", [](const PyGame& self) {
            py::dict dict;
            for (std::uint8_t slot = 0; slot < self.state.num_slots; ++slot) {
                const std::uint8_t count = self.state.trash[slot];
                if (count != 0U) {
                    dict[py::int_(self.state.slot_to_def[slot])] = py::int_(count);
                }
            }
            return dict;
        })
        .def("supply", [](const PyGame& self) {
            py::list list;
            for (std::uint8_t i = 0; i < self.state.num_piles; ++i) {
                const Pile& pile = self.state.piles[i];
                const Slot slot = pile_top_slot(pile);
                const DefId def = slot < self.state.num_slots ? self.state.slot_to_def[slot] : 0U;
                list.append(py::make_tuple(def, pile_count(pile)));
            }
            return list;
        })
        .def("in_play", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            py::list list;
            const PlayerState& player_state = self.state.players[player];
            for (std::uint8_t i = 0; i < player_state.in_play_size; ++i) {
                const Slot slot = player_state.in_play[i].slot;
                const DefId def = slot < self.state.num_slots ? self.state.slot_to_def[slot] : 0U;
                list.append(py::int_(def));
            }
            return list;
        })
        .def("set_aside", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            py::list list;
            const OrderedZone& set_aside = self.state.players[player].set_aside;
            for (std::uint8_t i = 0; i < set_aside.size; ++i) {
                const Slot slot = set_aside.cards[i];
                const DefId def = slot < self.state.num_slots ? self.state.slot_to_def[slot] : 0U;
                list.append(py::int_(def));
            }
            return list;
        })
        .def("resources", [](const PyGame& self, int player) {
            if (player < 0) {
                player = static_cast<int>(turn_player_id(self.state));
            }
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            const PlayerState& player_state = self.state.players[player];
            py::dict dict;
            dict["actions"] = py::int_(self.state.actions);
            dict["buys"] = py::int_(self.state.buys);
            dict["coins"] = py::int_(self.state.coins);
            dict["potion"] = py::int_(self.state.potion_coins);
            dict["debt"] = py::int_(player_state.debt);
            dict["coffers"] = py::int_(player_state.coffers);
            dict["villagers"] = py::int_(player_state.villagers);
            dict["favors"] = py::int_(player_state.favors);
            dict["vp_tokens"] = py::int_(vp_tokens(player_state));
            return dict;
        }, py::arg("player") = -1)
        .def("phase", [](const PyGame& self) {
            return self.state.phase;
        })
        .def("turn", [](const PyGame& self) {
            return self.state.turn_counter;
        })
        .def("game_over", [](const PyGame& self) {
            return self.state.phase == static_cast<std::uint8_t>(Phase::Over);
        })
        .def("winner", [](const PyGame& self) {
            return winner_object(self.state);
        })
        .def("truncated", [](const PyGame& self) {
            return self.state.truncated != 0U;
        })
        .def("state_hash", [](const PyGame& self) {
            return fnv1a_state(self.state);
        });

    py::class_<PyBatchRunner>(module, "BatchRunner")
        .def(py::init<const py::object&, std::size_t, std::uint64_t>(), py::arg("setups_or_kingdom"), py::arg("n_games"), py::arg("seed_base"))
        .def("reset", &PyBatchRunner::reset)
        .def("observations", &PyBatchRunner::observations)
        .def("legal_masks", &PyBatchRunner::legal_masks)
        .def("current_players", &PyBatchRunner::current_players)
        .def("step", &PyBatchRunner::step)
        .def("games_completed", &PyBatchRunner::games_completed);

    py::class_<PyDecisionSearcher>(module, "DecisionSearcher")
        .def(
            py::init<const PyGame&, int, const py::dict&>(),
            py::arg("game"),
            py::arg("perspective_player"),
            py::arg("config"))
        .def("collect_leaves", &decision_search_collect)
        .def("provide_evaluations", &decision_search_provide, py::arg("values"), py::arg("policies"))
        .def("done", &PyDecisionSearcher::done)
        .def("best_action", [](const PyDecisionSearcher& searcher) {
            if (!searcher.done()) {
                throw std::runtime_error("DecisionSearcher is not done");
            }
            return static_cast<std::uint32_t>(searcher.best_action());
        });

    py::enum_<ObsVersion>(module, "ObsVersion")
        .value("V1", ObsVersion::V1)
        .value("V2", ObsVersion::V2);

    py::enum_<SelfPlayKingdomMode>(module, "SelfPlayKingdomMode")
        .value("Fixed", SelfPlayKingdomMode::Fixed)
        .value("Random", SelfPlayKingdomMode::Random);

    py::enum_<SelfPlayValueTarget>(module, "SelfPlayValueTarget")
        .value("Outcome", SelfPlayValueTarget::Outcome)
        .value("Margin", SelfPlayValueTarget::Margin);

    py::enum_<SelfPlayScriptedBotKind>(module, "SelfPlayScriptedBotKind")
        .value("None_", SelfPlayScriptedBotKind::None)
        .value("BigMoney", SelfPlayScriptedBotKind::BigMoney)
        .value("Engine", SelfPlayScriptedBotKind::Engine)
        .value("Random", SelfPlayScriptedBotKind::Random)
        .value("Scaffold", SelfPlayScriptedBotKind::Scaffold);

    py::class_<SelfPlayConfig>(module, "SelfPlayConfig")
        .def(py::init([](
            std::uint32_t n_games,
            std::uint32_t sims_per_move,
            float c_puct,
            float dirichlet_alpha,
            float dirichlet_frac,
            std::uint16_t temp_moves,
            std::uint32_t max_batch,
            std::uint64_t seed,
            SelfPlayKingdomMode kingdom_mode,
            py::object kingdom,
            std::uint16_t max_recorded_moves,
            std::uint32_t max_tree_nodes,
            SelfPlayScriptedBotKind scripted_bot,
            PlayerId scripted_nn_player,
            bool auto_play_treasures,
            bool prune_treasure_plays,
            std::uint32_t scaffold_sims,
            SelfPlayValueTarget value_target,
            float margin_scale,
            std::uint8_t scripted_threads,
            std::uint8_t scaffold_determinizations,
            std::uint32_t scaffold_sims_opening,
            int obs_version) {
            SelfPlayConfig config{};
            config.n_games = n_games;
            config.sims_per_move = sims_per_move;
            config.c_puct = c_puct;
            config.dirichlet_alpha = dirichlet_alpha;
            config.dirichlet_frac = dirichlet_frac;
            config.temp_moves = temp_moves;
            config.max_batch = max_batch;
            config.seed = seed;
            config.kingdom_mode = kingdom_mode;
            config.max_recorded_moves = max_recorded_moves;
            config.max_tree_nodes = max_tree_nodes;
            config.scripted_bot = scripted_bot;
            config.scripted_nn_player = scripted_nn_player;
            config.auto_play_treasures = auto_play_treasures;
            config.prune_treasure_plays = prune_treasure_plays;
            config.scaffold_sims = scaffold_sims;
            config.value_target = value_target;
            config.margin_scale = margin_scale;
            config.scripted_threads = scripted_threads;
            config.scaffold_determinizations = scaffold_determinizations;
            config.scaffold_sims_opening = scaffold_sims_opening;
            config.obs_version = parse_obs_version(obs_version);
            if (!kingdom.is_none()) {
                PySetup setup(2, kingdom, false);
                config.fixed_setup = setup.setup;
            }
            return config;
        }),
            py::arg("n_games") = 64,
            py::arg("sims_per_move") = 64,
            py::arg("c_puct") = 1.25F,
            py::arg("dirichlet_alpha") = 0.30F,
            py::arg("dirichlet_frac") = 0.25F,
            py::arg("temp_moves") = 12,
            py::arg("max_batch") = 256,
            py::arg("seed") = 0x545241494EULL,
            py::arg("kingdom_mode") = SelfPlayKingdomMode::Random,
            py::arg("kingdom") = py::none(),
            py::arg("max_recorded_moves") = 512,
            py::arg("max_tree_nodes") = 4096,
            py::arg("scripted_bot") = SelfPlayScriptedBotKind::None,
            py::arg("scripted_nn_player") = 0U,
            py::arg("auto_play_treasures") = false,
            py::arg("prune_treasure_plays") = false,
            py::arg("scaffold_sims") = 400,
            py::arg("value_target") = SelfPlayValueTarget::Outcome,
            py::arg("margin_scale") = 20.0F,
            py::arg("scripted_threads") = 2U,
            py::arg("scaffold_determinizations") = 2U,
            py::arg("scaffold_sims_opening") = 0U,
            py::arg("obs_version") = static_cast<int>(ObsVersion::V1))
        .def_readwrite("n_games", &SelfPlayConfig::n_games)
        .def_readwrite("sims_per_move", &SelfPlayConfig::sims_per_move)
        .def_readwrite("c_puct", &SelfPlayConfig::c_puct)
        .def_readwrite("dirichlet_alpha", &SelfPlayConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_frac", &SelfPlayConfig::dirichlet_frac)
        .def_readwrite("temp_moves", &SelfPlayConfig::temp_moves)
        .def_readwrite("max_batch", &SelfPlayConfig::max_batch)
        .def_readwrite("seed", &SelfPlayConfig::seed)
        .def_property(
            "obs_version",
            [](const SelfPlayConfig& config) { return obs_version_value(config.obs_version); },
            [](SelfPlayConfig& config, int version) { config.obs_version = parse_obs_version(version); })
        .def_readwrite("kingdom_mode", &SelfPlayConfig::kingdom_mode)
        .def_readwrite("max_recorded_moves", &SelfPlayConfig::max_recorded_moves)
        .def_readwrite("max_tree_nodes", &SelfPlayConfig::max_tree_nodes)
        .def_readwrite("scaffold_sims", &SelfPlayConfig::scaffold_sims)
        .def_readwrite("scaffold_sims_opening", &SelfPlayConfig::scaffold_sims_opening)
        .def_readwrite("scaffold_determinizations", &SelfPlayConfig::scaffold_determinizations)
        .def_readwrite("scripted_threads", &SelfPlayConfig::scripted_threads)
        .def_readwrite("value_target", &SelfPlayConfig::value_target)
        .def_readwrite("margin_scale", &SelfPlayConfig::margin_scale)
        .def_readwrite("scripted_bot", &SelfPlayConfig::scripted_bot)
        .def_readwrite("scripted_nn_player", &SelfPlayConfig::scripted_nn_player)
        .def_readwrite("auto_play_treasures", &SelfPlayConfig::auto_play_treasures)
        .def_readwrite("prune_treasure_plays", &SelfPlayConfig::prune_treasure_plays);

    py::class_<SelfPlayRunner>(module, "SelfPlayRunner")
        .def(py::init<const SelfPlayConfig&>(), py::arg("config"))
        .def("collect_leaves", &selfplay_collect, py::arg("max_batch") = 0U)
        .def("leaf_players", &selfplay_leaf_players)
        .def("provide_evaluations", &selfplay_provide, py::arg("values"), py::arg("policies"))
        .def("finished_games", &selfplay_finished)
        .def("games_completed", &SelfPlayRunner::games_completed)
        .def("observation_size", &SelfPlayRunner::observation_size)
        .def("total_virtual_loss", &SelfPlayRunner::total_virtual_loss);

    py::enum_<EvalScriptedBotKind>(module, "EvalScriptedBotKind")
        .value("Engine", EvalScriptedBotKind::Engine)
        .value("BigMoney", EvalScriptedBotKind::BigMoney)
        .value("Heuristic", EvalScriptedBotKind::Heuristic)
        .value("Random", EvalScriptedBotKind::Random)
        .value("Mcts", EvalScriptedBotKind::Mcts);

    py::class_<EvalRunnerConfig>(module, "EvalRunnerConfig")
        .def(py::init([](
            std::uint32_t n_games,
            std::uint32_t sims_per_move,
            float c_puct,
            std::uint32_t max_batch,
            std::uint64_t seed,
            std::uint32_t target_games,
            SelfPlayKingdomMode kingdom_mode,
            py::object kingdom,
            std::uint32_t max_tree_nodes,
            EvalScriptedBotKind opponent,
            bool retain_finished_games,
            bool auto_play_treasures,
            bool prune_treasure_plays,
            int obs_version) {
            EvalRunnerConfig config{};
            config.n_games = n_games;
            config.sims_per_move = sims_per_move;
            config.c_puct = c_puct;
            config.max_batch = max_batch;
            config.seed = seed;
            config.target_games = target_games;
            config.kingdom_mode = kingdom_mode;
            config.max_tree_nodes = max_tree_nodes;
            config.opponent = opponent;
            config.retain_finished_games = retain_finished_games;
            config.auto_play_treasures = auto_play_treasures;
            config.prune_treasure_plays = prune_treasure_plays;
            config.obs_version = parse_obs_version(obs_version);
            if (!kingdom.is_none()) {
                PySetup setup(2, kingdom, false);
                config.fixed_setup = setup.setup;
            }
            return config;
        }),
            py::arg("n_games") = 64,
            py::arg("sims_per_move") = 400,
            py::arg("c_puct") = 1.25F,
            py::arg("max_batch") = 512,
            py::arg("seed") = 0x4556414CULL,
            py::arg("target_games") = 0,
            py::arg("kingdom_mode") = SelfPlayKingdomMode::Random,
            py::arg("kingdom") = py::none(),
            py::arg("max_tree_nodes") = 4096,
            py::arg("opponent") = EvalScriptedBotKind::Engine,
            py::arg("retain_finished_games") = false,
            py::arg("auto_play_treasures") = false,
            py::arg("prune_treasure_plays") = false,
            py::arg("obs_version") = static_cast<int>(ObsVersion::V1))
        .def_readwrite("n_games", &EvalRunnerConfig::n_games)
        .def_readwrite("sims_per_move", &EvalRunnerConfig::sims_per_move)
        .def_readwrite("c_puct", &EvalRunnerConfig::c_puct)
        .def_readwrite("max_batch", &EvalRunnerConfig::max_batch)
        .def_readwrite("seed", &EvalRunnerConfig::seed)
        .def_property(
            "obs_version",
            [](const EvalRunnerConfig& config) { return obs_version_value(config.obs_version); },
            [](EvalRunnerConfig& config, int version) { config.obs_version = parse_obs_version(version); })
        .def_readwrite("target_games", &EvalRunnerConfig::target_games)
        .def_readwrite("kingdom_mode", &EvalRunnerConfig::kingdom_mode)
        .def_readwrite("max_tree_nodes", &EvalRunnerConfig::max_tree_nodes)
        .def_readwrite("opponent", &EvalRunnerConfig::opponent)
        .def_readwrite("retain_finished_games", &EvalRunnerConfig::retain_finished_games)
        .def_readwrite("auto_play_treasures", &EvalRunnerConfig::auto_play_treasures)
        .def_readwrite("prune_treasure_plays", &EvalRunnerConfig::prune_treasure_plays);

    py::class_<EvalRunner>(module, "EvalRunner")
        .def(py::init<const EvalRunnerConfig&>(), py::arg("config"))
        .def("collect_leaves", &eval_collect, py::arg("max_batch") = 0U)
        .def("provide_evaluations", &eval_provide, py::arg("values"), py::arg("policies"))
        .def("result", &eval_result_dict)
        .def("games_completed", &EvalRunner::games_completed)
        .def("observation_size", &EvalRunner::observation_size)
        .def("total_virtual_loss", &EvalRunner::total_virtual_loss)
        .def("active_nn_player", &EvalRunner::active_nn_player, py::arg("index"))
        .def("active_sequence", &EvalRunner::active_sequence, py::arg("index"))
        .def("last_scripted_action", &EvalRunner::last_scripted_action)
        .def("finished_games", &eval_finished_games);

    module.def("new_game", &py_new_game, py::arg("setup"), py::arg("seed"));
    module.def("def_id", [](const std::string& name) {
        DefId def = 0;
        if (!def_from_name(name, def)) {
            throw std::invalid_argument("unknown card name: " + name);
        }
        return def;
    });

    module.attr("OBS_VERSION") = py::int_(OBS_VERSION);
    module.attr("OBS_SIZE") = py::int_(OBS_SIZE);
    module.attr("OBS_SIZE_V1") = py::int_(OBS_SIZE_V1);
    module.attr("OBS_SIZE_V2") = py::int_(OBS_SIZE_V2);
    module.def("obs_size_for", [](int version) {
        return py::int_(obs_size_for(parse_obs_version(version)));
    }, py::arg("version"));
    module.attr("ACTION_SPACE_SIZE") = py::int_(ACTION_SPACE_SIZE);
    module.attr("A_PASS") = py::int_(A_PASS);
    module.attr("A_PLAY_BASE") = py::int_(A_PLAY_BASE);
    module.attr("A_BUY_BASE") = py::int_(A_BUY_BASE);
    module.attr("A_SELECT_BASE") = py::int_(A_SELECT_BASE);
    module.attr("A_OPTION_BASE") = py::int_(A_OPTION_BASE);
    module.attr("A_CALL_BASE") = py::int_(A_CALL_BASE);
    module.attr("ACTION_DEF_COUNT") = py::int_(ACTION_DEF_COUNT);
    module.attr("MAX_PLAYERS") = py::int_(MAX_PLAYERS);
    module.attr("MAX_SLOTS") = py::int_(MAX_SLOTS);
    add_def_constants(module);
}
