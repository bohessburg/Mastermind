#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/determinize.h"
#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/encode/encoder.h"

#include <cstdint>
#include <cstring>
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

[[nodiscard]] py::array_t<float> encode_array(const GameState& state, PlayerId player) {
    py::array_t<float> array(static_cast<py::ssize_t>(OBS_SIZE));
    float* data = array.mutable_data();
    {
        py::gil_scoped_release release;
        encode(state, player, data);
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
        .def("encode", [](const PyGame& self, int player) {
            if (!valid_player(self.state, player)) {
                throw std::invalid_argument("invalid player");
            }
            return encode_array(self.state, static_cast<PlayerId>(player));
        })
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
