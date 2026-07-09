#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "v2/core/actions.h"
#include "v2/core/defs.h"
#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/encode/encoder.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

struct PySetupObject {
    PyObject_HEAD
    Setup setup;
};

struct PyGameObject {
    PyObject_HEAD
    GameState state;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
PyTypeObject PySetupType = {PyVarObject_HEAD_INIT(nullptr, 0)};
PyTypeObject PyGameType = {PyVarObject_HEAD_INIT(nullptr, 0)};
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

[[nodiscard]] bool valid_player(const GameState& state, int player) noexcept {
    return player >= 0 && player < static_cast<int>(state.num_players);
}

[[nodiscard]] PlayerId current_player_id(const GameState& state) noexcept {
    if (state.turn_queue.size == 0U) {
        return 0U;
    }
    return state.turn_queue.entries[state.turn_queue.head].player;
}

[[nodiscard]] int pile_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? static_cast<int>(pile.mixed_len) : static_cast<int>(pile.count);
}

[[nodiscard]] Slot pile_top_slot(const Pile& pile) noexcept {
    if (pile.mixed_len > 0U) {
        return pile.mixed[pile.mixed_len - 1U];
    }
    if (pile.count > 0U) {
        return pile.base;
    }
    return pile.base;
}

[[nodiscard]] std::int16_t vp_tokens(const PlayerState& player) noexcept {
    return static_cast<std::int16_t>(
        static_cast<std::uint16_t>(player.vp_tokens_lo)
        | (static_cast<std::uint16_t>(player.vp_tokens_hi) << 8U));
}

[[nodiscard]] bool def_from_name(const char* name, DefId& out) noexcept {
    if (name == nullptr) {
        return false;
    }
    const std::uint16_t count = card_def_count();
    for (DefId def = 0; def < count; ++def) {
        if (std::strcmp(card_def(def).name, name) == 0) {
            out = def;
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool parse_def(PyObject* object, DefId& out) {
    if (PyUnicode_Check(object) != 0) {
        const char* name = PyUnicode_AsUTF8(object);
        if (name == nullptr) {
            return false;
        }
        if (!def_from_name(name, out)) {
            PyErr_Format(PyExc_ValueError, "unknown card name: %s", name);
            return false;
        }
        return true;
    }

    const unsigned long value = PyLong_AsUnsignedLong(object);
    if (PyErr_Occurred() != nullptr) {
        return false;
    }
    if (value >= card_def_count()) {
        PyErr_SetString(PyExc_ValueError, "def id out of range");
        return false;
    }
    out = static_cast<DefId>(value);
    return true;
}

[[nodiscard]] bool fill_kingdom(Setup& setup, PyObject* kingdom_object) {
    if (kingdom_object == nullptr || kingdom_object == Py_None) {
        return true;
    }

    PyObject* sequence = PySequence_Fast(kingdom_object, "kingdom must be a sequence");
    if (sequence == nullptr) {
        return false;
    }

    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    if (size > MAX_KINGDOM_DEFS) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "kingdom has too many cards");
        return false;
    }

    setup.kingdom_count = static_cast<std::uint8_t>(size);
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t i = 0; i < size; ++i) {
        DefId def = 0;
        if (!parse_def(items[i], def)) {
            Py_DECREF(sequence);
            return false;
        }
        setup.kingdom[i] = def;
    }

    Py_DECREF(sequence);
    return true;
}

[[nodiscard]] PyObject* py_game_from_state(const GameState& state) {
    auto* object = PyObject_New(PyGameObject, &PyGameType);
    if (object == nullptr) {
        return nullptr;
    }
    object->state = state;
    return reinterpret_cast<PyObject*>(object);
}

int PySetup_init(PySetupObject* self, PyObject* args, PyObject* kwargs) {
    int players = 2;
    PyObject* kingdom = nullptr;
    int colony = 0;
    static const char* kwlist[] = {"players", "kingdom", "use_colony_platinum", nullptr};
    if (PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|iOp",
            const_cast<char**>(kwlist),
            &players,
            &kingdom,
            &colony) == 0) {
        return -1;
    }
    if (players < 2 || players > MAX_PLAYERS) {
        PyErr_SetString(PyExc_ValueError, "players must be between 2 and MAX_PLAYERS");
        return -1;
    }

    self->setup = Setup{};
    self->setup.num_players = static_cast<PlayerId>(players);
    self->setup.use_colony_platinum = colony != 0;
    if (!fill_kingdom(self->setup, kingdom)) {
        return -1;
    }
    return 0;
}

PyObject* PySetup_repr(PySetupObject* self) {
    return PyUnicode_FromFormat(
        "Setup(players=%u, kingdom_count=%u)",
        static_cast<unsigned>(self->setup.num_players),
        static_cast<unsigned>(self->setup.kingdom_count));
}

PyObject* py_def_id(PyObject*, PyObject* args) {
    const char* name = nullptr;
    if (PyArg_ParseTuple(args, "s", &name) == 0) {
        return nullptr;
    }
    DefId def = 0;
    if (!def_from_name(name, def)) {
        PyErr_Format(PyExc_ValueError, "unknown card name: %s", name);
        return nullptr;
    }
    return PyLong_FromUnsignedLong(def);
}

[[nodiscard]] bool set_dict_long(PyObject* dict, const char* key, long value) {
    PyObject* object = PyLong_FromLong(value);
    if (object == nullptr) {
        return false;
    }
    const int result = PyDict_SetItemString(dict, key, object);
    Py_DECREF(object);
    return result == 0;
}

PyObject* py_new_game(PyObject*, PyObject* args) {
    PyObject* setup_object = nullptr;
    unsigned long long seed = 0;
    if (PyArg_ParseTuple(args, "O!K", &PySetupType, &setup_object, &seed) == 0) {
        return nullptr;
    }

    const auto* setup = reinterpret_cast<PySetupObject*>(setup_object);
    GameState state{};
    Py_BEGIN_ALLOW_THREADS
    state = Game::new_game(setup->setup, static_cast<std::uint64_t>(seed));
    Py_END_ALLOW_THREADS
    return py_game_from_state(state);
}

PyObject* PyGame_step(PyGameObject* self, PyObject* args) {
    unsigned long action_value = 0;
    if (PyArg_ParseTuple(args, "k", &action_value) == 0) {
        return nullptr;
    }
    if (action_value >= ACTION_SPACE_SIZE) {
        PyErr_SetString(PyExc_ValueError, "action out of range");
        return nullptr;
    }

    ActionMask legal{};
    const int legal_count = Game::legal_actions(self->state, legal);
    const Action action = static_cast<Action>(action_value);
    if (legal_count <= 0 || !legal.test(action)) {
        PyErr_SetString(PyExc_ValueError, "illegal action");
        return nullptr;
    }

    bool done = false;
    Py_BEGIN_ALLOW_THREADS
    done = Game::step(self->state, action);
    Py_END_ALLOW_THREADS
    if (done) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* PyGame_legal_mask(PyGameObject* self, PyObject*) {
    npy_intp dims[1] = {static_cast<npy_intp>(ACTION_SPACE_SIZE)};
    auto* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_BOOL));
    if (array == nullptr) {
        return nullptr;
    }

    ActionMask legal{};
    int legal_count = 0;
    Py_BEGIN_ALLOW_THREADS
    legal_count = Game::legal_actions(self->state, legal);
    Py_END_ALLOW_THREADS
    (void)legal_count;

    auto* data = static_cast<npy_bool*>(PyArray_DATA(array));
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        data[action] = legal.test(action) ? NPY_TRUE : NPY_FALSE;
    }
    return reinterpret_cast<PyObject*>(array);
}

PyObject* PyGame_current_decision(PyGameObject* self, PyObject*) {
    const PendingDecision decision = Game::current_decision(self->state);
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }

    if (!set_dict_long(dict, "player", decision.player)
        || !set_dict_long(dict, "kind", decision.kind)
        || !set_dict_long(dict, "source", decision.source)
        || !set_dict_long(dict, "min", decision.min_left)
        || !set_dict_long(dict, "max", decision.max_left)) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* PyGame_encode(PyGameObject* self, PyObject* args) {
    int player = 0;
    if (PyArg_ParseTuple(args, "i", &player) == 0) {
        return nullptr;
    }
    if (!valid_player(self->state, player)) {
        PyErr_SetString(PyExc_ValueError, "invalid player");
        return nullptr;
    }

    npy_intp dims[1] = {static_cast<npy_intp>(OBS_SIZE)};
    auto* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_FLOAT32));
    if (array == nullptr) {
        return nullptr;
    }

    auto* data = static_cast<float*>(PyArray_DATA(array));
    Py_BEGIN_ALLOW_THREADS
    encode(self->state, static_cast<PlayerId>(player), data);
    Py_END_ALLOW_THREADS
    return reinterpret_cast<PyObject*>(array);
}

PyObject* PyGame_clone(PyGameObject* self, PyObject*) {
    return py_game_from_state(self->state);
}

PyObject* PyGame_score(PyGameObject* self, PyObject* args) {
    int player = 0;
    if (PyArg_ParseTuple(args, "i", &player) == 0) {
        return nullptr;
    }
    if (!valid_player(self->state, player)) {
        PyErr_SetString(PyExc_ValueError, "invalid player");
        return nullptr;
    }

    const std::int16_t value = score(self->state, static_cast<PlayerId>(player));
    return PyLong_FromLong(value);
}

PyObject* PyGame_hand(PyGameObject* self, PyObject* args) {
    int player = 0;
    if (PyArg_ParseTuple(args, "i", &player) == 0) {
        return nullptr;
    }
    if (!valid_player(self->state, player)) {
        PyErr_SetString(PyExc_ValueError, "invalid player");
        return nullptr;
    }

    const PlayerState& player_state = self->state.players[player];
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    for (std::uint8_t slot = 0; slot < self->state.num_slots; ++slot) {
        const std::uint8_t count = player_state.hand[slot];
        if (count == 0U) {
            continue;
        }
        PyObject* key = PyLong_FromUnsignedLong(self->state.slot_to_def[slot]);
        PyObject* value = PyLong_FromUnsignedLong(count);
        if (key == nullptr || value == nullptr || PyDict_SetItem(dict, key, value) != 0) {
            Py_XDECREF(key);
            Py_XDECREF(value);
            Py_DECREF(dict);
            return nullptr;
        }
        Py_DECREF(key);
        Py_DECREF(value);
    }
    return dict;
}

PyObject* PyGame_supply(PyGameObject* self, PyObject*) {
    PyObject* list = PyList_New(self->state.num_piles);
    if (list == nullptr) {
        return nullptr;
    }
    for (std::uint8_t i = 0; i < self->state.num_piles; ++i) {
        const Pile& pile = self->state.piles[i];
        const Slot slot = pile_top_slot(pile);
        const DefId def = slot < self->state.num_slots ? self->state.slot_to_def[slot] : 0U;
        PyObject* tuple = Py_BuildValue("(ii)", static_cast<int>(def), pile_count(pile));
        if (tuple == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, tuple);
    }
    return list;
}

PyObject* PyGame_in_play(PyGameObject* self, PyObject* args) {
    int player = 0;
    if (PyArg_ParseTuple(args, "i", &player) == 0) {
        return nullptr;
    }
    if (!valid_player(self->state, player)) {
        PyErr_SetString(PyExc_ValueError, "invalid player");
        return nullptr;
    }

    const PlayerState& player_state = self->state.players[player];
    PyObject* list = PyList_New(player_state.in_play_size);
    if (list == nullptr) {
        return nullptr;
    }
    for (std::uint8_t i = 0; i < player_state.in_play_size; ++i) {
        const Slot slot = player_state.in_play[i].slot;
        const DefId def = slot < self->state.num_slots ? self->state.slot_to_def[slot] : 0U;
        PyObject* value = PyLong_FromUnsignedLong(def);
        if (value == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, value);
    }
    return list;
}

PyObject* PyGame_resources(PyGameObject* self, PyObject* args) {
    int player = -1;
    if (PyArg_ParseTuple(args, "|i", &player) == 0) {
        return nullptr;
    }
    if (player < 0) {
        player = static_cast<int>(current_player_id(self->state));
    }
    if (!valid_player(self->state, player)) {
        PyErr_SetString(PyExc_ValueError, "invalid player");
        return nullptr;
    }

    const PlayerState& player_state = self->state.players[player];
    return Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i}",
        "actions",
        static_cast<int>(self->state.actions),
        "buys",
        static_cast<int>(self->state.buys),
        "coins",
        static_cast<int>(self->state.coins),
        "potion",
        static_cast<int>(self->state.potion_coins),
        "debt",
        static_cast<int>(player_state.debt),
        "coffers",
        static_cast<int>(player_state.coffers),
        "villagers",
        static_cast<int>(player_state.villagers),
        "favors",
        static_cast<int>(player_state.favors),
        "vp_tokens",
        static_cast<int>(vp_tokens(player_state)));
}

PyObject* PyGame_phase(PyGameObject* self, PyObject*) {
    return PyLong_FromUnsignedLong(self->state.phase);
}

PyObject* PyGame_turn(PyGameObject* self, PyObject*) {
    return PyLong_FromUnsignedLong(self->state.turn_counter);
}

PyObject* PyGame_game_over(PyGameObject* self, PyObject*) {
    if (self->state.phase == static_cast<std::uint8_t>(Phase::Over)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* PyGame_winner(PyGameObject* self, PyObject*) {
    if (self->state.phase != static_cast<std::uint8_t>(Phase::Over)) {
        Py_RETURN_NONE;
    }

    PlayerId winner = 0;
    bool tied = false;
    for (PlayerId player = 1; player < self->state.num_players; ++player) {
        const std::int16_t player_score = score(self->state, player);
        const std::int16_t winner_score = score(self->state, winner);
        if (player_score > winner_score) {
            winner = player;
            tied = false;
        } else if (player_score == winner_score) {
            tied = true;
        }
    }

    if (tied) {
        Py_RETURN_NONE;
    }
    return PyLong_FromUnsignedLong(winner);
}

PyMethodDef PyGame_methods[] = {
    {"step", reinterpret_cast<PyCFunction>(PyGame_step), METH_VARARGS, "Apply an action and return done."},
    {"legal_mask", reinterpret_cast<PyCFunction>(PyGame_legal_mask), METH_NOARGS, "Return legal actions as a numpy bool array."},
    {"current_decision", reinterpret_cast<PyCFunction>(PyGame_current_decision), METH_NOARGS, "Return the current decision metadata."},
    {"encode", reinterpret_cast<PyCFunction>(PyGame_encode), METH_VARARGS, "Return an observation for a player."},
    {"clone", reinterpret_cast<PyCFunction>(PyGame_clone), METH_NOARGS, "Clone the game state."},
    {"score", reinterpret_cast<PyCFunction>(PyGame_score), METH_VARARGS, "Return player score."},
    {"hand", reinterpret_cast<PyCFunction>(PyGame_hand), METH_VARARGS, "Return hand counts by def id."},
    {"supply", reinterpret_cast<PyCFunction>(PyGame_supply), METH_NOARGS, "Return supply piles as (def, count)."},
    {"in_play", reinterpret_cast<PyCFunction>(PyGame_in_play), METH_VARARGS, "Return in-play def ids."},
    {"resources", reinterpret_cast<PyCFunction>(PyGame_resources), METH_VARARGS, "Return resource scalars."},
    {"phase", reinterpret_cast<PyCFunction>(PyGame_phase), METH_NOARGS, "Return phase enum value."},
    {"turn", reinterpret_cast<PyCFunction>(PyGame_turn), METH_NOARGS, "Return completed turn count."},
    {"game_over", reinterpret_cast<PyCFunction>(PyGame_game_over), METH_NOARGS, "Return whether the game is over."},
    {"winner", reinterpret_cast<PyCFunction>(PyGame_winner), METH_NOARGS, "Return winner or None."},
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef module_methods[] = {
    {"new_game", py_new_game, METH_VARARGS, "Create a new game."},
    {"def_id", py_def_id, METH_VARARGS, "Look up a card definition id by name."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "dominion_v2_py",
    "DominionZero v2 Python bindings.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

int add_int_constant(PyObject* module, const char* name, long value) {
    return PyModule_AddIntConstant(module, name, value);
}

bool add_def_constants(PyObject* module) {
    for (DefId def = 0; def < card_def_count(); ++def) {
        const char* name = card_def(def).name;
        if (name == nullptr || name[0] == '\0') {
            continue;
        }
        char constant[96]{};
        int write = std::snprintf(constant, sizeof(constant), "DEF_%s", name);
        if (write <= 0 || write >= static_cast<int>(sizeof(constant))) {
            continue;
        }
        for (int i = 4; constant[i] != '\0'; ++i) {
            if (constant[i] == ' ') {
                constant[i] = '_';
            } else if (constant[i] >= 'a' && constant[i] <= 'z') {
                constant[i] = static_cast<char>(constant[i] - ('a' - 'A'));
            }
        }
        if (add_int_constant(module, constant, def) != 0) {
            return false;
        }
    }
    return true;
}

} // namespace

PyMODINIT_FUNC PyInit_dominion_v2_py() {
    import_array();

    PySetupType.tp_name = "dominion_v2_py.Setup";
    PySetupType.tp_basicsize = sizeof(PySetupObject);
    PySetupType.tp_flags = Py_TPFLAGS_DEFAULT;
    PySetupType.tp_doc = "Dominion v2 setup.";
    PySetupType.tp_init = reinterpret_cast<initproc>(PySetup_init);
    PySetupType.tp_new = PyType_GenericNew;
    PySetupType.tp_repr = reinterpret_cast<reprfunc>(PySetup_repr);
    if (PyType_Ready(&PySetupType) < 0) {
        return nullptr;
    }

    PyGameType.tp_name = "dominion_v2_py.Game";
    PyGameType.tp_basicsize = sizeof(PyGameObject);
    PyGameType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyGameType.tp_doc = "Dominion v2 game state.";
    PyGameType.tp_methods = PyGame_methods;
    if (PyType_Ready(&PyGameType) < 0) {
        return nullptr;
    }

    PyObject* module = PyModule_Create(&module_def);
    if (module == nullptr) {
        return nullptr;
    }

    Py_INCREF(&PySetupType);
    if (PyModule_AddObject(module, "Setup", reinterpret_cast<PyObject*>(&PySetupType)) != 0) {
        Py_DECREF(&PySetupType);
        Py_DECREF(module);
        return nullptr;
    }

    Py_INCREF(&PyGameType);
    if (PyModule_AddObject(module, "Game", reinterpret_cast<PyObject*>(&PyGameType)) != 0) {
        Py_DECREF(&PyGameType);
        Py_DECREF(module);
        return nullptr;
    }

    if (add_int_constant(module, "OBS_VERSION", OBS_VERSION) != 0
        || add_int_constant(module, "OBS_SIZE", static_cast<long>(OBS_SIZE)) != 0
        || add_int_constant(module, "ACTION_SPACE_SIZE", ACTION_SPACE_SIZE) != 0
        || add_int_constant(module, "MAX_PLAYERS", MAX_PLAYERS) != 0
        || add_int_constant(module, "MAX_SLOTS", MAX_SLOTS) != 0
        || !add_def_constants(module)) {
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}
