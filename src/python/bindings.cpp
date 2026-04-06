#include <pybind11/pybind11.h>
#include <pybind11/stl.h>           // automatic std::vector <-> Python list conversion
#include <pybind11/functional.h>    // std::function support

#include "steppable_game.h"
#include "../game/card.h"
#include "../game/player.h"
#include "../game/supply.h"
#include "../game/game_state.h"
#include "../engine/action_space.h"

namespace py = pybind11;

// This macro defines a Python module named "dominion_engine".
// When you "import dominion_engine" in Python, this code runs.
PYBIND11_MODULE(dominion_engine, m) {
    m.doc() = "Python bindings for DominionZero game engine";

    // --- Enums ---
    // py::arithmetic() lets Python do bitwise ops on these, just like C++

    py::enum_<Phase>(m, "Phase")
        .value("ACTION", Phase::ACTION)
        .value("TREASURE", Phase::TREASURE)
        .value("BUY", Phase::BUY)
        .value("NIGHT", Phase::NIGHT)
        .value("CLEANUP", Phase::CLEANUP)
        .value("GAME_OVER", Phase::GAME_OVER);

    py::enum_<DecisionType>(m, "DecisionType")
        .value("PLAY_ACTION", DecisionType::PLAY_ACTION)
        .value("PLAY_TREASURE", DecisionType::PLAY_TREASURE)
        .value("PLAY_ALL_TREASURES", DecisionType::PLAY_ALL_TREASURES)
        .value("BUY_CARD", DecisionType::BUY_CARD)
        .value("CHOOSE_CARDS_IN_HAND", DecisionType::CHOOSE_CARDS_IN_HAND)
        .value("CHOOSE_SUPPLY_PILE", DecisionType::CHOOSE_SUPPLY_PILE)
        .value("CHOOSE_OPTION", DecisionType::CHOOSE_OPTION)
        .value("REACT", DecisionType::REACT)
        .value("PLAY_NIGHT", DecisionType::PLAY_NIGHT);

    py::enum_<ChoiceType>(m, "ChoiceType")
        .value("DISCARD", ChoiceType::DISCARD)
        .value("TRASH", ChoiceType::TRASH)
        .value("GAIN", ChoiceType::GAIN)
        .value("TOPDECK", ChoiceType::TOPDECK)
        .value("EXILE", ChoiceType::EXILE)
        .value("PLAY_CARD", ChoiceType::PLAY_CARD)
        .value("YES_NO", ChoiceType::YES_NO)
        .value("REVEAL", ChoiceType::REVEAL)
        .value("MULTI_FATE", ChoiceType::MULTI_FATE)
        .value("ORDER", ChoiceType::ORDER)
        .value("SELECT_FROM_DISCARD", ChoiceType::SELECT_FROM_DISCARD);

    // --- ActionOption ---
    // These are the individual choices within a DecisionPoint.
    // Read-only properties since Python just inspects them.

    py::class_<ActionOption>(m, "ActionOption")
        .def_readonly("local_id", &ActionOption::local_id)
        .def_readonly("card_id", &ActionOption::card_id)
        .def_readonly("def_id", &ActionOption::def_id)
        .def_readonly("pile_index", &ActionOption::pile_index)
        .def_readonly("value", &ActionOption::value)
        .def_readonly("card_name", &ActionOption::card_name)
        .def_readonly("label", &ActionOption::label)
        .def_readonly("is_pass", &ActionOption::is_pass)
        .def_readonly("is_play_all", &ActionOption::is_play_all)
        .def("__repr__", [](const ActionOption& o) {
            if (o.is_pass) return std::string("<ActionOption: PASS>");
            if (o.is_play_all) return std::string("<ActionOption: PLAY_ALL>");
            return "<ActionOption: local_id=" + std::to_string(o.local_id) +
                   " card_id=" + std::to_string(o.card_id) +
                   " def_id=" + std::to_string(o.def_id) + ">";
        });

    // --- DecisionPoint ---
    // This is what the game presents to the agent: "here are your choices"

    py::class_<DecisionPoint>(m, "DecisionPoint")
        .def_readonly("player_id", &DecisionPoint::player_id)
        .def_readonly("type", &DecisionPoint::type)
        .def_readonly("options", &DecisionPoint::options)
        .def_readonly("source_card", &DecisionPoint::source_card)
        .def_readonly("sub_choice_type", &DecisionPoint::sub_choice_type)
        .def_readonly("min_choices", &DecisionPoint::min_choices)
        .def_readonly("max_choices", &DecisionPoint::max_choices)
        .def("num_options", &DecisionPoint::num_options)
        .def("__repr__", [](const DecisionPoint& dp) {
            return "<DecisionPoint: player=" + std::to_string(dp.player_id) +
                   " options=" + std::to_string(dp.options.size()) +
                   " min=" + std::to_string(dp.min_choices) +
                   " max=" + std::to_string(dp.max_choices) + ">";
        });

    // --- GameResult ---
    py::class_<GameResult>(m, "GameResult")
        .def_readonly("scores", &GameResult::scores)
        .def_readonly("winner", &GameResult::winner)
        .def_readonly("total_turns", &GameResult::total_turns)
        .def_readonly("total_decisions", &GameResult::total_decisions)
        .def("__repr__", [](const GameResult& r) {
            std::string s = "<GameResult: winner=" + std::to_string(r.winner) + " scores=[";
            for (size_t i = 0; i < r.scores.size(); i++) {
                if (i > 0) s += ", ";
                s += std::to_string(r.scores[i]);
            }
            return s + "] turns=" + std::to_string(r.total_turns) + ">";
        });

    // --- SupplyPile ---
    py::class_<SupplyPile>(m, "SupplyPile")
        .def_readonly("pile_name", &SupplyPile::pile_name)
        .def_readonly("card_ids", &SupplyPile::card_ids)
        .def_readonly("starting_count", &SupplyPile::starting_count)
        .def("__len__", [](const SupplyPile& p) { return p.card_ids.size(); });

    // --- Supply ---
    py::class_<Supply>(m, "Supply")
        .def("num_piles", &Supply::num_piles)
        .def("pile_at", &Supply::pile_at, py::return_value_policy::reference)
        .def("count_index", &Supply::count_index)
        .def("is_pile_empty_index", &Supply::is_pile_empty_index)
        .def("pile_index_of", &Supply::pile_index_of)
        .def("empty_pile_count", &Supply::empty_pile_count)
        .def("is_game_over", &Supply::is_game_over);

    // --- Player ---
    // Note: return_value_policy::reference_internal means the returned
    // reference stays valid as long as the Player object is alive.

    py::class_<Player>(m, "Player")
        .def("get_id", &Player::get_id)
        .def("hand_size", &Player::hand_size)
        .def("deck_size", &Player::deck_size)
        .def("discard_size", &Player::discard_size)
        .def("get_hand", &Player::get_hand, py::return_value_policy::reference_internal)
        .def("get_deck", &Player::get_deck, py::return_value_policy::reference_internal)
        .def("get_discard", &Player::get_discard, py::return_value_policy::reference_internal)
        .def("get_in_play", &Player::get_in_play, py::return_value_policy::reference_internal)
        .def("total_card_count", &Player::total_card_count);

    // --- GameState (read-only access for observation building) ---
    py::class_<GameState>(m, "GameState")
        .def("num_players", &GameState::num_players)
        .def("get_player", py::overload_cast<int>(&GameState::get_player, py::const_),
             py::return_value_policy::reference_internal)
        .def("get_supply", py::overload_cast<>(&GameState::get_supply, py::const_),
             py::return_value_policy::reference_internal)
        .def("current_phase", &GameState::current_phase)
        .def("current_player_id", &GameState::current_player_id)
        .def("actions", &GameState::actions)
        .def("buys", &GameState::buys)
        .def("coins", &GameState::coins)
        .def("turn_number", &GameState::turn_number)
        .def("card_name", &GameState::card_name, py::return_value_policy::reference_internal)
        .def("card_def_id", &GameState::card_def_id)
        .def("is_game_over", &GameState::is_game_over)
        .def("calculate_scores", &GameState::calculate_scores)
        .def("get_trash", &GameState::get_trash, py::return_value_policy::reference_internal)
        .def("pile_copper", &GameState::pile_copper)
        .def("pile_silver", &GameState::pile_silver)
        .def("pile_gold", &GameState::pile_gold)
        .def("pile_estate", &GameState::pile_estate)
        .def("pile_duchy", &GameState::pile_duchy)
        .def("pile_province", &GameState::pile_province)
        .def("pile_curse", &GameState::pile_curse);

    // --- SteppableGame (the main entry point for Python) ---
    py::class_<SteppableGame>(m, "SteppableGame")
        .def(py::init<int, const std::vector<std::string>&>(),
             py::arg("num_players"), py::arg("kingdom_cards"),
             "Create a new steppable game environment")
        .def("reset", &SteppableGame::reset,
             py::return_value_policy::reference_internal,
             "Start a new game. Returns the first DecisionPoint.")
        .def("step", &SteppableGame::step,
             py::arg("action"),
             py::return_value_policy::reference_internal,
             "Provide action for current decision. Returns next DecisionPoint.")
        .def("is_game_over", &SteppableGame::is_game_over)
        .def("get_result", &SteppableGame::get_result,
             py::return_value_policy::reference_internal)
        .def("get_state", &SteppableGame::get_state,
             py::return_value_policy::reference_internal,
             "Access the underlying GameState (read-only)")
        .def("num_players", &SteppableGame::num_players)
        .def("set_ai_agent", &SteppableGame::set_ai_agent,
             py::arg("player_id"), py::arg("agent_type"),
             "Assign AI to a player slot: 'random', 'big_money', 'heuristic', 'engine_bot', or 'human'")
        .def("is_ai_player", &SteppableGame::is_ai_player,
             py::arg("player_id"),
             "Check if a player slot is AI-controlled");

    // --- Card lookup utilities ---
    // These let Python look up card info by name or def_id

    m.def("card_registry_get_by_name", [](const std::string& name) -> py::object {
        const Card* card = CardRegistry::get(name);
        if (!card) return py::none();
        // Return a dict with the card's key properties
        py::dict d;
        d["def_id"] = card->def_id;
        d["name"] = card->name;
        d["cost"] = card->cost;
        d["victory_points"] = card->victory_points;
        d["coin_value"] = card->coin_value;
        d["is_action"] = card->is_action();
        d["is_treasure"] = card->is_treasure();
        d["is_victory"] = card->is_victory();
        d["is_curse"] = card->is_curse();
        d["is_attack"] = card->is_attack();
        d["is_reaction"] = card->is_reaction();
        return d;
    }, py::arg("name"), "Look up a card definition by name");

    m.def("card_registry_all_names", &CardRegistry::all_names,
          "Get all registered card names");

    // Initialize card definitions on module import.
    // This registers all cards so they're available when you create a game.
    BaseCards::register_all();
    Level1Cards::register_all();
    GameState::cache_well_known_def_ids();
}
