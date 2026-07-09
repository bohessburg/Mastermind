#include "diff_bridge.h"

#include "game/cards/base_cards.h"
#include "game/cards/level_1_cards.h"
#include "game/game_state.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace dz_diff {
namespace {

void ensure_v1_cards_registered() {
    static bool registered = false;
    if (!registered) {
        BaseCards::register_all();
        Level1Cards::register_all();
        registered = true;
    }
}

void add_cards_to_hand(GameState& state, int player_id, const std::vector<std::string>& names) {
    Player& player = state.get_player(player_id);
    for (const std::string& name : names) {
        player.add_to_hand(state.create_card(name));
    }
}

void add_cards_to_deck(GameState& state, int player_id, const std::vector<std::string>& names) {
    Player& player = state.get_player(player_id);
    for (const std::string& name : names) {
        player.add_to_deck_top(state.create_card(name));
    }
}

void add_cards_to_discard(GameState& state, int player_id, const std::vector<std::string>& names) {
    Player& player = state.get_player(player_id);
    for (const std::string& name : names) {
        player.add_to_discard(state.create_card(name));
    }
}

[[nodiscard]] int pile_index_for_name(const GameState& state, const std::string& name) {
    const int index = state.get_supply().pile_index_of(name);
    if (index < 0) {
        throw std::runtime_error("v1 missing supply pile: " + name);
    }
    return index;
}

[[nodiscard]] int hand_option_named(
    const GameState& state,
    int player_id,
    const std::vector<int>& options,
    const std::string& name) {
    const Player& player = state.get_player(player_id);
    const auto& hand = player.get_hand();
    for (std::size_t i = 0; i < options.size(); ++i) {
        const int hand_index = options[i];
        if (hand_index >= 0
            && static_cast<std::size_t>(hand_index) < hand.size()
            && state.card_name(hand[static_cast<std::size_t>(hand_index)]) == name) {
            return static_cast<int>(i);
        }
    }
    throw std::runtime_error("v1 missing hand option: " + name);
}

[[nodiscard]] int gain_option_named(
    const GameState& state,
    const std::vector<int>& options,
    const std::string& name) {
    for (std::size_t i = 0; i < options.size(); ++i) {
        if (state.card_name(options[i]) == name) {
            return static_cast<int>(i);
        }
    }
    throw std::runtime_error("v1 missing gain option: " + name);
}

[[nodiscard]] DecisionFn decision_fn_for(GameState& state, const Step& step) {
    return [&state, choices = step.choices, cursor = std::size_t{0}](
               int player_id,
               ChoiceType choice_type,
               const std::vector<int>& options,
               int min_choices,
               int /*max_choices*/) mutable -> std::vector<int> {
        if (cursor >= choices.size()) {
            std::vector<int> fallback;
            for (int i = 0; i < min_choices && static_cast<std::size_t>(i) < options.size(); ++i) {
                fallback.push_back(i);
            }
            return fallback;
        }

        const Choice choice = choices[cursor];
        ++cursor;
        if (choice.kind == ChoiceKind::Pass) {
            return {};
        }
        if (choice.kind == ChoiceKind::Option) {
            return {choice.option};
        }
        if (choice_type == ChoiceType::YES_NO) {
            return {choice.name == "Moat" ? 1 : 0};
        }
        if (choice_type == ChoiceType::GAIN) {
            return {gain_option_named(state, options, choice.name)};
        }
        return {hand_option_named(state, player_id, options, choice.name)};
    };
}

void play_named(GameState& state, int player_id, const Step& step) {
    Player& player = state.get_player(player_id);
    const auto& hand = player.get_hand();
    for (std::size_t i = 0; i < hand.size(); ++i) {
        if (state.card_name(hand[i]) == step.name) {
            state.play_card_from_hand(player_id, static_cast<int>(i), decision_fn_for(state, step));
            return;
        }
    }
    throw std::runtime_error("v1 missing hand card to play: " + step.name);
}

void start_buy(GameState& state) {
    if (state.current_phase() == Phase::ACTION) {
        state.advance_phase();
    }
}

void end_turn(GameState& state);

void buy_named(GameState& state, int player_id, const std::string& name) {
    while (state.current_phase() != Phase::BUY) {
        state.advance_phase();
    }
    const int pile = pile_index_for_name(state, name);
    const int top = state.get_supply().top_card_index(pile);
    const Card* card = state.card_def(top);
    if (card == nullptr) {
        throw std::runtime_error("v1 empty buy pile: " + name);
    }
    state.add_coins(-card->cost);
    state.add_buys(-1);
    state.gain_card(player_id, pile);
    if (state.buys() <= 0) {
        end_turn(state);
    }
}

void end_turn(GameState& state) {
    while (state.current_phase() != Phase::CLEANUP) {
        state.advance_phase();
    }
    state.advance_phase();
}

void add_card_name(std::map<std::string, int>& out, const GameState& state, int card_id) {
    ++out[state.card_name(card_id)];
}

[[nodiscard]] std::map<std::string, int> owned_counts(const GameState& state, int player_id) {
    std::map<std::string, int> result;
    for (int card_id : state.get_player(player_id).all_cards()) {
        add_card_name(result, state, card_id);
    }
    return result;
}

[[nodiscard]] std::map<std::string, int> trash_counts(const GameState& state) {
    std::map<std::string, int> result;
    for (int card_id : state.get_trash()) {
        add_card_name(result, state, card_id);
    }
    return result;
}

[[nodiscard]] std::map<std::string, int> supply_counts(const GameState& state) {
    std::map<std::string, int> result;
    for (const SupplyPile& pile : state.get_supply().piles()) {
        result[pile.pile_name] = static_cast<int>(pile.card_ids.size());
    }
    return result;
}

[[nodiscard]] std::string dump_zones(const GameState& state) {
    std::ostringstream out;
    for (int player_id = 0; player_id < state.num_players(); ++player_id) {
        const Player& player = state.get_player(player_id);
        out << "zones p" << player_id
            << " hand=" << player.hand_size()
            << " deck=" << player.deck_size()
            << " discard=" << player.discard_size()
            << " in_play=" << player.get_in_play().size()
            << "\n";
    }
    return out.str();
}

} // namespace

Snapshot run_v1(const Scenario& scenario) {
    ensure_v1_cards_registered();

    GameState state(2);
    BaseCards::setup_supply(state, scenario.kingdom);
    for (std::size_t player = 0; player < scenario.players.size(); ++player) {
        const PlayerSetup& setup = scenario.players[player];
        const int player_id = static_cast<int>(player);
        add_cards_to_hand(state, player_id, setup.hand);
        add_cards_to_deck(state, player_id, setup.deck);
        add_cards_to_discard(state, player_id, setup.discard);
    }
    state.start_game();

    for (const Step& step : scenario.steps) {
        switch (step.kind) {
        case StepKind::StartBuy:
            start_buy(state);
            break;
        case StepKind::Play:
            play_named(state, state.current_player_id(), step);
            break;
        case StepKind::Buy:
            buy_named(state, state.current_player_id(), step.name);
            break;
        case StepKind::EndTurn:
            end_turn(state);
            break;
        }
    }

    Snapshot snapshot{};
    snapshot.owned.resize(static_cast<std::size_t>(state.num_players()));
    for (int player = 0; player < state.num_players(); ++player) {
        snapshot.owned[static_cast<std::size_t>(player)] = owned_counts(state, player);
    }
    snapshot.supply = supply_counts(state);
    snapshot.trash = trash_counts(state);
    snapshot.scores = state.calculate_scores();
    snapshot.completed_turns = state.turn_number() - 1;
    snapshot.dump = dump_zones(state);
    return snapshot;
}

} // namespace dz_diff
