#include "v2/core/score.h"
#include "v2/drivers/bots.h"
#include "v2/observe/card_text.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Option {
    Action action = A_PASS;
    std::string label;
};

[[nodiscard]] DefId def_for_slot(const GameState& state, Slot slot) noexcept {
    return slot < state.num_slots ? state.slot_to_def[slot] : DEF_COPPER;
}

[[nodiscard]] int pile_count(const Pile& pile) noexcept {
    return pile.mixed_len > 0U ? static_cast<int>(pile.mixed_len) : static_cast<int>(pile.count);
}

[[nodiscard]] DefId pile_top_def(const GameState& state, const Pile& pile) noexcept {
    const Slot slot = pile.mixed_len > 0U ? pile.mixed[pile.mixed_len - 1U] : pile.base;
    return def_for_slot(state, slot);
}

[[nodiscard]] std::string card_name(DefId def) {
    return def < card_def_count() ? card_def(def).name : "Unknown";
}

[[nodiscard]] std::string kind_name(const PendingDecision& decision) {
    switch (static_cast<DecisionKind>(decision.kind)) {
    case DecisionKind::PhaseAction:
        return "Action";
    case DecisionKind::PhaseBuy:
        return "Buy";
    case DecisionKind::PhaseNight:
        return "Night";
    case DecisionKind::Choose:
        return "Choose";
    case DecisionKind::ChooseGain:
        return "Gain";
    case DecisionKind::ChooseOption:
        return "Option";
    case DecisionKind::ChooseOrder:
        return "Order";
    case DecisionKind::ReactWindow:
        return "Reaction";
    case DecisionKind::OrderTriggers:
        return "Trigger order";
    case DecisionKind::None:
    default:
        return "None";
    }
}

[[nodiscard]] std::string prompt_for(const GameState& state) {
    const PendingDecision& decision = state.decision;
    const std::string source = decision.source < card_def_count() ? card_name(decision.source) : "";
    switch (static_cast<DecisionKind>(decision.kind)) {
    case DecisionKind::PhaseAction:
        return "Action phase";
    case DecisionKind::PhaseBuy:
        return "Buy phase";
    case DecisionKind::PhaseNight:
        return "Night phase";
    case DecisionKind::ReactWindow:
        return source + ": reveal a Reaction?";
    case DecisionKind::OrderTriggers:
        return "Choose the next trigger to resolve";
    case DecisionKind::ChooseGain:
        return source + ": gain a card";
    case DecisionKind::ChooseOrder:
        return source + ": choose an order";
    case DecisionKind::ChooseOption:
        return source + ": choose an option";
    case DecisionKind::Choose:
        return source + ": choose cards";
    case DecisionKind::None:
    default:
        return "Waiting";
    }
}

[[nodiscard]] std::string option_label(const GameState& state, Action action) {
    if (action == A_PASS) {
        const DecisionKind kind = static_cast<DecisionKind>(state.decision.kind);
        return (kind == DecisionKind::Choose || kind == DecisionKind::ChooseGain) ? "Done" : "Pass";
    }
    if (action_is_play(action)) {
        return "Play " + card_name(action_def(action, A_PLAY_BASE));
    }
    if (action_is_buy(action)) {
        return "Buy " + card_name(action_def(action, A_BUY_BASE));
    }
    if (action_is_select(action)) {
        const DefId def = action_def(action, A_SELECT_BASE);
        const DecisionKind kind = static_cast<DecisionKind>(state.decision.kind);
        if (kind == DecisionKind::ChooseGain) {
            return "Gain " + card_name(def);
        }
        if (kind == DecisionKind::ReactWindow) {
            return "Reveal " + card_name(def);
        }
        if (state.decision.source == DEF_MILITIA) {
            return "Keep " + card_name(def);
        }
        if (state.decision.source == DEF_CHAPEL || state.decision.source == DEF_REMODEL
            || state.decision.source == DEF_MINE || state.decision.source == DEF_BANDIT
            || state.decision.source == DEF_MONEYLENDER) {
            return "Trash " + card_name(def);
        }
        return "Select " + card_name(def);
    }
    if (action_is_option(action)) {
        const std::uint8_t option = static_cast<std::uint8_t>(action_def(action, A_OPTION_BASE));
        if (state.decision.source == DEF_SENTRY
            && static_cast<DecisionKind>(state.decision.kind) == DecisionKind::ChooseOption) {
            constexpr const char* kSentry[] = {"Trash", "Discard", "Keep"};
            if (option < 3U) {
                return kSentry[option];
            }
        }
        if (state.decision.source == DEF_LIBRARY) {
            return option == 0U ? "Keep Action" : "Set aside Action";
        }
        if (state.decision.source == DEF_VASSAL) {
            return option == 0U ? "Decline" : "Play Action";
        }
        return "Option " + std::to_string(static_cast<int>(option + 1U));
    }
    return "Action " + std::to_string(action);
}

[[nodiscard]] std::vector<Option> legal_options(const GameState& state, const ActionMask& mask) {
    std::vector<Option> options;
    for (Action action = 0; action < ACTION_SPACE_SIZE; ++action) {
        if (mask.test(action)) {
            options.push_back(Option{action, option_label(state, action)});
        }
    }
    return options;
}

void add_default_kingdom(Setup& setup) noexcept {
    constexpr DefId kKingdom[] = {
        DEF_VILLAGE,
        DEF_SMITHY,
        DEF_MARKET,
        DEF_MILITIA,
        DEF_MOAT,
        DEF_WITCH,
        DEF_BANDIT,
        DEF_THRONE_ROOM,
        DEF_LIBRARY,
        DEF_SENTRY,
    };
    setup.kingdom_count = static_cast<std::uint8_t>(sizeof(kKingdom) / sizeof(kKingdom[0]));
    for (std::uint8_t i = 0; i < setup.kingdom_count; ++i) {
        setup.kingdom[i] = kKingdom[i];
    }
}

[[nodiscard]] BotKind parse_bot(const std::string& name) noexcept {
    if (name == "random") {
        return BotKind::Random;
    }
    if (name == "heuristic") {
        return BotKind::Heuristic;
    }
    if (name == "engine") {
        return BotKind::Engine;
    }
    if (name == "engine3") {
        return BotKind::EngineV3;
    }
    return BotKind::BigMoney;
}

[[nodiscard]] Action choose_bot(
    BotKind kind,
    RandomBot& random,
    BigMoneyBot& big_money,
    HeuristicBot& heuristic,
    EngineBot& engine,
    EngineBotV3& engine_v3,
    const GameState& state,
    const ActionMask& legal,
    int legal_count) noexcept {
    switch (kind) {
    case BotKind::Random:
        return random.choose_action(state, legal, legal_count);
    case BotKind::Heuristic:
        return heuristic.choose_action(state, legal, legal_count);
    case BotKind::Engine:
        return engine.choose_action(state, legal, legal_count);
    case BotKind::EngineV3:
        return engine_v3.choose_action(state, legal, legal_count);
    case BotKind::BigMoney:
    default:
        return big_money.choose_action(state, legal, legal_count);
    }
}

void render_zone_counts(const GameState& state, const PlayerState& player) {
    bool any = false;
    for (Slot slot = 0; slot < state.num_slots; ++slot) {
        if (player.hand[slot] == 0U) {
            continue;
        }
        any = true;
        const DefId def = def_for_slot(state, slot);
        std::cout << "  " << card_name(def) << " x" << static_cast<int>(player.hand[slot]) << '\n';
    }
    if (!any) {
        std::cout << "  (empty)\n";
    }
}

void render(const GameState& state, PlayerId human) {
    std::cout << "\033[2J\033[H";
    std::cout << "DominionZero v2  Turn " << state.turn_counter << "\n\n";
    std::cout << "Supply\n";
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const DefId def = pile_top_def(state, state.piles[i]);
        std::cout << "  " << card_name(def) << " [" << pile_count(state.piles[i]) << "]  "
                  << card_text(def) << '\n';
    }
    std::cout << "\nP1 score " << score(state, 0) << " | P2 score " << score(state, 1) << "\n";
    const PlayerState& me = state.players[human];
    std::cout << "Deck " << static_cast<int>(me.deck.size)
              << "  Discard " << static_cast<int>(me.discard.size)
              << "  Actions " << static_cast<int>(state.actions)
              << "  Buys " << static_cast<int>(state.buys)
              << "  Coins " << state.coins << "\n\n";
    std::cout << "Hand\n";
    render_zone_counts(state, me);
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    BotKind bot_kind = BotKind::Engine;
    std::uint64_t seed = 0xD02A'5100ULL;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bot" && i + 1 < argc) {
            bot_kind = parse_bot(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 0));
        }
    }

    Setup setup{};
    setup.num_players = 2;
    add_default_kingdom(setup);
    GameState state = Game::new_game(setup, seed);

    RandomBot random{seed ^ 0xA11CEULL};
    BigMoneyBot big_money{};
    HeuristicBot heuristic{};
    EngineBot engine{};
    EngineBotV3 engine_v3{};

    constexpr PlayerId kHuman = 0;
    bool done = state.phase == static_cast<std::uint8_t>(Phase::Over);
    while (!done) {
        ActionMask legal{};
        const int legal_count = Game::legal_actions(state, legal);
        if (legal_count <= 0) {
            std::cout << "No legal actions.\n";
            return 1;
        }

        const PlayerId current = Game::current_decision(state).player;
        if (current != kHuman) {
            const Action action = choose_bot(
                bot_kind,
                random,
                big_money,
                heuristic,
                engine,
                engine_v3,
                state,
                legal,
                legal_count);
            std::cout << "Bot chooses " << option_label(state, action) << "\n";
            done = Game::step(state, action);
            continue;
        }

        render(state, kHuman);
        std::cout << prompt_for(state) << " (" << kind_name(state.decision) << ")\n";
        const std::vector<Option> options = legal_options(state, legal);
        for (std::size_t i = 0; i < options.size(); ++i) {
            std::cout << "  " << (i + 1U) << ". " << options[i].label << '\n';
        }
        std::cout << "Choose, q to quit/concede: ";
        std::string input;
        if (!std::getline(std::cin, input)) {
            break;
        }
        if (input == "q" || input == "quit" || input == "concede") {
            std::cout << "Conceded.\n";
            return 0;
        }
        const auto choice = static_cast<std::size_t>(std::strtoull(input.c_str(), nullptr, 10));
        if (choice == 0U || choice > options.size()) {
            continue;
        }
        done = Game::step(state, options[choice - 1U].action);
    }

    render(state, kHuman);
    std::cout << "Game over. P1 " << score(state, 0) << " - P2 " << score(state, 1) << '\n';
    return 0;
}
