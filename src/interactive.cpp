#include "engine/game_runner.h"
#include "engine/action_space.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

// ─── Display helpers ────────────────────────────────────────────────

static void print_divider() {
    std::cout << "────────────────────────────────────────────────────\n";
}

static void print_supply(const GameState& state) {
    std::cout << "  SUPPLY\n";
    std::cout << "  Treasures: ";
    for (auto& n : {"Copper", "Silver", "Gold"})
        std::cout << n << "(" << state.get_supply().count(n) << ") ";
    std::cout << "\n  Victory:   ";
    for (auto& n : {"Estate", "Duchy", "Province"})
        std::cout << n << "(" << state.get_supply().count(n) << ") ";
    std::cout << "\n  Curse:     Curse(" << state.get_supply().count("Curse") << ")\n";

    std::cout << "  Kingdom:   ";
    int col = 0;
    for (const auto& pile_name : state.get_supply().all_pile_names()) {
        if (pile_name == "Copper" || pile_name == "Silver" || pile_name == "Gold" ||
            pile_name == "Estate" || pile_name == "Duchy" || pile_name == "Province" ||
            pile_name == "Curse") continue;

        const Card* card = CardRegistry::get(pile_name);
        int cost = card ? card->cost : 0;
        int count = state.get_supply().count(pile_name);
        std::cout << pile_name << "(" << cost << "$,x" << count << ") ";
        col++;
        if (col % 5 == 0) std::cout << "\n             ";
    }
    std::cout << "\n";
}

static void print_hand(const GameState& state, int pid, const std::string& label) {
    const Player& player = state.get_player(pid);
    const auto& hand = player.get_hand();
    std::cout << "  " << label << " hand [" << hand.size() << "]: ";
    for (int i = 0; i < static_cast<int>(hand.size()); i++) {
        const Card* card = state.card_def(hand[i]);
        std::string name = card ? card->name : "???";
        std::string types;
        if (card) {
            if (card->is_action()) types += "A";
            if (card->is_treasure()) types += "T";
            if (card->is_victory()) types += "V";
            if (card->is_curse()) types += "C";
            if (card->is_attack()) types += "!";
            if (card->is_reaction()) types += "R";
        }
        std::cout << name;
        if (!types.empty()) std::cout << "[" << types << "]";
        if (i < static_cast<int>(hand.size()) - 1) std::cout << ", ";
    }
    std::cout << "\n";
}

static void print_in_play(const GameState& state, int pid, const std::string& label) {
    const auto& in_play = state.get_player(pid).get_in_play();
    if (!in_play.empty()) {
        std::cout << "  " << label << " in play: ";
        for (int i = 0; i < static_cast<int>(in_play.size()); i++) {
            std::cout << state.card_name(in_play[i]);
            if (i < static_cast<int>(in_play.size()) - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

static void print_trash(const GameState& state) {
    const auto& trash = state.get_trash();
    if (!trash.empty()) {
        std::cout << "  Trash [" << trash.size() << "]: ";
        for (int i = 0; i < static_cast<int>(trash.size()); i++) {
            std::cout << state.card_name(trash[i]);
            if (i < static_cast<int>(trash.size()) - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

static void print_full_state(const GameState& state, int active_pid) {
    std::cout << "\n";
    print_divider();
    print_supply(state);
    print_trash(state);
    std::cout << "\n";

    // Your hand (full)
    print_hand(state, 0, "Your");
    print_in_play(state, 0, "Your");
    std::cout << "  You: Deck(" << state.get_player(0).deck_size()
              << ") Discard(" << state.get_player(0).discard_size() << ")\n";

    // Bot summary
    const Player& bot = state.get_player(1);
    std::cout << "  Bot: Hand(" << bot.hand_size()
              << ") Deck(" << bot.deck_size()
              << ") Discard(" << bot.discard_size() << ")\n";
    print_in_play(state, 1, "Bot");

    std::cout << "\n";
    std::cout << "  Actions: " << state.actions()
              << "  Buys: " << state.buys()
              << "  Coins: " << state.coins() << "\n";
    print_divider();
}

static int prompt_choice(const std::string& prompt, int min_val, int max_val) {
    while (true) {
        std::cout << prompt;
        if (min_val == max_val) {
            std::cout << " (auto " << min_val << "): ";
            return min_val;
        }
        std::cout << " [" << min_val << "-" << max_val << "]: ";
        std::string line;
        if (!std::getline(std::cin, line)) return min_val;
        if (line.empty()) continue;
        try {
            int val = std::stoi(line);
            if (val >= min_val && val <= max_val) return val;
        } catch (...) {}
        std::cout << "  Invalid input.\n";
    }
}

// ─── HumanAgent ─────────────────────────────────────────────────────

class HumanAgent : public Agent {
public:
    explicit HumanAgent(int player_id) : pid_(player_id) {}

    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override {
        // Always show full state
        print_full_state(state, dp.player_id);

        // Sub-decisions during card resolution
        if (dp.type == DecisionType::CHOOSE_CARDS_IN_HAND ||
            dp.type == DecisionType::CHOOSE_OPTION ||
            dp.type == DecisionType::REACT) {

            switch (dp.sub_choice_type) {
                case ChoiceType::DISCARD:
                    std::cout << "  Choose card(s) to DISCARD";
                    break;
                case ChoiceType::TRASH:
                    std::cout << "  Choose card(s) to TRASH";
                    break;
                case ChoiceType::GAIN:
                    std::cout << "  Choose a card to GAIN";
                    break;
                case ChoiceType::TOPDECK:
                    std::cout << "  Choose card to put on top of DECK";
                    break;
                case ChoiceType::YES_NO:
                    std::cout << "  Choose";
                    break;
                case ChoiceType::PLAY_CARD:
                    std::cout << "  Choose a card to PLAY";
                    break;
                case ChoiceType::REVEAL:
                    std::cout << "  Choose a card to REVEAL";
                    break;
                case ChoiceType::MULTI_FATE:
                    std::cout << "  Choose fate for this card";
                    break;
                case ChoiceType::ORDER:
                    std::cout << "  Choose which card goes on TOP";
                    break;
                case ChoiceType::SELECT_FROM_DISCARD:
                    std::cout << "  Choose a card from your discard pile";
                    break;
                default:
                    std::cout << "  Make a choice";
                    break;
            }

            if (dp.min_choices == dp.max_choices && dp.min_choices > 1) {
                std::cout << " (pick exactly " << dp.min_choices << ")";
            } else if (dp.min_choices != dp.max_choices) {
                std::cout << " (pick " << dp.min_choices << "-" << dp.max_choices << ")";
            }
            std::cout << "\n";

            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                const auto& opt = dp.options[i];
                std::cout << "    " << i << ") ";
                if (opt.card_id >= 0) {
                    const Card* card = state.card_def(opt.card_id);
                    std::cout << state.card_name(opt.card_id);
                    if (card && card->cost > 0) std::cout << " (" << card->cost << "$)";
                } else if (!opt.label.empty()) {
                    std::cout << opt.label;
                } else {
                    std::cout << "Option " << i;
                }
                std::cout << "\n";
            }

            if (dp.max_choices <= 1) {
                int choice = prompt_choice("  >", 0, static_cast<int>(dp.options.size()) - 1);
                return {choice};
            } else {
                while (true) {
                    std::cout << "  Enter " << dp.min_choices << " indices separated by spaces: ";
                    std::string line;
                    if (!std::getline(std::cin, line)) return {};
                    std::vector<int> result;
                    std::istringstream iss(line);
                    int val;
                    while (iss >> val) {
                        if (val >= 0 && val < static_cast<int>(dp.options.size()))
                            result.push_back(val);
                    }
                    int n = static_cast<int>(result.size());
                    if (n >= dp.min_choices && n <= dp.max_choices) return result;
                    std::cout << "  Must pick ";
                    if (dp.min_choices == dp.max_choices)
                        std::cout << "exactly " << dp.min_choices;
                    else
                        std::cout << dp.min_choices << "-" << dp.max_choices;
                    std::cout << " (you picked " << n << ").\n";
                }
            }
        }

        // Top-level phase decisions
        std::cout << "  YOUR TURN  |  Turn " << state.turn_number() << "\n";

        switch (dp.type) {
            case DecisionType::PLAY_ACTION:
                std::cout << "  === ACTION PHASE ===\n";
                break;
            case DecisionType::PLAY_TREASURE:
                std::cout << "  === TREASURE PHASE ===\n";
                break;
            case DecisionType::BUY_CARD:
                std::cout << "  === BUY PHASE (" << state.coins() << " coins, "
                          << state.buys() << " buys) ===\n";
                break;
            default:
                std::cout << "  === DECISION ===\n";
                break;
        }

        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            const auto& opt = dp.options[i];
            std::cout << "    " << i << ") " << opt.label;
            if (!opt.card_name.empty() && !opt.is_pass) {
                const Card* card = CardRegistry::get(opt.card_name);
                if (card) {
                    std::cout << " [" << card->cost << "$";
                    if (!card->text.empty()) std::cout << " | " << card->text;
                    std::cout << "]";
                }
            }
            std::cout << "\n";
        }

        int choice = prompt_choice("  >", 0, static_cast<int>(dp.options.size()) - 1);
        return {choice};
    }

private:
    int pid_;
};

// ─── Main ───────────────────────────────────────────────────────────

int main() {
    std::cout << R"(
  ╔══════════════════════════════════════╗
  ║       DOMINION  —  Base Set         ║
  ║    You (P0) vs. Engine Bot (P1)      ║
  ╚══════════════════════════════════════╝
)" << "\n";

    // All 26 base kingdom cards
    std::vector<std::string> all_kingdom = {
        "Cellar", "Chapel", "Moat", "Harbinger", "Merchant",
        "Vassal", "Village", "Workshop", "Bureaucrat", "Gardens",
        "Laboratory", "Market", "Militia", "Moneylender", "Poacher",
        "Remodel", "Smithy", "Throne Room", "Bandit", "Council Room",
        "Festival", "Library", "Mine", "Sentry", "Witch", "Artisan"
    };

    // Shuffle and pick 10
    auto seed = static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::shuffle(all_kingdom.begin(), all_kingdom.end(), rng);
    std::vector<std::string> kingdom(all_kingdom.begin(), all_kingdom.begin() + 10);
    std::sort(kingdom.begin(), kingdom.end());

    std::cout << "  Kingdom: ";
    for (const auto& k : kingdom) std::cout << k << "  ";
    std::cout << "\n\n";

    BaseCards::register_all();
    BaseKingdom::register_all();
    GameRunner runner(2, kingdom);

    runner.set_observer([](const std::string& msg) {
        std::cout << msg << "\n";
    });

    HumanAgent human(0);
    EngineBot bot;
    std::vector<Agent*> agents = {&human, &bot};

    auto result = runner.run(agents);

    std::cout << "\n";
    print_divider();
    std::cout << "  GAME OVER!\n\n";
    std::cout << "  You:        " << result.scores[0] << " VP\n";
    std::cout << "  Engine Bot: " << result.scores[1] << " VP\n\n";
    if (result.winner == 0) std::cout << "  >>> YOU WIN! <<<\n";
    else if (result.winner == 1) std::cout << "  Engine Bot wins.\n";
    else std::cout << "  It's a tie!\n";
    std::cout << "  Total turns: " << result.total_turns << "\n";
    print_divider();

    return 0;
}
