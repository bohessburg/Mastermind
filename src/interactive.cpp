#include "engine/game_runner.h"
#include "engine/action_space.h"
#include "engine/card_ids.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
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

    auto discard_top = [&](int pid) -> std::string {
        const auto& d = state.get_player(pid).get_discard();
        if (d.empty()) return "empty";
        return state.card_name(d.back());
    };

    // Your hand (full)
    print_hand(state, 0, "Your");
    print_in_play(state, 0, "Your");
    std::cout << "  You: Deck(" << state.get_player(0).deck_size()
              << ") Discard(" << state.get_player(0).discard_size()
              << ", top: " << discard_top(0) << ")\n";

    // Bot summary
    const Player& bot = state.get_player(1);
    std::cout << "  Bot: Hand(" << bot.hand_size()
              << ") Deck(" << bot.deck_size()
              << ") Discard(" << bot.discard_size()
              << ", top: " << discard_top(1) << ")\n";
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
                    std::cout << "  Choose fate for ";
                    if (!dp.source_card.empty())
                        std::cout << dp.source_card;
                    else
                        std::cout << "this card";
                    break;
                case ChoiceType::ORDER:
                    std::cout << "  Choose which card to put on TOP of your deck";
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
                if (dp.min_choices == 0) {
                    // Optional single-select: add a skip option
                    std::cout << "    s) Skip\n";
                    while (true) {
                        std::cout << "  > [0-" << dp.options.size() - 1 << " or s]: ";
                        std::string line;
                        if (!std::getline(std::cin, line)) return {};
                        if (line == "s" || line == "S") return {};
                        try {
                            int val = std::stoi(line);
                            if (val >= 0 && val < static_cast<int>(dp.options.size()))
                                return {val};
                        } catch (...) {}
                        std::cout << "  Invalid input.\n";
                    }
                }
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
            // Construct label from def_id if not populated
            std::string display_label = opt.label;
            if (display_label.empty()) {
                if (opt.is_play_all) {
                    display_label = "Play all Treasures";
                } else if (opt.is_pass) {
                    display_label = (dp.type == DecisionType::PLAY_ACTION) ? "End Actions" :
                                    (dp.type == DecisionType::PLAY_TREASURE) ? "Stop playing Treasures" :
                                    "End Buys";
                } else if (opt.def_id >= 0) {
                    const Card* c = CardRegistry::get(opt.def_id);
                    std::string prefix = (dp.type == DecisionType::BUY_CARD) ? "Buy " : "Play ";
                    display_label = prefix + (c ? c->name : "???");
                }
            }
            std::cout << "    " << i << ") " << display_label;
            if (opt.def_id >= 0 && !opt.is_pass) {
                const Card* card = CardRegistry::get(opt.def_id);
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
        // Level 1 (Base Set)
        "Cellar", "Chapel", "Moat", "Harbinger", "Merchant",
        "Vassal", "Village", "Workshop", "Bureaucrat", "Gardens",
        "Laboratory", "Market", "Militia", "Moneylender", "Poacher",
        "Remodel", "Smithy", "Throne Room", "Bandit", "Council Room",
        "Festival", "Library", "Mine", "Sentry", "Witch", "Artisan",
        // Level 2
        "Worker's Village", "Courtyard", "Hamlet", "Lookout", "Swindler", "Scholar"
    };

    BaseCards::register_all();
    Level1Cards::register_all();
    CardIds::init();

    // Kingdom selection
    std::cout << "  Kingdom setup:\n";
    std::cout << "    r) Random 10 cards\n";
    std::cout << "    p) Pick cards manually\n";
    std::cout << "    a) Use all 26 cards (NOT standard Dominion)\n";
    std::cout << "  > ";

    std::vector<std::string> kingdom;
    std::string mode;
    std::getline(std::cin, mode);

    if (mode == "a" || mode == "A") {
        kingdom = all_kingdom;
    } else if (mode == "p" || mode == "P") {
        std::cout << "\n  Available cards:\n";
        for (int i = 0; i < static_cast<int>(all_kingdom.size()); i++) {
            const Card* card = CardRegistry::get(all_kingdom[i]);
            std::cout << "    " << std::setw(2) << i << ") " << all_kingdom[i];
            if (card) std::cout << " (" << card->cost << "$ | " << card->text << ")";
            std::cout << "\n";
        }
        std::cout << "\n  Enter card numbers separated by spaces (pick 10):\n  > ";
        std::string line;
        std::getline(std::cin, line);
        std::istringstream iss(line);
        int val;
        while (iss >> val) {
            if (val >= 0 && val < static_cast<int>(all_kingdom.size())) {
                kingdom.push_back(all_kingdom[val]);
            }
        }
        if (kingdom.empty()) {
            std::cout << "  No valid cards selected, using random.\n";
            mode = "r";
        }
    }

    if (mode == "r" || mode == "R" || kingdom.empty()) {
        auto seed = static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        std::mt19937 rng(seed);
        std::shuffle(all_kingdom.begin(), all_kingdom.end(), rng);
        kingdom.assign(all_kingdom.begin(), all_kingdom.begin() + 10);
    }

    std::sort(kingdom.begin(), kingdom.end());
    std::cout << "\n  Kingdom: ";
    for (const auto& k : kingdom) std::cout << k << "  ";
    std::cout << "\n\n";

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
