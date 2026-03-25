#include "game_runner.h"
#include "rules.h"

#include <algorithm>
#include <sstream>

// --- RandomAgent ---

RandomAgent::RandomAgent(uint64_t seed) : rng_(seed) {}

std::vector<int> RandomAgent::decide(const DecisionPoint& dp, const GameState& /*state*/) {
    if (dp.options.empty()) return {};
    std::uniform_int_distribution<int> dist(0, static_cast<int>(dp.options.size()) - 1);
    return {dist(rng_)};
}

// --- BigMoneyAgent ---

// Optimized Big Money based on wiki.dominionstrategy.com/index.php/Money_strategies
// Buy rules from "Big Money optimized" section:
//   $8: Province (unless very early with no Gold and <5 Silvers → Gold)
//   $6-7: Gold (unless ≤4 Provinces left → Duchy)
//   $5: Silver (unless ≤5 Provinces left → Duchy)
//   $3-4: Silver (unless ≤2 Provinces left → Estate)
//   $2: Estate only if ≤3 Provinces left, else nothing
std::vector<int> BigMoneyAgent::decide(const DecisionPoint& dp, const GameState& state) {
    if (dp.options.empty()) return {};

    switch (dp.type) {
        case DecisionType::PLAY_ACTION: {
            // Pure Big Money: skip all actions
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }
        case DecisionType::PLAY_TREASURE: {
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].label == "Play all Treasures") return {i};
            }
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (!dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }
        case DecisionType::BUY_CARD: {
            int provinces_left = state.get_supply().count("Province");
            int coins = state.coins();

            auto find_option = [&](const std::string& name) -> int {
                for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                    if (dp.options[i].card_name == name) return i;
                }
                return -1;
            };

            // Count Golds and Silvers the player owns (all zones)
            int pid = state.current_player_id();
            auto all = state.get_player(pid).all_cards();
            int gold_count = 0, silver_count = 0;
            for (int cid : all) {
                const std::string& n = state.card_name(cid);
                if (n == "Gold") gold_count++;
                else if (n == "Silver") silver_count++;
            }

            if (coins >= 8) {
                // Early game exception: if no Gold and <5 Silvers, buy Gold instead
                if (gold_count == 0 && silver_count < 5) {
                    int idx = find_option("Gold");
                    if (idx >= 0) return {idx};
                }
                int idx = find_option("Province");
                if (idx >= 0) return {idx};
            }
            if (coins >= 6) {
                // Endgame: Duchy if ≤4 Provinces left
                if (provinces_left <= 4) {
                    int idx = find_option("Duchy");
                    if (idx >= 0) return {idx};
                }
                int idx = find_option("Gold");
                if (idx >= 0) return {idx};
            }
            if (coins == 5) {
                // Endgame: Duchy if ≤5 Provinces left
                if (provinces_left <= 5) {
                    int idx = find_option("Duchy");
                    if (idx >= 0) return {idx};
                }
                int idx = find_option("Silver");
                if (idx >= 0) return {idx};
            }
            if (coins >= 3) {
                // Endgame: Estate if ≤2 Provinces left
                if (provinces_left <= 2) {
                    int idx = find_option("Estate");
                    if (idx >= 0) return {idx};
                }
                int idx = find_option("Silver");
                if (idx >= 0) return {idx};
            }
            if (coins == 2) {
                // Estate only if ≤3 Provinces left
                if (provinces_left <= 3) {
                    int idx = find_option("Estate");
                    if (idx >= 0) return {idx};
                }
            }
            // $0-1 or nothing worth buying: pass
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }
        default: {
            std::vector<int> result;
            int needed = std::max(1, dp.min_choices);
            for (int i = 0; i < static_cast<int>(dp.options.size()) && static_cast<int>(result.size()) < needed; i++) {
                if (!dp.options[i].is_pass) result.push_back(i);
            }
            if (result.empty()) return {0};
            return result;
        }
    }
}

// --- HeuristicAgent ---

// Action play priority: villages first (need +Actions to play more cards),
// then cantrips (+1 Action, safe to chain), then strong terminals.
static int action_priority(const std::string& name) {
    // Tier 0: Villages — give +2 Actions, play these first
    if (name == "Village")    return 0;
    if (name == "Festival")   return 1;

    // Tier 1: Cantrips — give +1 Action, safe to chain
    if (name == "Laboratory") return 10;
    if (name == "Market")     return 11;
    if (name == "Sentry")     return 12;
    if (name == "Poacher")    return 13;
    if (name == "Merchant")   return 14;
    if (name == "Harbinger")  return 15;
    if (name == "Cellar")     return 16;

    // Tier 2: Throne Room — play before a terminal to double it
    if (name == "Throne Room") return 20;

    // Tier 3: Terminals — these consume your last action
    if (name == "Witch")       return 30; // strong attack + draw
    if (name == "Council Room") return 31;
    if (name == "Smithy")      return 32;
    if (name == "Militia")     return 33;
    if (name == "Library")     return 34;
    if (name == "Mine")        return 35;
    if (name == "Remodel")     return 36;
    if (name == "Moneylender") return 37;
    if (name == "Chapel")      return 38;
    if (name == "Artisan")     return 39;
    if (name == "Bandit")      return 40;
    if (name == "Bureaucrat")  return 41;
    if (name == "Workshop")    return 42;
    if (name == "Moat")        return 43;
    if (name == "Vassal")      return 44;

    return 50; // unknown cards
}

// Buy priority: lower = buy first when costs are equal.
// Treasures are inserted at reasonable positions among the actions.
static int buy_priority(const std::string& name) {
    // $8
    if (name == "Province")    return 0;

    // $6
    if (name == "Gold")        return 10;
    if (name == "Artisan")     return 11;

    // $5 actions
    if (name == "Witch")        return 20;
    if (name == "Laboratory")   return 21;
    if (name == "Festival")     return 22;
    if (name == "Market")       return 23;
    if (name == "Sentry")       return 24;
    if (name == "Council Room") return 25;
    if (name == "Mine")         return 26;
    if (name == "Library")      return 27;
    if (name == "Duchy")        return 28; // late game only (filtered separately)

    // $4
    if (name == "Militia")      return 30;
    if (name == "Smithy")       return 31;
    if (name == "Throne Room")  return 32;
    if (name == "Remodel")      return 33;
    if (name == "Moneylender")  return 34;
    if (name == "Poacher")      return 35;
    if (name == "Bureaucrat")   return 36;
    if (name == "Gardens")      return 37;

    // $3
    if (name == "Silver")       return 40;
    if (name == "Village")      return 41;
    if (name == "Merchant")     return 42;
    if (name == "Workshop")     return 43;
    if (name == "Harbinger")    return 44;
    if (name == "Vassal")       return 45;

    // $2
    if (name == "Chapel")       return 50;
    if (name == "Moat")         return 51;
    if (name == "Cellar")       return 52;
    if (name == "Estate")       return 53; // late game only (filtered separately)

    // Never buy
    if (name == "Copper")       return 900;
    if (name == "Curse")        return 999;

    return 100; // unknown cards
}

// For discard decisions: prefer discarding junk (victory, curse) over useful cards
static int discard_priority(const Card* card) {
    if (!card) return 50;
    if (card->is_curse()) return 0;     // discard curses first
    if (card->name == "Estate") return 1;
    if (card->name == "Duchy") return 2;
    if (card->name == "Province") return 3; // keep provinces if possible but still victory
    if (card->name == "Copper") return 10;
    if (card->is_victory()) return 5;    // other victory cards
    if (card->is_treasure()) return 20;  // keep treasures
    if (card->is_action()) return 15;    // keep actions
    return 25;
}

// For trash decisions: prefer trashing junk
static int trash_priority(const Card* card) {
    if (!card) return 50;
    if (card->is_curse()) return 0;
    if (card->name == "Estate") return 1;
    if (card->name == "Copper") return 2;
    return 50; // don't want to trash other stuff
}

std::vector<int> HeuristicAgent::decide(const DecisionPoint& dp, const GameState& state) {
    if (dp.options.empty()) return {};

    switch (dp.type) {
        case DecisionType::PLAY_ACTION: {
            // Pick the highest-priority action card
            int best_idx = -1;
            int best_prio = 999;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) continue;
                int prio = action_priority(dp.options[i].card_name);
                if (prio < best_prio) {
                    best_prio = prio;
                    best_idx = i;
                }
            }
            if (best_idx >= 0) return {best_idx};
            // Pass if nothing to play
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        case DecisionType::PLAY_TREASURE: {
            // Play all treasures
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].label == "Play all Treasures") return {i};
            }
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (!dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        case DecisionType::BUY_CARD: {
            // Buy the most expensive card, breaking ties by buy_priority.
            int provinces_left = state.get_supply().count("Province");
            int best_idx = -1;
            int best_cost = -1;
            int best_prio = 999;

            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) continue;
                const std::string& name = dp.options[i].card_name;
                const Card* card = CardRegistry::get(name);
                if (!card) continue;
                // Never buy Curse or Copper
                if (name == "Curse" || name == "Copper") continue;
                // Only buy Estate late game
                if (name == "Estate" && provinces_left > 2) continue;
                // Only buy Duchy when provinces running low
                if (name == "Duchy" && provinces_left > 5) continue;

                int cost = card->cost;
                int prio = buy_priority(name);
                if (cost > best_cost || (cost == best_cost && prio < best_prio)) {
                    best_cost = cost;
                    best_prio = prio;
                    best_idx = i;
                }
            }
            if (best_idx >= 0) return {best_idx};
            // Pass
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        default: {
            // Sub-decisions: use heuristics based on choice type
            switch (dp.sub_choice_type) {
                case ChoiceType::DISCARD: {
                    // Discard the worst cards (victory, curse, copper)
                    std::vector<std::pair<int, int>> scored; // (priority, index)
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        scored.push_back({discard_priority(card), i});
                    }
                    std::sort(scored.begin(), scored.end());
                    std::vector<int> result;
                    int needed = std::max(1, dp.min_choices);
                    for (int i = 0; i < needed && i < static_cast<int>(scored.size()); i++) {
                        result.push_back(scored[i].second);
                    }
                    return result;
                }

                case ChoiceType::TRASH: {
                    // Trash the worst cards, but only if they're actually junk
                    std::vector<std::pair<int, int>> scored;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        scored.push_back({trash_priority(card), i});
                    }
                    std::sort(scored.begin(), scored.end());
                    std::vector<int> result;
                    int limit = dp.max_choices;
                    for (int i = 0; i < limit && i < static_cast<int>(scored.size()); i++) {
                        // Only trash if priority < 50 (actual junk)
                        if (scored[i].first < 50 || static_cast<int>(result.size()) < dp.min_choices) {
                            result.push_back(scored[i].second);
                        }
                    }
                    // Ensure minimum
                    while (static_cast<int>(result.size()) < dp.min_choices &&
                           static_cast<int>(result.size()) < static_cast<int>(dp.options.size())) {
                        bool found = false;
                        for (int i = 0; i < static_cast<int>(scored.size()); i++) {
                            bool already = false;
                            for (int r : result) if (r == scored[i].second) { already = true; break; }
                            if (!already) { result.push_back(scored[i].second); found = true; break; }
                        }
                        if (!found) break;
                    }
                    return result;
                }

                case ChoiceType::GAIN: {
                    // Gain the most expensive option
                    int best_idx = 0;
                    int best_cost = -1;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        int cost = card ? card->cost : 0;
                        if (cost > best_cost) {
                            best_cost = cost;
                            best_idx = i;
                        }
                    }
                    return {best_idx};
                }

                case ChoiceType::YES_NO: {
                    // Generally say yes (reveal Moat, trash Copper with Moneylender, etc.)
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        if (dp.options[i].label == "Yes") return {i};
                    }
                    return {static_cast<int>(dp.options.size()) - 1};
                }

                case ChoiceType::PLAY_CARD: {
                    // Throne Room: pick the best action by priority
                    int best_idx = 0;
                    int best_prio = 999;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        if (card) {
                            int prio = action_priority(card->name);
                            if (prio < best_prio) {
                                best_prio = prio;
                                best_idx = i;
                            }
                        }
                    }
                    return {best_idx};
                }

                case ChoiceType::MULTI_FATE: {
                    // Sentry: trash curses/estates/coppers, keep everything else
                    // Options: 0=keep, 1=discard, 2=trash
                    // The card_id in the DecisionPoint won't help here since
                    // MULTI_FATE options are {0,1,2} not card references.
                    // Default: keep (0)
                    return {0};
                }

                default: {
                    // Fallback: pick first min_choices options
                    std::vector<int> result;
                    int needed = std::max(1, dp.min_choices);
                    for (int i = 0; i < static_cast<int>(dp.options.size()) &&
                         static_cast<int>(result.size()) < needed; i++) {
                        result.push_back(i);
                    }
                    return result;
                }
            }
        }
    }
}

// --- EngineBot ---
// Build phase: buy engine components, avoid VP/Gold.
// Green phase: switch to aggressive VP buying.
// Transition when engine is "ready" (has components + thin deck or enough draw).

// Engine buy priority during build phase (lower = buy first)
static int engine_build_priority(const std::string& name) {
    // Trashing — top priority early
    if (name == "Chapel")       return 0;

    // Cantrip draw — always safe, no terminal collision
    if (name == "Laboratory")   return 10;
    if (name == "Market")       return 11;
    if (name == "Festival")     return 12;
    if (name == "Sentry")       return 13;
    if (name == "Merchant")     return 14;

    // Villages — needed to support terminals
    if (name == "Village")      return 20;

    // Terminal draw / attacks
    if (name == "Witch")        return 30;
    if (name == "Smithy")       return 31;
    if (name == "Council Room") return 32;
    if (name == "Militia")      return 33;
    if (name == "Moat")         return 34;

    // Utility
    if (name == "Cellar")       return 40;
    if (name == "Harbinger")    return 41;
    if (name == "Throne Room")  return 42;
    if (name == "Remodel")      return 43;
    if (name == "Poacher")      return 44;
    if (name == "Workshop")     return 45;
    if (name == "Artisan")      return 46;
    if (name == "Mine")         return 47;
    if (name == "Library")      return 48;
    if (name == "Vassal")       return 49;
    if (name == "Moneylender")  return 50;
    if (name == "Bandit")       return 51;
    if (name == "Bureaucrat")   return 52;

    // Treasure — only as fallback
    if (name == "Silver")       return 80;
    if (name == "Gold")         return 81;

    // Never during build
    if (name == "Province")     return 900;
    if (name == "Duchy")        return 901;
    if (name == "Estate")       return 902;
    if (name == "Gardens")      return 903;
    if (name == "Curse")        return 999;
    if (name == "Copper")       return 999;

    return 100;
}

struct DeckProfile {
    int villages;       // Village, Festival
    int terminal_draw;  // Smithy, Council Room, Witch, Moat, Library
    int cantrips;       // Laboratory, Market, Sentry, Merchant, Harbinger, Poacher, Cellar
    int chapels;
    int total_actions;
    int total_cards;
    int treasures;      // Copper + Silver + Gold
    int junk;           // Estate + Curse + Copper
};

static DeckProfile analyze_deck(const GameState& state, int pid) {
    DeckProfile p = {};
    auto all = state.get_player(pid).all_cards();
    p.total_cards = static_cast<int>(all.size());
    for (int cid : all) {
        const std::string& name = state.card_name(cid);
        const Card* card = state.card_def(cid);
        if (!card) continue;

        if (card->is_action()) p.total_actions++;
        if (card->is_treasure()) p.treasures++;

        if (name == "Village" || name == "Festival") p.villages++;
        if (name == "Smithy" || name == "Council Room" || name == "Witch" ||
            name == "Moat" || name == "Library") p.terminal_draw++;
        if (name == "Laboratory" || name == "Market" || name == "Sentry" ||
            name == "Merchant" || name == "Harbinger" || name == "Poacher" ||
            name == "Cellar") p.cantrips++;
        if (name == "Chapel") p.chapels++;
        if (name == "Copper" || name == "Estate" || name == "Curse") p.junk++;
    }
    return p;
}

static bool should_green(const GameState& state, int pid, int turn) {
    auto prof = analyze_deck(state, pid);

    // Failsafe: don't build forever
    if (turn > 12) return true;

    // Have Chapel and deck is thin (few junk cards left)
    if (prof.chapels >= 1 && prof.junk <= 3 && prof.total_actions >= 2) return true;

    // Have enough village + draw to reliably hit $8
    if (prof.villages >= 2 && prof.terminal_draw >= 2) return true;
    if (prof.villages >= 1 && prof.terminal_draw >= 1 && prof.cantrips >= 2) return true;

    // Lots of cantrips can substitute for village+draw
    if (prof.cantrips >= 4) return true;

    return false;
}

std::vector<int> EngineBot::decide(const DecisionPoint& dp, const GameState& state) {
    if (dp.options.empty()) return {};

    int pid = dp.player_id;
    int provinces_left = state.get_supply().count("Province");

    switch (dp.type) {
        case DecisionType::PLAY_ACTION: {
            // Same play priority as HeuristicAgent: villages → cantrips → terminals
            int best_idx = -1;
            int best_prio = 999;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) continue;
                int prio = action_priority(dp.options[i].card_name);
                if (prio < best_prio) {
                    best_prio = prio;
                    best_idx = i;
                }
            }
            if (best_idx >= 0) return {best_idx};
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        case DecisionType::PLAY_TREASURE: {
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].label == "Play all Treasures") return {i};
            }
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (!dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        case DecisionType::BUY_CARD: {
            int coins = state.coins();
            int turn = state.turn_number();
            bool greening = should_green(state, pid, turn);

            auto find_option = [&](const std::string& name) -> int {
                for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                    if (dp.options[i].card_name == name) return i;
                }
                return -1;
            };

            if (greening) {
                // --- GREEN PHASE: same as optimized Big Money ---
                auto all = state.get_player(pid).all_cards();
                int gold_count = 0, silver_count = 0;
                for (int cid : all) {
                    const std::string& n = state.card_name(cid);
                    if (n == "Gold") gold_count++;
                    else if (n == "Silver") silver_count++;
                }

                if (coins >= 8) {
                    if (gold_count == 0 && silver_count < 5) {
                        int idx = find_option("Gold");
                        if (idx >= 0) return {idx};
                    }
                    int idx = find_option("Province");
                    if (idx >= 0) return {idx};
                }
                if (coins >= 6) {
                    if (provinces_left <= 4) {
                        int idx = find_option("Duchy");
                        if (idx >= 0) return {idx};
                    }
                    int idx = find_option("Gold");
                    if (idx >= 0) return {idx};
                }
                if (coins == 5) {
                    if (provinces_left <= 5) {
                        int idx = find_option("Duchy");
                        if (idx >= 0) return {idx};
                    }
                    // Still buy a Lab/Market if available during green (keep engine running)
                    for (auto& name : {"Laboratory", "Market", "Festival"}) {
                        int idx = find_option(name);
                        if (idx >= 0) return {idx};
                    }
                    int idx = find_option("Silver");
                    if (idx >= 0) return {idx};
                }
                if (coins >= 3) {
                    if (provinces_left <= 2) {
                        int idx = find_option("Estate");
                        if (idx >= 0) return {idx};
                    }
                    int idx = find_option("Silver");
                    if (idx >= 0) return {idx};
                }
                if (coins == 2 && provinces_left <= 3) {
                    int idx = find_option("Estate");
                    if (idx >= 0) return {idx};
                }
            } else {
                // --- BUILD PHASE: buy engine components ---
                auto prof = analyze_deck(state, pid);

                // Component limits
                auto is_limited = [&](const std::string& name) -> bool {
                    if (name == "Chapel" && prof.chapels >= 1) return true;
                    if ((name == "Smithy" || name == "Council Room" || name == "Witch" ||
                         name == "Moat" || name == "Library") && prof.terminal_draw >= 2) return true;
                    if ((name == "Village" || name == "Festival") && prof.villages >= 3) return true;
                    return false;
                };

                // Pick the best available engine component by priority
                int best_idx = -1;
                int best_prio = 999;
                for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                    if (dp.options[i].is_pass) continue;
                    const std::string& name = dp.options[i].card_name;
                    if (name.empty()) continue;

                    // Never buy VP or junk during build
                    if (name == "Province" || name == "Duchy" || name == "Estate" ||
                        name == "Gardens" || name == "Curse" || name == "Copper") continue;

                    // Respect component limits
                    if (is_limited(name)) continue;

                    // Don't buy Gold during build (prefer actions)
                    // Exception: if nothing else is available
                    int prio = engine_build_priority(name);

                    // If we have no village yet but have terminals, boost Village priority
                    if ((name == "Village" || name == "Festival") &&
                        prof.villages == 0 && prof.terminal_draw >= 1) {
                        prio = 5; // urgent
                    }

                    // If we have village but no terminal draw, boost draw priority
                    if ((name == "Smithy" || name == "Witch" || name == "Council Room") &&
                        prof.terminal_draw == 0 && prof.villages >= 1) {
                        prio = 5; // urgent
                    }

                    if (prio < best_prio) {
                        best_prio = prio;
                        best_idx = i;
                    }
                }
                if (best_idx >= 0) return {best_idx};
            }

            // Pass
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        default: {
            // Sub-decisions: same heuristics as HeuristicAgent
            switch (dp.sub_choice_type) {
                case ChoiceType::DISCARD: {
                    std::vector<std::pair<int, int>> scored;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        scored.push_back({discard_priority(card), i});
                    }
                    std::sort(scored.begin(), scored.end());
                    std::vector<int> result;
                    int needed = std::max(1, dp.min_choices);
                    for (int i = 0; i < needed && i < static_cast<int>(scored.size()); i++)
                        result.push_back(scored[i].second);
                    return result;
                }

                case ChoiceType::TRASH: {
                    // Engine bot trashes MORE aggressively: trash all junk up to max
                    std::vector<std::pair<int, int>> scored;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        scored.push_back({trash_priority(card), i});
                    }
                    std::sort(scored.begin(), scored.end());
                    std::vector<int> result;
                    int limit = dp.max_choices;
                    for (int i = 0; i < limit && i < static_cast<int>(scored.size()); i++) {
                        if (scored[i].first < 50 || static_cast<int>(result.size()) < dp.min_choices)
                            result.push_back(scored[i].second);
                    }
                    while (static_cast<int>(result.size()) < dp.min_choices &&
                           static_cast<int>(result.size()) < static_cast<int>(dp.options.size())) {
                        for (int i = 0; i < static_cast<int>(scored.size()); i++) {
                            bool already = false;
                            for (int r : result) if (r == scored[i].second) { already = true; break; }
                            if (!already) { result.push_back(scored[i].second); break; }
                        }
                    }
                    return result;
                }

                case ChoiceType::GAIN: {
                    // Gain the most expensive option
                    int best_idx = 0;
                    int best_cost = -1;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        int cost = card ? card->cost : 0;
                        if (cost > best_cost) { best_cost = cost; best_idx = i; }
                    }
                    return {best_idx};
                }

                case ChoiceType::YES_NO: {
                    // Always yes (reveal Moat, trash Copper, etc.)
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        if (dp.options[i].label == "Yes") return {i};
                    }
                    return {static_cast<int>(dp.options.size()) - 1};
                }

                case ChoiceType::PLAY_CARD: {
                    // Throne Room: pick best action by priority
                    int best_idx = 0;
                    int best_prio = 999;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        if (card) {
                            int prio = action_priority(card->name);
                            if (prio < best_prio) { best_prio = prio; best_idx = i; }
                        }
                    }
                    return {best_idx};
                }

                default: {
                    std::vector<int> result;
                    int needed = std::max(1, dp.min_choices);
                    for (int i = 0; i < static_cast<int>(dp.options.size()) &&
                         static_cast<int>(result.size()) < needed; i++)
                        result.push_back(i);
                    return result;
                }
            }
        }
    }
}

// --- GameRunner ---

GameRunner::GameRunner(int num_players, const std::vector<std::string>& kingdom_cards)
    : state_(num_players)
    , kingdom_cards_(kingdom_cards)
    , total_decisions_(0)
{
}

const GameState& GameRunner::state() const { return state_; }

void GameRunner::set_observer(GameObserver observer) {
    observer_ = std::move(observer);
}

void GameRunner::observe(const std::string& msg) {
    if (observer_) observer_(msg);
}

DecisionFn GameRunner::make_decision_fn() {
    return [this](int player_id, ChoiceType choice_type,
                  const std::vector<int>& options,
                  int min_choices, int max_choices) -> std::vector<int> {
        DecisionPoint dp;
        dp.player_id = player_id;
        dp.type = DecisionType::CHOOSE_CARDS_IN_HAND;
        dp.sub_choice_type = choice_type;
        dp.min_choices = min_choices;
        dp.max_choices = max_choices;

        const Player& player = state_.get_player(player_id);
        const auto& hand = player.get_hand();
        const auto& discard = player.get_discard();

        for (int i = 0; i < static_cast<int>(options.size()); i++) {
            ActionOption opt;
            opt.local_id = i;
            opt.card_id = -1;
            opt.is_pass = false;

            switch (choice_type) {
                // These pass hand indices — translate to card IDs
                case ChoiceType::DISCARD:
                case ChoiceType::TRASH:
                case ChoiceType::TOPDECK:
                case ChoiceType::PLAY_CARD:
                case ChoiceType::REVEAL: {
                    int idx = options[i];
                    opt.card_id = (idx < static_cast<int>(hand.size())) ? hand[idx] : -1;
                    break;
                }
                // Discard pile indices
                case ChoiceType::SELECT_FROM_DISCARD: {
                    int idx = options[i];
                    opt.card_id = (idx < static_cast<int>(discard.size())) ? discard[idx] : -1;
                    break;
                }
                // GAIN options are now top-card IDs from supply piles
                case ChoiceType::GAIN: {
                    opt.card_id = options[i];
                    break;
                }
                // Binary choice
                case ChoiceType::YES_NO: {
                    opt.label = (options[i] == 0) ? "No" : "Yes";
                    break;
                }
                // Per-card fate
                case ChoiceType::MULTI_FATE: {
                    if (options[i] == 0) opt.label = "Keep (put back on deck)";
                    else if (options[i] == 1) opt.label = "Discard";
                    else if (options[i] == 2) opt.label = "Trash";
                    else opt.label = "Option " + std::to_string(options[i]);
                    break;
                }
                // Ordering
                case ChoiceType::ORDER: {
                    opt.label = "Position " + std::to_string(options[i]);
                    break;
                }
                default: {
                    opt.card_id = options[i];
                    break;
                }
            }

            dp.options.push_back(opt);
        }

        total_decisions_++;

        // Ask the agent, then enforce the rules. Re-ask on violation.
        int num_opts = static_cast<int>(dp.options.size());
        while (true) {
            auto result = agents_[player_id]->decide(dp, state_);
            int n = static_cast<int>(result.size());

            // Validate count
            if (n < min_choices || n > max_choices) {
                observe("  [engine] Invalid: need " + std::to_string(min_choices) +
                        "-" + std::to_string(max_choices) + " choices, got " +
                        std::to_string(n) + ". Retrying.");
                continue;
            }

            // Validate indices in range and no duplicates
            bool valid = true;
            for (int idx : result) {
                if (idx < 0 || idx >= num_opts) { valid = false; break; }
            }
            if (valid) {
                for (int i = 0; i < n && valid; i++) {
                    for (int j = i + 1; j < n && valid; j++) {
                        if (result[i] == result[j]) valid = false;
                    }
                }
            }
            if (!valid) {
                observe("  [engine] Invalid indices. Retrying.");
                continue;
            }

            return result;
        }
    };
}

GameResult GameRunner::run(std::vector<Agent*> agents) {
    agents_ = agents;
    BaseCards::register_all();
    BaseKingdom::register_all();
    BaseCards::setup_supply(state_, kingdom_cards_);
    BaseCards::setup_starting_decks(state_);
    state_.start_game();
    total_decisions_ = 0;

    while (!state_.is_game_over()) {
        int pid = state_.current_player_id();
        observe("--- Player " + std::to_string(pid) + "'s turn (Turn " +
                std::to_string(state_.turn_number()) + ") ---");
        run_turn(pid);
    }

    agents_ = {};
    return {
        state_.calculate_scores(),
        state_.winner(),
        state_.turn_number(),
        total_decisions_
    };
}

void GameRunner::run_turn(int pid) {
    state_.start_turn();
    run_action_phase(pid);
    run_treasure_phase(pid);
    run_buy_phase(pid);
    state_.set_phase(Phase::CLEANUP);
    state_.advance_phase();
}

void GameRunner::run_action_phase(int pid) {
    Agent* agent = agents_[pid];
    while (state_.actions() > 0) {
        auto playable = Rules::playable_actions(state_, pid);
        if (playable.empty()) break;

        DecisionPoint dp = build_action_decision(state_);
        auto choice = agent->decide(dp, state_);
        total_decisions_++;
        if (choice.empty()) break;

        int chosen_idx = choice[0];
        if (chosen_idx >= static_cast<int>(dp.options.size()) || dp.options[chosen_idx].is_pass) break;

        int card_id = dp.options[chosen_idx].card_id;
        Player& player = state_.get_player(pid);
        int hand_idx = player.find_in_hand(card_id);
        if (hand_idx == -1) break;

        observe("  Player " + std::to_string(pid) + " plays " +
                dp.options[chosen_idx].card_name);

        state_.add_actions(-1);
        DecisionFn decide_fn = make_decision_fn();
        state_.play_card_from_hand(pid, hand_idx, decide_fn);
    }
}

void GameRunner::run_treasure_phase(int pid) {
    Agent* agent = agents_[pid];
    while (true) {
        auto treasures = Rules::playable_treasures(state_, pid);
        if (treasures.empty()) break;

        DecisionPoint dp = build_treasure_decision(state_);
        auto choice = agent->decide(dp, state_);
        total_decisions_++;
        if (choice.empty()) break;

        int chosen_idx = choice[0];
        if (chosen_idx >= static_cast<int>(dp.options.size()) || dp.options[chosen_idx].is_pass) break;

        // "Play all Treasures" shortcut
        if (dp.options[chosen_idx].label == "Play all Treasures") {
            Player& player = state_.get_player(pid);
            DecisionFn decide_fn = make_decision_fn();

            std::vector<int> treasure_indices;
            for (int i = 0; i < player.hand_size(); i++) {
                const Card* card = state_.card_def(player.get_hand()[i]);
                if (card && card->is_treasure()) treasure_indices.push_back(i);
            }

            // Build narration
            std::ostringstream oss;
            oss << "  Player " << pid << " plays all Treasures:";
            for (int i : treasure_indices) {
                oss << " " << state_.card_name(player.get_hand()[i]);
            }
            observe(oss.str());

            std::sort(treasure_indices.begin(), treasure_indices.end(), std::greater<int>());

            int merchant_count = state_.get_turn_flag("merchant_count");
            bool silver_triggered = state_.get_turn_flag("merchant_silver_triggered") != 0;

            for (int idx : treasure_indices) {
                int cid = player.get_hand()[idx];
                const Card* card = state_.card_def(cid);
                player.play_from_hand(idx);
                if (card && card->on_play) {
                    card->on_play(state_, pid, decide_fn);
                }
                if (card && card->name == "Silver" && merchant_count > 0 && !silver_triggered) {
                    state_.add_coins(merchant_count);
                    state_.set_turn_flag("merchant_silver_triggered", 1);
                    silver_triggered = true;
                }
            }
            break;
        }

        // Play a single treasure
        int card_id = dp.options[chosen_idx].card_id;
        Player& player = state_.get_player(pid);
        int hand_idx = player.find_in_hand(card_id);
        if (hand_idx == -1) break;

        const Card* card = state_.card_def(card_id);
        observe("  Player " + std::to_string(pid) + " plays " +
                (card ? card->name : "???"));

        DecisionFn decide_fn = make_decision_fn();
        player.play_from_hand(hand_idx);
        if (card && card->on_play) {
            card->on_play(state_, pid, decide_fn);
        }

        int merchant_count = state_.get_turn_flag("merchant_count");
        bool silver_triggered = state_.get_turn_flag("merchant_silver_triggered") != 0;
        if (card && card->name == "Silver" && merchant_count > 0 && !silver_triggered) {
            state_.add_coins(merchant_count);
            state_.set_turn_flag("merchant_silver_triggered", 1);
        }
    }

    observe("  Player " + std::to_string(pid) + " has " +
            std::to_string(state_.coins()) + " coins");
}

void GameRunner::run_buy_phase(int pid) {
    Agent* agent = agents_[pid];
    while (state_.buys() > 0) {
        DecisionPoint dp = build_buy_decision(state_);
        if (dp.options.size() <= 1) break;

        auto choice = agent->decide(dp, state_);
        total_decisions_++;
        if (choice.empty()) break;

        int chosen_idx = choice[0];
        if (chosen_idx >= static_cast<int>(dp.options.size()) || dp.options[chosen_idx].is_pass) break;

        const std::string& pile_name = dp.options[chosen_idx].card_name;
        const Card* card = CardRegistry::get(pile_name);
        if (!card) break;

        observe("  Player " + std::to_string(pid) + " buys " + pile_name);

        state_.add_coins(-card->cost);
        state_.add_buys(-1);
        state_.gain_card(pid, pile_name);
    }
}
