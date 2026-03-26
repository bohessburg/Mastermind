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

// ---BetterRandomAgent ---

// Play's randomly but filters moves to eliminate completely unreasonable turn ending, and buys provinces if possible.

BetterRandomAgent::BetterRandomAgent(uint64_t seed) : rng_(seed) {}

std::vector<int> BetterRandomAgent::decide(const DecisionPoint& dp, const GameState& /*state*/) {
    if (dp.options.empty()) return {};

    // Play all treasures at once when available
    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].label == "Play all Treasures") return {i};
        }
    }

    // During buy phase, always buy Province if we can afford it, otherwise buy random
    if (dp.type == DecisionType::BUY_CARD) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].card_name == "Province") return {i};
        }
    }

    // Multi-choice sub-decisions (e.g. "discard 2 cards"): pick min_choices random options
    if (dp.min_choices > 1) {
        std::vector<int> all_indices;
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            all_indices.push_back(i);
        }
        std::shuffle(all_indices.begin(), all_indices.end(), rng_);
        int pick = std::min(dp.min_choices, static_cast<int>(all_indices.size()));
        return {all_indices.begin(), all_indices.begin() + pick};
    }

    // Collect non-pass options
    std::vector<int> candidates;
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        if (!dp.options[i].is_pass) {
            candidates.push_back(i);
        }
    }

    // If everything is a pass, just pass
    if (candidates.empty()) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].is_pass) return {i};
        }
        return {0};
    }

    // Pick randomly among non-pass options
    std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
    return {candidates[dist(rng_)]};
}

// --- BigMoneyAgent ---

// Big Money: build treasure base first, then green.
// Phase 1 (building): Buy only money until total money value >= 16
//   (7 Copper + 3 Silver + 1 Gold = 7+6+3 = 16)
// Phase 2 (greening): Buy Province at $8, Gold at $6-7, Silver at $3-5.
//   Endgame duchy/estate dancing when provinces running low.
std::vector<int> BigMoneyAgent::decide(const DecisionPoint& dp, const GameState& state) {
    if (dp.options.empty()) return {};

    switch (dp.type) {
        case DecisionType::PLAY_ACTION: {
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

            // Count total money value in deck
            int pid = state.current_player_id();
            auto all = state.get_player(pid).all_cards();
            int total_money = 0;
            for (int cid : all) {
                const Card* c = state.card_def(cid);
                if (c && c->is_treasure()) total_money += c->coin_value;
            }

            bool greening = (total_money >= 16);

            if (greening) {
                // --- GREENING PHASE ---
                if (coins >= 8) {
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
                    int idx = find_option("Silver");
                    if (idx >= 0) return {idx};
                }
            } else {
                // --- BUILDING PHASE: only buy money ---
                if (coins >= 6) {
                    int idx = find_option("Gold");
                    if (idx >= 0) return {idx};
                }
                if (coins >= 3) {
                    int idx = find_option("Silver");
                    if (idx >= 0) return {idx};
                }
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
            // Penalize cards we already own copies of: priority *= (1 + copies)^2
            // This means: 0 copies = 1x, 1 = 4x, 2 = 9x, 3 = 16x penalty.
            int provinces_left = state.get_supply().count("Province");

            // Count copies of each card name in our deck
            int pid = dp.player_id;
            auto all = state.get_player(pid).all_cards();
            std::unordered_map<std::string, int> owned;
            for (int cid : all) {
                owned[state.card_name(cid)]++;
            }

            int best_idx = -1;
            int best_cost = -1;
            int best_prio = 999999;

            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) continue;
                const std::string& name = dp.options[i].card_name;
                const Card* card = CardRegistry::get(name);
                if (!card) continue;
                if (name == "Curse" || name == "Copper") continue;
                if (name == "Estate" && provinces_left > 2) continue;
                if (name == "Duchy" && provinces_left > 5) continue;

                int cost = card->cost;
                int base_prio = buy_priority(name);

                // Diminishing returns: penalize by (1 + copies)^2
                // Exception: treasures and VP cards don't get penalized
                int copies = owned.count(name) ? owned[name] : 0;
                int prio;
                if (card->is_treasure() || card->is_victory() || card->is_curse()) {
                    prio = base_prio;
                } else {
                    prio = base_prio * (1 + copies) * (1 + copies);
                }

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
// Kingdom-aware strategy selection:
//   FULL_ENGINE: has Village + terminal draw → build engine, then green
//   BM_PLUS_X:  has good terminal but no village → BigMoney + 1-2 key actions
//   PURE_BM:    nothing useful → fall back to pure BigMoney

enum class EngineStrategy { FULL_ENGINE, BM_PLUS_X, PURE_BM };

struct KingdomAnalysis {
    bool has_village;        // Village or Festival
    bool has_chapel;
    bool has_terminal_draw;  // Smithy, Council Room, Witch, Library, Moat
    bool has_cantrip_draw;   // Laboratory, Market
    bool has_witch;
    std::string best_terminal;  // best terminal to buy for BM+X
};

static KingdomAnalysis analyze_kingdom(const GameState& state) {
    KingdomAnalysis k = {};
    for (const auto& name : state.get_supply().all_pile_names()) {
        if (state.get_supply().is_pile_empty(name)) continue;
        if (name == "Village" || name == "Festival") k.has_village = true;
        if (name == "Chapel") k.has_chapel = true;
        if (name == "Witch") { k.has_witch = true; k.has_terminal_draw = true; }
        if (name == "Smithy" || name == "Council Room" || name == "Library" ||
            name == "Moat") k.has_terminal_draw = true;
        if (name == "Laboratory" || name == "Market") k.has_cantrip_draw = true;
    }
    // Pick best terminal for BM+X: Witch > Smithy > Council Room > Moat > Library
    for (auto& t : {"Witch", "Smithy", "Council Room", "Moat", "Library"}) {
        if (!state.get_supply().is_pile_empty(t)) { k.best_terminal = t; break; }
    }
    return k;
}

static EngineStrategy pick_strategy(const KingdomAnalysis& k) {
    // Full engine ONLY with Chapel to thin the deck. Without trashing,
    // Village/Smithy engines get bloated and lose to pure money.
    if (k.has_village && k.has_terminal_draw && k.has_chapel)
        return EngineStrategy::FULL_ENGINE;
    // BM+X: has a good terminal, chapel, or cantrip draw
    if (k.has_terminal_draw || k.has_chapel || k.has_cantrip_draw) return EngineStrategy::BM_PLUS_X;
    // Nothing useful
    return EngineStrategy::PURE_BM;
}

struct DeckProfile {
    int villages, terminal_draw, cantrips, chapels;
    int total_actions, total_cards, treasures, junk;
    int total_money;  // sum of coin_value of all treasures
    std::unordered_map<std::string, int> counts;
};

static DeckProfile analyze_deck(const GameState& state, int pid) {
    DeckProfile p = {};
    auto all = state.get_player(pid).all_cards();
    p.total_cards = static_cast<int>(all.size());
    for (int cid : all) {
        const std::string& name = state.card_name(cid);
        const Card* card = state.card_def(cid);
        if (!card) continue;

        p.counts[name]++;
        if (card->is_action()) p.total_actions++;
        if (card->is_treasure()) { p.treasures++; p.total_money += card->coin_value; }
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

// Engine build priority (lower = buy first)
static int engine_build_priority(const std::string& name) {
    if (name == "Chapel")       return 0;
    if (name == "Laboratory")   return 10;
    if (name == "Market")       return 11;
    if (name == "Festival")     return 12;
    if (name == "Sentry")       return 13;
    if (name == "Merchant")     return 14;
    if (name == "Village")      return 20;
    if (name == "Witch")        return 30;
    if (name == "Smithy")       return 31;
    if (name == "Council Room") return 32;
    if (name == "Militia")      return 33;
    if (name == "Moat")         return 34;
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
    if (name == "Silver")       return 80;
    if (name == "Gold")         return 81;
    return 100;
}

// BigMoney green/buy logic shared by all engine strategies once greening
static int engine_green_buy(const DecisionPoint& dp, const GameState& state, int /*pid*/) {
    int provinces_left = state.get_supply().count("Province");
    int coins = state.coins();

    auto find = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].card_name == name) return i;
        return -1;
    };

    if (coins >= 8) {
        int idx = find("Province");
        if (idx >= 0) return idx;
    }
    if (coins >= 6) {
        if (provinces_left <= 4) { int idx = find("Duchy"); if (idx >= 0) return idx; }
        int idx = find("Gold");
        if (idx >= 0) return idx;
    }
    if (coins == 5) {
        if (provinces_left <= 5) { int idx = find("Duchy"); if (idx >= 0) return idx; }
        // Keep engine running at $5
        for (auto& n : {"Laboratory", "Market", "Festival"}) {
            int idx = find(n);
            if (idx >= 0) return idx;
        }
        int idx = find("Silver");
        if (idx >= 0) return idx;
    }
    if (coins >= 3) {
        if (provinces_left <= 2) { int idx = find("Estate"); if (idx >= 0) return idx; }
        int idx = find("Silver");
        if (idx >= 0) return idx;
    }
    if (coins == 2 && provinces_left <= 3) {
        int idx = find("Estate");
        if (idx >= 0) return idx;
    }
    return -1; // pass
}

// Sub-decision logic shared across engine strategies
static std::vector<int> engine_sub_decide(const DecisionPoint& dp, const GameState& state) {
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
            std::vector<std::pair<int, int>> scored;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                const Card* card = state.card_def(dp.options[i].card_id);
                scored.push_back({trash_priority(card), i});
            }
            std::sort(scored.begin(), scored.end());
            std::vector<int> result;
            for (int i = 0; i < dp.max_choices && i < static_cast<int>(scored.size()); i++) {
                if (scored[i].first < 50 || static_cast<int>(result.size()) < dp.min_choices)
                    result.push_back(scored[i].second);
            }
            while (static_cast<int>(result.size()) < dp.min_choices) {
                for (int i = 0; i < static_cast<int>(scored.size()); i++) {
                    bool already = false;
                    for (int r : result) if (r == scored[i].second) { already = true; break; }
                    if (!already) { result.push_back(scored[i].second); break; }
                }
                break;
            }
            return result;
        }
        case ChoiceType::GAIN: {
            int best_idx = 0, best_cost = -1;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                const Card* card = state.card_def(dp.options[i].card_id);
                int cost = card ? card->cost : 0;
                if (cost > best_cost) { best_cost = cost; best_idx = i; }
            }
            return {best_idx};
        }
        case ChoiceType::YES_NO: {
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
                if (dp.options[i].label == "Yes") return {i};
            return {static_cast<int>(dp.options.size()) - 1};
        }
        case ChoiceType::PLAY_CARD: {
            int best_idx = 0, best_prio = 999;
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

std::vector<int> EngineBot::decide(const DecisionPoint& dp, const GameState& state) {
    if (dp.options.empty()) return {};

    int pid = dp.player_id;
    auto kingdom = analyze_kingdom(state);
    auto strategy = pick_strategy(kingdom);

    // --- PLAY_ACTION: always play actions by priority (all strategies) ---
    if (dp.type == DecisionType::PLAY_ACTION) {
        // PURE_BM skips all actions
        if (strategy == EngineStrategy::PURE_BM) {
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
                if (dp.options[i].is_pass) return {i};
            return {static_cast<int>(dp.options.size()) - 1};
        }
        // Others play by priority
        int best_idx = -1, best_prio = 999;
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].is_pass) continue;
            int prio = action_priority(dp.options[i].card_name);
            if (prio < best_prio) { best_prio = prio; best_idx = i; }
        }
        if (best_idx >= 0) return {best_idx};
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].is_pass) return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    }

    // --- PLAY_TREASURE: always play all ---
    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].label == "Play all Treasures") return {i};
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (!dp.options[i].is_pass) return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    }

    // --- SUB-DECISIONS ---
    if (dp.type != DecisionType::BUY_CARD) {
        return engine_sub_decide(dp, state);
    }

    // --- BUY_CARD ---
    auto prof = analyze_deck(state, pid);
    int coins = state.coins();

    auto find = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].card_name == name) return i;
        return -1;
    };

    auto pass = [&]() -> std::vector<int> {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].is_pass) return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    };

    // ==================== PURE_BM ====================
    if (strategy == EngineStrategy::PURE_BM) {
        // Identical to BigMoneyAgent
        bool greening = (prof.total_money >= 16);
        if (greening) {
            int idx = engine_green_buy(dp, state, pid);
            if (idx >= 0) return {idx};
        } else {
            if (coins >= 6) { int idx = find("Gold"); if (idx >= 0) return {idx}; }
            if (coins >= 3) { int idx = find("Silver"); if (idx >= 0) return {idx}; }
        }
        return pass();
    }

    // ==================== BM_PLUS_X ====================
    // Buy 1 Chapel (if available), 1-2 copies of the best terminal, rest is money.
    if (strategy == EngineStrategy::BM_PLUS_X) {
        bool greening = (prof.total_money >= 16) || (prof.chapels >= 1 && prof.junk <= 2);
        int my_turns = (state.turn_number() + 1) / state.num_players();
        if (my_turns > 5) greening = true;

        if (greening) {
            int idx = engine_green_buy(dp, state, pid);
            if (idx >= 0) return {idx};
            return pass();
        }

        // Build phase: Chapel first, then 1-2 terminals, then money
        if (kingdom.has_chapel && prof.chapels == 0 && coins >= 2) {
            int idx = find("Chapel");
            if (idx >= 0) return {idx};
        }
        // Buy best terminal (up to 2 copies)
        if (!kingdom.best_terminal.empty()) {
            int owned = prof.counts.count(kingdom.best_terminal) ?
                        prof.counts.at(kingdom.best_terminal) : 0;
            if (owned < 2) {
                int idx = find(kingdom.best_terminal);
                if (idx >= 0) return {idx};
            }
        }
        // Also buy Lab/Market if available at $5 (cantrips don't collide)
        if (coins >= 5) {
            for (auto& n : {"Laboratory", "Market"}) {
                int owned = prof.counts.count(n) ? prof.counts.at(n) : 0;
                if (owned < 2) { int idx = find(n); if (idx >= 0) return {idx}; }
            }
        }
        // Money
        if (coins >= 6) { int idx = find("Gold"); if (idx >= 0) return {idx}; }
        if (coins >= 3) { int idx = find("Silver"); if (idx >= 0) return {idx}; }
        return pass();
    }

    // ==================== FULL_ENGINE ====================
    // Build: Chapel opening → Village/Draw components → Green when ready
    bool greening = false;
    int my_turns = (state.turn_number() + 1) / state.num_players();
    if (my_turns > 5) greening = true;
    if (prof.chapels >= 1 && prof.junk <= 2) greening = true;
    if (prof.villages >= 1 && prof.terminal_draw >= 1) greening = true;
    if (prof.cantrips >= 3) greening = true;
    if (prof.total_actions >= 3 && my_turns >= 4) greening = true;

    if (greening) {
        int idx = engine_green_buy(dp, state, pid);
        if (idx >= 0) return {idx};
        return pass();
    }

    // Build phase
    // Chapel is top priority on turns 1-2
    if (kingdom.has_chapel && prof.chapels == 0) {
        int idx = find("Chapel");
        if (idx >= 0) return {idx};
    }

    // At $6+ with 2+ actions, buy Gold (need economy)
    if (coins >= 6 && prof.total_actions >= 2) {
        int idx = find("Gold");
        if (idx >= 0) return {idx};
    }

    // Component limits
    auto is_limited = [&](const std::string& name) -> bool {
        if (name == "Chapel" && prof.chapels >= 1) return true;
        if ((name == "Smithy" || name == "Council Room" || name == "Witch" ||
             name == "Moat" || name == "Library") && prof.terminal_draw >= 2) return true;
        if ((name == "Village" || name == "Festival") && prof.villages >= 3) return true;
        return false;
    };

    int best_idx = -1, best_prio = 999;
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        if (dp.options[i].is_pass) continue;
        const std::string& name = dp.options[i].card_name;
        if (name.empty()) continue;
        if (name == "Province" || name == "Duchy" || name == "Estate" ||
            name == "Gardens" || name == "Curse" || name == "Copper") continue;
        if (is_limited(name)) continue;

        int prio = engine_build_priority(name);

        // Boost village if we have terminals but no village
        if ((name == "Village" || name == "Festival") &&
            prof.villages == 0 && prof.terminal_draw >= 1) prio = 5;
        // Boost draw if we have village but no terminal
        if ((name == "Smithy" || name == "Witch" || name == "Council Room") &&
            prof.terminal_draw == 0 && prof.villages >= 1) prio = 5;
        // Silver more attractive once we have key actions
        if (name == "Silver" && prof.total_actions >= 2 && coins <= 4) prio = 15;

        if (prio < best_prio) { best_prio = prio; best_idx = i; }
    }
    if (best_idx >= 0) return {best_idx};
    return pass();
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

        // For MULTI_FATE (Sentry), read the stashed card_id so UI can show it
        if (choice_type == ChoiceType::MULTI_FATE) {
            int sentry_cid = state_.get_turn_flag("sentry_card_id");
            if (sentry_cid > 0) {
                dp.source_card = state_.card_name(sentry_cid);
            }
        }

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
                    // Read stashed card IDs from Sentry
                    std::string flag_key = "sentry_order_card_" + std::to_string(options[i]);
                    int order_cid = state_.get_turn_flag(flag_key);
                    if (order_cid > 0) {
                        opt.label = state_.card_name(order_cid);
                        opt.card_id = order_cid;
                    } else {
                        opt.label = "Position " + std::to_string(options[i]);
                    }
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
    BaseCards::setup_supply(state_, kingdom_cards_);
    BaseCards::setup_starting_decks(state_);
    state_.start_game();
    total_decisions_ = 0;

    static constexpr int MAX_TURNS = 80;
    static constexpr int MAX_DECISIONS = 5000;

    while (!state_.is_game_over() && state_.turn_number() < MAX_TURNS
           && total_decisions_ < MAX_DECISIONS) {
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
