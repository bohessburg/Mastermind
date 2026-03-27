#include "agents.h"

#include <algorithm>
#include <unordered_map>

// ═══════════════════════════════════════════════════════════════════
// Shared priority functions used by multiple agents
// ═══════════════════════════════════════════════════════════════════

// Action play priority: villages first (need +Actions to play more cards),
// then cantrips (+1 Action, safe to chain), then strong terminals.
int action_priority(const std::string& name) {
    // Tier 0: Villages — give +2 Actions, play these first
    if (name == "Village")          return 0;
    if (name == "Festival")         return 1;
    if (name == "Worker's Village") return 2;

    // Tier 1: Cantrips — give +1 Action, safe to chain
    if (name == "Laboratory") return 10;
    if (name == "Market")     return 11;
    if (name == "Sentry")     return 12;
    if (name == "Poacher")    return 13;
    if (name == "Merchant")   return 14;
    if (name == "Harbinger")  return 15;
    if (name == "Cellar")     return 16;
    if (name == "Hamlet")     return 17;
    if (name == "Lookout")    return 18;

    // Tier 2: Throne Room — play before a terminal to double it
    if (name == "Throne Room") return 20;

    // Tier 3: Terminals — these consume your last action
    if (name == "Witch")       return 30;
    if (name == "Scholar")     return 31;
    if (name == "Council Room") return 32;
    if (name == "Smithy")      return 33;
    if (name == "Courtyard")   return 34;
    if (name == "Militia")     return 35;
    if (name == "Swindler")    return 36;
    if (name == "Library")     return 37;
    if (name == "Mine")        return 38;
    if (name == "Remodel")     return 39;
    if (name == "Moneylender") return 40;
    if (name == "Chapel")      return 41;
    if (name == "Artisan")     return 42;
    if (name == "Bandit")      return 43;
    if (name == "Bureaucrat")  return 44;
    if (name == "Workshop")    return 45;
    if (name == "Moat")        return 46;
    if (name == "Vassal")      return 47;

    return 50; // unknown cards
}

// Buy priority: lower = buy first when costs are equal.
static int buy_priority(const std::string& name) {
    // $8
    if (name == "Province")    return 0;
    // $6
    if (name == "Gold")        return 10;
    if (name == "Artisan")     return 11;
    // $5
    if (name == "Witch")        return 20;
    if (name == "Scholar")      return 21;
    if (name == "Laboratory")   return 22;
    if (name == "Festival")     return 23;
    if (name == "Market")       return 24;
    if (name == "Sentry")       return 25;
    if (name == "Council Room") return 26;
    if (name == "Mine")         return 27;
    if (name == "Library")      return 28;
    if (name == "Duchy")        return 29;
    // $4
    if (name == "Militia")           return 30;
    if (name == "Smithy")            return 31;
    if (name == "Worker's Village")  return 32;
    if (name == "Throne Room")       return 33;
    if (name == "Remodel")           return 34;
    if (name == "Moneylender")       return 35;
    if (name == "Poacher")           return 36;
    if (name == "Bureaucrat")        return 37;
    if (name == "Gardens")           return 38;
    // $3
    if (name == "Silver")       return 40;
    if (name == "Village")      return 41;
    if (name == "Swindler")     return 42;
    if (name == "Lookout")      return 43;
    if (name == "Merchant")     return 44;
    if (name == "Workshop")     return 45;
    if (name == "Harbinger")    return 46;
    if (name == "Vassal")       return 47;
    // $2
    if (name == "Chapel")       return 50;
    if (name == "Hamlet")       return 51;
    if (name == "Courtyard")    return 52;
    if (name == "Moat")         return 53;
    if (name == "Cellar")       return 54;
    if (name == "Estate")       return 55;
    if (name == "Copper")       return 900;
    if (name == "Curse")        return 999;
    return 100;
}

// For discard decisions: prefer discarding junk (victory, curse) over useful cards
static int discard_priority(const Card* card) {
    if (!card) return 50;
    if (card->is_curse()) return 0;
    if (card->name == "Estate") return 1;
    if (card->name == "Duchy") return 2;
    if (card->name == "Province") return 3;
    if (card->name == "Copper") return 10;
    if (card->is_victory()) return 5;
    if (card->is_treasure()) return 20;
    if (card->is_action()) return 15;
    return 25;
}

// For trash decisions: prefer trashing junk
static int trash_priority(const Card* card) {
    if (!card) return 50;
    if (card->is_curse()) return 0;
    if (card->name == "Estate") return 1;
    if (card->name == "Copper") return 2;
    return 50;
}

// Engine build priority (lower = buy first)
static int engine_build_priority(const std::string& name) {
    // Trashing
    if (name == "Chapel")            return 0;
    if (name == "Lookout")           return 1;
    // Cantrip draw
    if (name == "Laboratory")        return 10;
    if (name == "Market")            return 11;
    if (name == "Festival")          return 12;
    if (name == "Sentry")            return 13;
    if (name == "Merchant")          return 14;
    if (name == "Hamlet")            return 15;
    // Villages
    if (name == "Village")           return 20;
    if (name == "Worker's Village")  return 21;
    // Terminal draw / attacks
    if (name == "Witch")             return 30;
    if (name == "Scholar")           return 31;
    if (name == "Smithy")            return 32;
    if (name == "Courtyard")         return 33;
    if (name == "Council Room")      return 34;
    if (name == "Swindler")          return 35;
    if (name == "Militia")           return 36;
    if (name == "Moat")              return 37;
    // Utility
    if (name == "Cellar")            return 40;
    if (name == "Harbinger")         return 41;
    if (name == "Throne Room")       return 42;
    if (name == "Remodel")           return 43;
    if (name == "Poacher")           return 44;
    if (name == "Workshop")          return 45;
    if (name == "Artisan")           return 46;
    if (name == "Mine")              return 47;
    if (name == "Library")           return 48;
    if (name == "Vassal")            return 49;
    if (name == "Moneylender")       return 50;
    if (name == "Bandit")            return 51;
    if (name == "Bureaucrat")        return 52;
    // Treasure
    if (name == "Silver")            return 80;
    if (name == "Gold")              return 81;
    return 100;
}

// ═══════════════════════════════════════════════════════════════════
// RandomAgent
// ═══════════════════════════════════════════════════════════════════

RandomAgent::RandomAgent(uint64_t seed) : rng_(seed) {}

std::vector<int> RandomAgent::decide(const DecisionPoint& dp, const GameState& /*state*/) {
    if (dp.options.empty()) return {};
    std::uniform_int_distribution<int> dist(0, static_cast<int>(dp.options.size()) - 1);
    return {dist(rng_)};
}

// ═══════════════════════════════════════════════════════════════════
// BetterRandomAgent
// ═══════════════════════════════════════════════════════════════════

BetterRandomAgent::BetterRandomAgent(uint64_t seed) : rng_(seed) {}

std::vector<int> BetterRandomAgent::decide(const DecisionPoint& dp, const GameState& /*state*/) {
    if (dp.options.empty()) return {};

    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].label == "Play all Treasures") return {i};
        }
    }

    if (dp.type == DecisionType::BUY_CARD) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].card_name == "Province") return {i};
        }
    }

    if (dp.min_choices > 1) {
        std::vector<int> all_indices;
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            all_indices.push_back(i);
        }
        std::shuffle(all_indices.begin(), all_indices.end(), rng_);
        int pick = std::min(dp.min_choices, static_cast<int>(all_indices.size()));
        return {all_indices.begin(), all_indices.begin() + pick};
    }

    std::vector<int> candidates;
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        if (!dp.options[i].is_pass) candidates.push_back(i);
    }

    if (candidates.empty()) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].is_pass) return {i};
        }
        return {0};
    }

    std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
    return {candidates[dist(rng_)]};
}

// ═══════════════════════════════════════════════════════════════════
// BigMoneyAgent
// ═══════════════════════════════════════════════════════════════════

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
            // Optimized Big Money from wiki.dominionstrategy.com:
            // $8: Province (unless early: no Gold and <5 Silvers → Gold)
            // $6-7: Gold (unless ≤4 Provinces → Duchy)
            // $5: Silver (unless ≤5 Provinces → Duchy)
            // $3-4: Silver (unless ≤2 Provinces → Estate)
            // $2: Estate only if ≤3 Provinces
            int provinces_left = state.get_supply().count("Province");
            int coins = state.coins();

            auto find_option = [&](const std::string& name) -> int {
                for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                    if (dp.options[i].card_name == name) return i;
                }
                return -1;
            };

            if (coins >= 8) {
                // Early game exception: no Gold and <5 Silvers → buy Gold instead
                int pid = state.current_player_id();
                auto all = state.get_player(pid).all_cards();
                int gold_count = 0, silver_count = 0;
                for (int cid : all) {
                    const std::string& n = state.card_name(cid);
                    if (n == "Gold") gold_count++;
                    else if (n == "Silver") silver_count++;
                }
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
            if (coins == 2) {
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

// ═══════════════════════════════════════════════════════════════════
// HeuristicAgent
// ═══════════════════════════════════════════════════════════════════

std::vector<int> HeuristicAgent::decide(const DecisionPoint& dp, const GameState& state) {
    if (dp.options.empty()) return {};

    switch (dp.type) {
        case DecisionType::PLAY_ACTION: {
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
            int provinces_left = state.get_supply().count("Province");
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
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        default: {
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
                    for (int i = 0; i < needed && i < static_cast<int>(scored.size()); i++) {
                        result.push_back(scored[i].second);
                    }
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
                    int limit = dp.max_choices;
                    for (int i = 0; i < limit && i < static_cast<int>(scored.size()); i++) {
                        if (scored[i].first < 50 || static_cast<int>(result.size()) < dp.min_choices) {
                            result.push_back(scored[i].second);
                        }
                    }
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
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        if (dp.options[i].label == "Yes") return {i};
                    }
                    return {static_cast<int>(dp.options.size()) - 1};
                }
                case ChoiceType::PLAY_CARD: {
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
                    return {0};
                }
                default: {
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

// ═══════════════════════════════════════════════════════════════════
// EngineBot — Kingdom-aware strategy selection
// ═══════════════════════════════════════════════════════════════════

enum class EngineStrategy { FULL_ENGINE, BM_PLUS_X, PURE_BM };

struct KingdomAnalysis {
    bool has_village;
    bool has_chapel;
    bool has_terminal_draw;
    bool has_cantrip_draw;
    bool has_witch;
    std::string best_terminal;
};

static KingdomAnalysis analyze_kingdom(const GameState& state) {
    KingdomAnalysis k = {};
    for (const auto& pile : state.get_supply().piles()) {
        if (pile.card_ids.empty()) continue;
        const auto& name = pile.pile_name;
        if (name == "Village" || name == "Festival" || name == "Worker's Village") k.has_village = true;
        if (name == "Chapel" || name == "Lookout") k.has_chapel = true;
        if (name == "Witch") { k.has_witch = true; k.has_terminal_draw = true; }
        if (name == "Smithy" || name == "Council Room" || name == "Library" ||
            name == "Moat" || name == "Scholar" || name == "Courtyard") k.has_terminal_draw = true;
        if (name == "Laboratory" || name == "Market") k.has_cantrip_draw = true;
    }
    for (auto& t : {"Witch", "Scholar", "Smithy", "Courtyard", "Council Room", "Moat", "Library"}) {
        if (!state.get_supply().is_pile_empty(t)) { k.best_terminal = t; break; }
    }
    return k;
}

static EngineStrategy pick_strategy(const KingdomAnalysis& k) {
    if (k.has_village && k.has_terminal_draw && k.has_chapel)
        return EngineStrategy::FULL_ENGINE;
    if (k.has_terminal_draw || k.has_chapel || k.has_cantrip_draw)
        return EngineStrategy::BM_PLUS_X;
    return EngineStrategy::PURE_BM;
}

struct DeckProfile {
    int villages, terminal_draw, cantrips, chapels;
    int total_actions, total_cards, treasures, junk;
    int total_money;
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
        if (name == "Village" || name == "Festival" || name == "Worker's Village") p.villages++;
        if (name == "Smithy" || name == "Council Room" || name == "Witch" ||
            name == "Moat" || name == "Library" || name == "Scholar" ||
            name == "Courtyard") p.terminal_draw++;
        if (name == "Laboratory" || name == "Market" || name == "Sentry" ||
            name == "Merchant" || name == "Harbinger" || name == "Poacher" ||
            name == "Cellar" || name == "Hamlet" || name == "Lookout") p.cantrips++;
        if (name == "Chapel") p.chapels++;
        if (name == "Copper" || name == "Estate" || name == "Curse") p.junk++;
    }
    return p;
}

static int engine_green_buy(const DecisionPoint& dp, const GameState& state, int /*pid*/) {
    int provinces_left = state.get_supply().count("Province");
    int coins = state.coins();

    auto find = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].card_name == name) return i;
        return -1;
    };

    if (coins >= 8) { int idx = find("Province"); if (idx >= 0) return idx; }
    if (coins >= 6) {
        if (provinces_left <= 4) { int idx = find("Duchy"); if (idx >= 0) return idx; }
        int idx = find("Gold"); if (idx >= 0) return idx;
    }
    if (coins == 5) {
        if (provinces_left <= 5) { int idx = find("Duchy"); if (idx >= 0) return idx; }
        for (auto& n : {"Laboratory", "Market", "Festival"}) {
            int idx = find(n); if (idx >= 0) return idx;
        }
        int idx = find("Silver"); if (idx >= 0) return idx;
    }
    if (coins >= 3) {
        if (provinces_left <= 2) { int idx = find("Estate"); if (idx >= 0) return idx; }
        int idx = find("Silver"); if (idx >= 0) return idx;
    }
    if (coins == 2 && provinces_left <= 3) {
        int idx = find("Estate"); if (idx >= 0) return idx;
    }
    return -1;
}

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

    // PLAY_ACTION
    if (dp.type == DecisionType::PLAY_ACTION) {
        if (strategy == EngineStrategy::PURE_BM) {
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
                if (dp.options[i].is_pass) return {i};
            return {static_cast<int>(dp.options.size()) - 1};
        }
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

    // PLAY_TREASURE
    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].label == "Play all Treasures") return {i};
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (!dp.options[i].is_pass) return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    }

    // SUB-DECISIONS
    if (dp.type != DecisionType::BUY_CARD) {
        return engine_sub_decide(dp, state);
    }

    // BUY_CARD
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

    // PURE_BM
    if (strategy == EngineStrategy::PURE_BM) {
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

    // BM_PLUS_X
    if (strategy == EngineStrategy::BM_PLUS_X) {
        bool greening = (prof.total_money >= 16) || (prof.chapels >= 1 && prof.junk <= 2);
        int my_turns = (state.turn_number() + 1) / state.num_players();
        if (my_turns > 5) greening = true;

        if (greening) {
            int idx = engine_green_buy(dp, state, pid);
            if (idx >= 0) return {idx};
            return pass();
        }

        if (kingdom.has_chapel && prof.chapels == 0 && coins >= 2) {
            int idx = find("Chapel");
            if (idx >= 0) return {idx};
        }
        if (!kingdom.best_terminal.empty()) {
            int owned = prof.counts.count(kingdom.best_terminal) ?
                        prof.counts.at(kingdom.best_terminal) : 0;
            if (owned < 2) {
                int idx = find(kingdom.best_terminal);
                if (idx >= 0) return {idx};
            }
        }
        if (coins >= 5) {
            for (auto& n : {"Laboratory", "Market"}) {
                int owned = prof.counts.count(n) ? prof.counts.at(n) : 0;
                if (owned < 2) { int idx = find(n); if (idx >= 0) return {idx}; }
            }
        }
        if (coins >= 6) { int idx = find("Gold"); if (idx >= 0) return {idx}; }
        if (coins >= 3) { int idx = find("Silver"); if (idx >= 0) return {idx}; }
        return pass();
    }

    // FULL_ENGINE
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

    if (kingdom.has_chapel && prof.chapels == 0) {
        int idx = find("Chapel");
        if (idx >= 0) return {idx};
    }

    if (coins >= 6 && prof.total_actions >= 2) {
        int idx = find("Gold");
        if (idx >= 0) return {idx};
    }

    auto is_limited = [&](const std::string& name) -> bool {
        if ((name == "Chapel" || name == "Lookout") && prof.chapels >= 1) return true;
        if ((name == "Smithy" || name == "Council Room" || name == "Witch" ||
             name == "Moat" || name == "Library" || name == "Scholar" ||
             name == "Courtyard") && prof.terminal_draw >= 2) return true;
        if ((name == "Village" || name == "Festival" || name == "Worker's Village")
            && prof.villages >= 3) return true;
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

        if ((name == "Village" || name == "Festival" || name == "Worker's Village") &&
            prof.villages == 0 && prof.terminal_draw >= 1) prio = 5;
        if ((name == "Smithy" || name == "Witch" || name == "Council Room" ||
             name == "Scholar" || name == "Courtyard") &&
            prof.terminal_draw == 0 && prof.villages >= 1) prio = 5;
        if (name == "Silver" && prof.total_actions >= 2 && coins <= 4) prio = 15;

        if (prio < best_prio) { best_prio = prio; best_idx = i; }
    }
    if (best_idx >= 0) return {best_idx};
    return pass();
}
