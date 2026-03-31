#include "agents.h"

#include <algorithm>
#include <unordered_map>

// Forward declarations
static std::vector<int> sentry_fate_decide(const DecisionPoint& dp, const GameState& state);

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
    int did = card->def_id;
    if (card->is_curse()) return 0;
    if (did == GameState::def_estate()) return 1;
    if (did == GameState::def_duchy()) return 2;
    if (did == GameState::def_province()) return 3;
    if (did == GameState::def_copper()) return 10;
    if (card->is_victory()) return 5;
    if (card->is_treasure()) return 20;
    if (card->is_action()) return 15;
    return 25;
}

// For trash decisions: prefer trashing junk
static int trash_priority(const Card* card) {
    if (!card) return 50;
    int did = card->def_id;
    if (card->is_curse()) return 0;
    if (did == GameState::def_estate()) return 1;
    if (did == GameState::def_copper()) return 2;
    return 50;
}

// Trash decision with money floor protection.
// Never trash Coppers if it would bring total deck money below 4.
static std::vector<int> smart_trash(const DecisionPoint& dp, const GameState& state) {
    // Count total money in deck
    int pid = dp.player_id;
    auto all = state.get_player(pid).all_cards();
    int total_money = 0;
    for (int cid : all) {
        const Card* c = state.card_def(cid);
        if (c && c->is_treasure()) total_money += c->coin_value;
    }

    std::vector<std::pair<int, int>> scored; // (priority, option_index)
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        const Card* card = state.card_def(dp.options[i].card_id);
        scored.push_back({trash_priority(card), i});
    }
    std::sort(scored.begin(), scored.end());

    std::vector<int> result;
    int money_after_trash = total_money;

    for (int i = 0; i < dp.max_choices && i < static_cast<int>(scored.size()); i++) {
        if (scored[i].first >= 50 && static_cast<int>(result.size()) >= dp.min_choices)
            break; // don't trash non-junk beyond minimum

        // Check money floor for Coppers
        const Card* card = state.card_def(dp.options[scored[i].second].card_id);
        if (card && card->is_treasure()) {
            if (money_after_trash - card->coin_value < 4 &&
                static_cast<int>(result.size()) >= dp.min_choices) {
                continue; // skip this Copper to protect money floor
            }
            money_after_trash -= card->coin_value;
        }

        result.push_back(scored[i].second);
    }

    // Ensure minimum (forced trash even if it breaks the floor)
    if (static_cast<int>(result.size()) < dp.min_choices) {
        for (int i = 0; i < static_cast<int>(scored.size()) &&
             static_cast<int>(result.size()) < dp.min_choices; i++) {
            bool already = false;
            for (int r : result) if (r == scored[i].second) { already = true; break; }
            if (!already) result.push_back(scored[i].second);
        }
    }

    return result;
}

// Engine build priority (lower = buy first)
static int engine_build_priority(const std::string& name) {
    // Tier 0: Trashing — thin the deck ASAP
    if (name == "Chapel")            return 0;
    if (name == "Lookout")           return 1;
    if (name == "Sentry")            return 2;

    // Cantrip draw
    if (name == "Laboratory")        return 10;
    if (name == "Market")            return 11;
    if (name == "Festival")          return 12;
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
    if (name == "King's Court")      return 43;
    if (name == "Remodel")           return 44;
    if (name == "Poacher")           return 45;
    if (name == "Workshop")          return 46;
    if (name == "Artisan")           return 47;
    if (name == "Mine")              return 48;
    if (name == "Library")           return 49;
    if (name == "Vassal")            return 50;
    if (name == "Moneylender")       return 51;
    if (name == "Bandit")            return 52;
    if (name == "Bureaucrat")        return 53;
    if (name == "Storeroom")         return 54;
    if (name == "Oasis")             return 55;
    if (name == "Menagerie")         return 56;
    if (name == "Conspirator")       return 57;

    // Treasure — fallback
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
            int provinces_left = state.get_supply().count_index(state.pile_province());
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
                int gold_def = GameState::def_gold();
                int silver_def = GameState::def_silver();
                int gold_count = 0, silver_count = 0;
                for (int cid : all) {
                    int did = state.card_def_id(cid);
                    if (did == gold_def) gold_count++;
                    else if (did == silver_def) silver_count++;
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
            int provinces_left = state.get_supply().count_index(state.pile_province());
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
                const Card* card = state.card_def(dp.options[i].card_id);
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
                    return smart_trash(dp, state);
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
                    return sentry_fate_decide(dp, state);
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
        if (name == "Chapel" || name == "Lookout" || name == "Sentry") k.has_chapel = true;
        if (name == "Witch") { k.has_witch = true; k.has_terminal_draw = true; }
        if (name == "Smithy" || name == "Council Room" || name == "Library" ||
            name == "Moat" || name == "Scholar" || name == "Courtyard") k.has_terminal_draw = true;
        if (name == "Laboratory" || name == "Market") k.has_cantrip_draw = true;
    }
    for (auto& t : {"Witch", "Scholar", "Smithy", "Courtyard", "Council Room", "Moat", "Library"}) {
        int pi = state.get_supply().pile_index_of(t);
        if (pi >= 0 && !state.get_supply().is_pile_empty_index(pi)) {
            k.best_terminal = t;
            break;
        }
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
    int total_action_cards, deck_size, treasures, junk;
    int total_money;          // sum of coin_value of treasures
    int total_plus_actions;   // sum of +Actions from all action cards
    int total_plus_cards;     // sum of +Cards from all action cards
    int total_plus_coins;     // sum of +Coins from all action cards
    std::unordered_map<std::string, int> counts;
};

// +Actions, +Cards, +Coins each card provides when played
struct CardYield { int actions; int cards; int coins; };
static CardYield card_yield(const std::string& name) {
    // Villages
    if (name == "Village")           return {2, 1, 0};
    if (name == "Festival")          return {2, 0, 2};
    if (name == "Worker's Village")  return {2, 1, 1};
    // Cantrips
    if (name == "Laboratory")        return {1, 2, 0};
    if (name == "Market")            return {1, 1, 1};
    if (name == "Sentry")            return {1, 1, 0};
    if (name == "Poacher")           return {1, 1, 1};
    if (name == "Merchant")          return {1, 1, 0};
    if (name == "Harbinger")         return {1, 1, 0};
    if (name == "Cellar")            return {1, 0, 0}; // draw depends on discard
    if (name == "Hamlet")            return {1, 1, 0}; // base only, discards are optional
    if (name == "Lookout")           return {1, 0, 0};
    if (name == "Oasis")             return {1, 1, 1};
    if (name == "Menagerie")         return {1, 1, 0}; // conservative (could be 3)
    if (name == "Conspirator")       return {0, 0, 2}; // conditional +1 Card +1 Action
    // Terminal draw
    if (name == "Smithy")            return {0, 3, 0};
    if (name == "Witch")             return {0, 2, 0};
    if (name == "Council Room")      return {0, 4, 0};
    if (name == "Moat")              return {0, 2, 0};
    if (name == "Scholar")           return {0, 7, 0}; // net depends on hand, but huge draw
    if (name == "Courtyard")         return {0, 3, 0}; // net +2 (topdecks 1)
    if (name == "Library")           return {0, 2, 0}; // draws to 7, estimate +2
    // Terminal utility
    if (name == "Militia")           return {0, 0, 2};
    if (name == "Swindler")          return {0, 0, 2};
    if (name == "Throne Room")       return {0, 0, 0};
    if (name == "King's Court")      return {0, 0, 0};
    if (name == "Chapel")            return {0, 0, 0};
    if (name == "Remodel")           return {0, 0, 0};
    if (name == "Mine")              return {0, 0, 0};
    if (name == "Moneylender")       return {0, 0, 3}; // if trashing Copper
    if (name == "Workshop")          return {0, 0, 0};
    if (name == "Artisan")           return {0, 0, 0};
    if (name == "Bandit")            return {0, 0, 0};
    if (name == "Bureaucrat")        return {0, 0, 0};
    if (name == "Vassal")            return {0, 0, 2};
    if (name == "Storeroom")         return {0, 0, 0};
    return {0, 0, 0};
}

static DeckProfile analyze_deck(const GameState& state, int pid) {
    DeckProfile p = {};
    auto all = state.get_player(pid).all_cards();
    p.deck_size = static_cast<int>(all.size());
    for (int cid : all) {
        const Card* card = state.card_def(cid);
        if (!card) continue;

        const std::string& name = card->name;
        p.counts[name]++;
        if (card->is_action()) {
            p.total_action_cards++;
            auto y = card_yield(name);
            p.total_plus_actions += y.actions;
            p.total_plus_cards += y.cards;
            p.total_plus_coins += y.coins;
        }
        if (card->is_treasure()) { p.treasures++; p.total_money += card->coin_value; }
        if (name == "Village" || name == "Festival" || name == "Worker's Village") p.villages++;
        if (name == "Smithy" || name == "Council Room" || name == "Witch" ||
            name == "Moat" || name == "Library" || name == "Scholar" ||
            name == "Courtyard") p.terminal_draw++;
        if (name == "Laboratory" || name == "Market" || name == "Sentry" ||
            name == "Merchant" || name == "Harbinger" || name == "Poacher" ||
            name == "Cellar" || name == "Hamlet" || name == "Lookout") p.cantrips++;
        if (name == "Chapel") p.chapels++;
        int did = card->def_id;
        if (did == GameState::def_copper() || did == GameState::def_estate() ||
            did == GameState::def_curse()) p.junk++;
    }
    return p;
}

static int engine_green_buy(const DecisionPoint& dp, const GameState& state, int /*pid*/) {
    int provinces_left = state.get_supply().count_index(state.pile_province());
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

static std::vector<int> sentry_fate_decide(const DecisionPoint& dp, const GameState& state) {
    int sentry_cid = state.get_turn_flag(TurnFlag::SentryCardId);
    if (sentry_cid > 0) {
        const Card* card = state.card_def(sentry_cid);
        if (card) {
            if (card->is_curse()) return {2};       // trash
            if (card->def_id == GameState::def_estate()) return {2};  // trash
            if (card->def_id == GameState::def_copper()) {
                int pid_s = dp.player_id;
                auto all = state.get_player(pid_s).all_cards();
                int total_money = 0;
                for (int cid : all) {
                    const Card* c = state.card_def(cid);
                    if (c && c->is_treasure()) total_money += c->coin_value;
                }
                if (total_money > 6) return {2};
                if (total_money > 4) return {1};
                return {0};
            }
        }
    }
    return {0}; // keep
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
            return smart_trash(dp, state);
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
        case ChoiceType::MULTI_FATE: {
            return sentry_fate_decide(dp, state);
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

    // PURE_BM — exact wiki Big Money optimized strategy
    if (strategy == EngineStrategy::PURE_BM) {
        int provinces_left = state.get_supply().count_index(state.pile_province());

        if (coins >= 8) {
            // Early game exception: no Gold and <5 Silvers → buy Gold
            int gold_def = GameState::def_gold();
            int silver_def = GameState::def_silver();
            int gold_count = 0, silver_count = 0;
            for (int cid : state.get_player(pid).all_cards()) {
                int did = state.card_def_id(cid);
                if (did == gold_def) gold_count++;
                else if (did == silver_def) silver_count++;
            }
            if (gold_count == 0 && silver_count < 5) {
                int idx = find("Gold");
                if (idx >= 0) return {idx};
            }
            int idx = find("Province");
            if (idx >= 0) return {idx};
        }
        if (coins >= 6) {
            if (provinces_left <= 4) { int idx = find("Duchy"); if (idx >= 0) return {idx}; }
            int idx = find("Gold");
            if (idx >= 0) return {idx};
        }
        if (coins == 5) {
            if (provinces_left <= 5) { int idx = find("Duchy"); if (idx >= 0) return {idx}; }
            int idx = find("Silver");
            if (idx >= 0) return {idx};
        }
        if (coins >= 3) {
            if (provinces_left <= 2) { int idx = find("Estate"); if (idx >= 0) return {idx}; }
            int idx = find("Silver");
            if (idx >= 0) return {idx};
        }
        if (coins == 2) {
            if (provinces_left <= 3) { int idx = find("Estate"); if (idx >= 0) return {idx}; }
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

    // ==================== FULL_ENGINE ====================
    int my_turns = (state.turn_number() + 1) / state.num_players();
    double ds = static_cast<double>(prof.deck_size);
    double draw_ratio = prof.total_plus_cards / ds;
    double action_ratio = prof.total_plus_actions / ds;
    double money_density = (prof.total_money + prof.total_plus_coins) / ds;

    bool is_p2 = (pid == 1);
    double draw_thresh    = is_p2 ? 0.3  : 0.45;
    double action_thresh  = is_p2 ? 0.15 : 0.25;
    double money_thresh   = is_p2 ? 1.0  : 1.2;
    int failsafe_turn     = is_p2 ? 4    : 5;

    bool greening = false;
    if (my_turns > failsafe_turn) greening = true;
    if (draw_ratio >= draw_thresh && action_ratio >= action_thresh &&
        money_density >= money_thresh) greening = true;
    if (prof.chapels >= 1 && prof.junk <= 1 && prof.deck_size <= 10 &&
        money_density >= 1.0) greening = true;

    if (greening) {
        int idx = engine_green_buy(dp, state, pid);
        if (idx >= 0) return {idx};
        return pass();
    }

    // --- BUILD PHASE: buy whichever metric is the bottleneck ---

    if (kingdom.has_chapel && prof.chapels == 0) {
        int idx = find("Chapel");
        if (idx >= 0) return {idx};
    }

    enum class Bottleneck { ACTIONS, DRAW, MONEY, BALANCED };
    Bottleneck bottleneck = Bottleneck::BALANCED;

    if (action_ratio < 0.15 && prof.total_action_cards >= 1) {
        bottleneck = Bottleneck::ACTIONS;
    } else if (draw_ratio < 0.3) {
        bottleneck = Bottleneck::DRAW;
    } else if (money_density < 0.8) {
        bottleneck = Bottleneck::MONEY;
    }

    auto is_limited = [&](const std::string& name) -> bool {
        if ((name == "Chapel" || name == "Lookout") && prof.chapels >= 1) return true;
        if (name == "Sentry" && prof.chapels >= 1) return true;
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
        auto y = card_yield(name);

        if (bottleneck == Bottleneck::ACTIONS && y.actions >= 2) {
            prio = std::min(prio, 3);
        } else if (bottleneck == Bottleneck::DRAW && y.cards >= 2) {
            prio = std::min(prio, 5);
        } else if (bottleneck == Bottleneck::MONEY) {
            if (name == "Gold" && coins >= 6) prio = 2;
            else if (name == "Silver" && coins >= 3) prio = 8;
            else if (y.coins >= 2) prio = std::min(prio, 10);
        }

        if (name == "Witch" && prof.terminal_draw < 2) prio = std::min(prio, 15);

        if (prio < best_prio) { best_prio = prio; best_idx = i; }
    }
    if (best_idx >= 0) return {best_idx};
    return pass();
}
