#include "agents.h"
#include "card_ids.h"

#include <algorithm>
#include <unordered_map>

// Forward declarations
static std::vector<int> sentry_fate_decide(const DecisionPoint& dp, const GameState& state);

// ═══════════════════════════════════════════════════════════════════
// Shared priority functions using integer def_id comparison
// ═══════════════════════════════════════════════════════════════════

// Action play priority: villages first (need +Actions to play more cards),
// then cantrips (+1 Action, safe to chain), then strong terminals.
static int action_priority(int did) {
    // Tier 0: Villages — give +2 Actions, play these first
    if (did == CardIds::Village)          return 0;
    if (did == CardIds::Festival)         return 1;
    if (did == CardIds::WorkersVillage)   return 2;

    // Tier 1: Cantrips — give +1 Action, safe to chain
    if (did == CardIds::Laboratory) return 10;
    if (did == CardIds::Market)     return 11;
    if (did == CardIds::Sentry)     return 12;
    if (did == CardIds::Poacher)    return 13;
    if (did == CardIds::Merchant)   return 14;
    if (did == CardIds::Harbinger)  return 15;
    if (did == CardIds::Cellar)     return 16;
    if (did == CardIds::Hamlet)     return 17;
    if (did == CardIds::Lookout)    return 18;

    // Tier 0: Throne Room / King's Court — play first to double villages, cantrips, or terminals
    if (did == CardIds::ThroneRoom) return 0;
    if (did == CardIds::KingsCourt) return 0;

    // Tier 3: Terminals — these consume your last action
    if (did == CardIds::Witch)       return 30;
    if (did == CardIds::Scholar)     return 31;
    if (did == CardIds::CouncilRoom) return 32;
    if (did == CardIds::Smithy)      return 33;
    if (did == CardIds::Courtyard)   return 34;
    if (did == CardIds::Militia)     return 35;
    if (did == CardIds::Swindler)    return 36;
    if (did == CardIds::Library)     return 37;
    if (did == CardIds::Mine)        return 38;
    if (did == CardIds::Remodel)     return 39;
    if (did == CardIds::Moneylender) return 40;
    if (did == CardIds::Chapel)      return 41;
    if (did == CardIds::Artisan)     return 42;
    if (did == CardIds::Bandit)      return 43;
    if (did == CardIds::Bureaucrat)  return 44;
    if (did == CardIds::Workshop)    return 45;
    if (did == CardIds::Moat)        return 46;
    if (did == CardIds::Vassal)      return 47;

    return 50; // unknown cards
}

// Buy priority: lower = buy first when costs are equal.
static int buy_priority(int did) {
    // $8
    if (did == CardIds::Province)    return 0;
    // $6
    if (did == CardIds::Gold)        return 10;
    if (did == CardIds::Artisan)     return 11;
    // $5
    if (did == CardIds::Witch)        return 20;
    if (did == CardIds::Scholar)      return 21;
    if (did == CardIds::Laboratory)   return 22;
    if (did == CardIds::Festival)     return 23;
    if (did == CardIds::Market)       return 24;
    if (did == CardIds::Sentry)       return 25;
    if (did == CardIds::CouncilRoom)  return 26;
    if (did == CardIds::Mine)         return 27;
    if (did == CardIds::Library)      return 28;
    if (did == CardIds::Duchy)        return 29;
    // $4
    if (did == CardIds::Militia)          return 30;
    if (did == CardIds::Smithy)           return 31;
    if (did == CardIds::WorkersVillage)   return 32;
    if (did == CardIds::ThroneRoom)       return 33;
    if (did == CardIds::Remodel)          return 34;
    if (did == CardIds::Moneylender)      return 35;
    if (did == CardIds::Poacher)          return 36;
    if (did == CardIds::Bureaucrat)       return 37;
    if (did == CardIds::Gardens)          return 38;
    // $3
    if (did == CardIds::Silver)       return 40;
    if (did == CardIds::Village)      return 41;
    if (did == CardIds::Swindler)     return 42;
    if (did == CardIds::Lookout)      return 43;
    if (did == CardIds::Merchant)     return 44;
    if (did == CardIds::Workshop)     return 45;
    if (did == CardIds::Harbinger)    return 46;
    if (did == CardIds::Vassal)       return 47;
    // $2
    if (did == CardIds::Chapel)       return 50;
    if (did == CardIds::Hamlet)       return 51;
    if (did == CardIds::Courtyard)    return 52;
    if (did == CardIds::Moat)         return 53;
    if (did == CardIds::Cellar)       return 54;
    if (did == CardIds::Estate)       return 55;
    if (did == CardIds::Copper)       return 900;
    if (did == CardIds::Curse)        return 999;
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

// Trash decision with money floor protection.
// Never trash Coppers if it would bring total deck money below 4.
static std::vector<int> smart_trash(const DecisionPoint& dp, const GameState& state) {
    // Count total money in deck
    int pid = dp.player_id;
    int total_money = 0;
    state.get_player(pid).for_each_card([&](int cid) {
        const Card* c = state.card_def(cid);
        if (c && c->is_treasure()) total_money += c->coin_value;
    });

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
            if (money_after_trash - card->coin_value < 3 &&
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
static int engine_build_priority(int did) {
    // Tier 0: Trashing — thin the deck ASAP
    if (did == CardIds::Chapel)            return 0;
    if (did == CardIds::Lookout)           return 1;
    if (did == CardIds::Sentry)            return 2;

    // Cantrip draw
    if (did == CardIds::Laboratory)        return 10;
    if (did == CardIds::Market)            return 11;
    if (did == CardIds::Festival)          return 12;
    if (did == CardIds::Merchant)          return 14;
    if (did == CardIds::Hamlet)            return 15;

    // Villages
    if (did == CardIds::Village)           return 20;
    if (did == CardIds::WorkersVillage)    return 21;

    // Terminal draw / attacks
    if (did == CardIds::Witch)             return 30;
    if (did == CardIds::Scholar)           return 31;
    if (did == CardIds::Smithy)            return 32;
    if (did == CardIds::Courtyard)         return 33;
    if (did == CardIds::CouncilRoom)       return 34;
    if (did == CardIds::Swindler)          return 35;
    if (did == CardIds::Militia)           return 36;
    if (did == CardIds::Moat)              return 37;

    // Utility
    if (did == CardIds::Cellar)            return 40;
    if (did == CardIds::Harbinger)         return 41;
    if (did == CardIds::ThroneRoom)        return 42;
    if (did == CardIds::KingsCourt)        return 43;
    if (did == CardIds::Remodel)           return 44;
    if (did == CardIds::Poacher)           return 45;
    if (did == CardIds::Workshop)          return 46;
    if (did == CardIds::Artisan)           return 47;
    if (did == CardIds::Mine)              return 48;
    if (did == CardIds::Library)           return 49;
    if (did == CardIds::Vassal)            return 50;
    if (did == CardIds::Moneylender)       return 51;
    if (did == CardIds::Bandit)            return 52;
    if (did == CardIds::Bureaucrat)        return 53;
    if (did == CardIds::Storeroom)         return 54;
    if (did == CardIds::Oasis)             return 55;
    if (did == CardIds::Menagerie)         return 56;
    if (did == CardIds::Conspirator)       return 57;

    // Treasure — fallback
    if (did == CardIds::Silver)            return 80;
    if (did == CardIds::Gold)              return 81;
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
            if (dp.options[i].is_play_all) return {i};
        }
    }

    if (dp.type == DecisionType::BUY_CARD) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].def_id == CardIds::Province) return {i};
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
                if (dp.options[i].is_play_all) return {i};
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

            auto find_option = [&](int target_def_id) -> int {
                for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                    if (dp.options[i].def_id == target_def_id) return i;
                }
                return -1;
            };

            if (coins >= 8) {
                // Early game exception: no Gold and <5 Silvers → buy Gold instead
                int pid = state.current_player_id();
                int gold_def = GameState::def_gold();
                int silver_def = GameState::def_silver();
                int gold_count = 0, silver_count = 0;
                state.get_player(pid).for_each_card([&](int cid) {
                    int did = state.card_def_id(cid);
                    if (did == gold_def) gold_count++;
                    else if (did == silver_def) silver_count++;
                });
                if (gold_count == 0 && silver_count < 5) {
                    int idx = find_option(CardIds::Gold);
                    if (idx >= 0) return {idx};
                }
                int idx = find_option(CardIds::Province);
                if (idx >= 0) return {idx};
            }
            if (coins >= 6) {
                if (provinces_left <= 4) {
                    int idx = find_option(CardIds::Duchy);
                    if (idx >= 0) return {idx};
                }
                int idx = find_option(CardIds::Gold);
                if (idx >= 0) return {idx};
            }
            if (coins == 5) {
                if (provinces_left <= 5) {
                    int idx = find_option(CardIds::Duchy);
                    if (idx >= 0) return {idx};
                }
                int idx = find_option(CardIds::Silver);
                if (idx >= 0) return {idx};
            }
            if (coins >= 3) {
                if (provinces_left <= 2) {
                    int idx = find_option(CardIds::Estate);
                    if (idx >= 0) return {idx};
                }
                int idx = find_option(CardIds::Silver);
                if (idx >= 0) return {idx};
            }
            if (coins == 2) {
                if (provinces_left <= 3) {
                    int idx = find_option(CardIds::Estate);
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
                int prio = action_priority(dp.options[i].def_id);
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
                if (dp.options[i].is_play_all) return {i};
            }
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (!dp.options[i].is_pass) return {i};
            }
            return {static_cast<int>(dp.options.size()) - 1};
        }

        case DecisionType::BUY_CARD: {
            int provinces_left = state.get_supply().count("Province");
            int pid = dp.player_id;
            std::unordered_map<int, int> owned;
            state.get_player(pid).for_each_card([&](int cid) {
                owned[state.card_def_id(cid)]++;
            });

            int best_idx = -1;
            int best_cost = -1;
            int best_prio = 999999;

            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) continue;
                int did = dp.options[i].def_id;
                const Card* card = state.card_def(dp.options[i].card_id);
                if (!card) continue;
                if (did == CardIds::Curse || did == CardIds::Copper) continue;
                if (did == CardIds::Estate && provinces_left > 2) continue;
                if (did == CardIds::Duchy && provinces_left > 5) continue;

                int cost = card->cost;
                int base_prio = buy_priority(did);

                int copies = owned.count(did) ? owned[did] : 0;
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
                        if (dp.options[i].value != 0) return {i};
                    }
                    return {static_cast<int>(dp.options.size()) - 1};
                }
                case ChoiceType::PLAY_CARD: {
                    int best_idx = 0;
                    int best_prio = 999;
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        int did = dp.options[i].def_id;
                        if (did >= 0) {
                            int prio = action_priority(did);
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
    bool has_chapel;            // any trasher (Chapel, Lookout, Sentry)
    bool has_fast_trasher;      // Chapel specifically (trashes up to 4 per play)
    bool has_terminal_draw;
    bool has_strong_draw;       // Witch, Smithy, Scholar, Council Room (draw 3+)
    bool has_cantrip_draw;
    bool has_lab;
    bool has_witch;
    bool has_good_action;       // any action worth splashing into BM
    bool has_attack;            // Witch, Militia, Bandit, Swindler, Bureaucrat
    int best_terminal;  // def_id, -1 if none
    int best_action;    // def_id, -1 if none — best single action to buy for BM+X
};

static KingdomAnalysis analyze_kingdom(const GameState& state) {
    KingdomAnalysis k = {};
    k.best_terminal = -1;
    k.best_action = -1;

    // Build a set of available def_ids from supply piles
    // (small fixed-size — just check membership inline)
    static constexpr int MAX_PILES = 20;
    int available[MAX_PILES];
    int num_available = 0;

    for (const auto& pile : state.get_supply().piles()) {
        if (pile.card_ids.empty()) continue;
        const Card* card = state.card_def(pile.card_ids.back());
        if (!card) continue;
        int did = card->def_id;
        if (num_available < MAX_PILES) available[num_available++] = did;
        if (did == CardIds::Village || did == CardIds::Festival || did == CardIds::WorkersVillage) k.has_village = true;
        if (did == CardIds::Chapel || did == CardIds::Lookout || did == CardIds::Sentry) k.has_chapel = true;
        if (did == CardIds::Chapel) k.has_fast_trasher = true;
        if (did == CardIds::Witch) { k.has_witch = true; k.has_terminal_draw = true; k.has_strong_draw = true; }
        if (did == CardIds::Smithy || did == CardIds::CouncilRoom || did == CardIds::Library ||
            did == CardIds::Moat || did == CardIds::Scholar || did == CardIds::Courtyard) k.has_terminal_draw = true;
        if (did == CardIds::Smithy || did == CardIds::Scholar || did == CardIds::CouncilRoom) k.has_strong_draw = true;
        if (did == CardIds::Laboratory || did == CardIds::Market) k.has_cantrip_draw = true;
        if (did == CardIds::Laboratory) k.has_lab = true;
        if (did == CardIds::Witch || did == CardIds::Militia || did == CardIds::Bandit) k.has_attack = true;
    }

    // Find best terminal from available piles (priority order)
    int best_terminal_ids[] = {
        CardIds::Witch, CardIds::Scholar, CardIds::Smithy,
        CardIds::Courtyard, CardIds::CouncilRoom, CardIds::Moat, CardIds::Library
    };
    for (int tid : best_terminal_ids) {
        if (tid < 0) continue;
        for (int j = 0; j < num_available; j++) {
            if (available[j] == tid) { k.best_terminal = tid; goto found; }
        }
    }
    found:

    // Best action to splash into a money deck (ordered by strength for BM+X)
    int best_action_ids[] = {
        CardIds::Witch, CardIds::Bandit, CardIds::Militia, CardIds::Laboratory,
        CardIds::Smithy, CardIds::CouncilRoom, CardIds::Festival,
        CardIds::Market, CardIds::Swindler, CardIds::Courtyard, CardIds::Library,
        CardIds::Moat, CardIds::Scholar, CardIds::Moneylender
    };
    for (int aid : best_action_ids) {
        if (aid < 0) continue;
        for (int j = 0; j < num_available; j++) {
            if (available[j] == aid) {
                k.has_good_action = true;
                if (k.best_action < 0) k.best_action = aid;
            }
        }
    }

    return k;
}

static EngineStrategy pick_strategy(const KingdomAnalysis& k) {
    // FULL_ENGINE disabled — BM_PLUS_X outperforms on all tested boards
    // if (k.has_village && k.has_strong_draw && k.has_fast_trasher)
    //     return EngineStrategy::FULL_ENGINE;
    // Don't enter BM_PLUS_X just for Chapel/Sentry alone — trashing without
    // a good action to pair it with loses to pure BigMoney.
    if (k.has_good_action || k.has_terminal_draw || k.has_cantrip_draw)
        return EngineStrategy::BM_PLUS_X;
    // Chapel alone with no good actions → play pure money
    if (k.has_chapel)
        return EngineStrategy::PURE_BM;
    return EngineStrategy::PURE_BM;
}

struct DeckProfile {
    int villages, terminal_draw, cantrips, chapels;
    int total_action_cards, deck_size, treasures, junk;
    int total_money;          // sum of coin_value of treasures
    int total_plus_actions;   // sum of +Actions from all action cards
    int total_plus_cards;     // sum of +Cards from all action cards
    int total_plus_coins;     // sum of +Coins from all action cards
    std::unordered_map<int, int> counts;  // def_id → count
};

// +Actions, +Cards, +Coins each card provides when played
struct CardYield { int actions; int cards; int coins; };
static CardYield card_yield(int did) {
    // Villages
    if (did == CardIds::Village)           return {2, 1, 0};
    if (did == CardIds::Festival)          return {2, 0, 2};
    if (did == CardIds::WorkersVillage)    return {2, 1, 1};
    // Cantrips
    if (did == CardIds::Laboratory)        return {1, 2, 0};
    if (did == CardIds::Market)            return {1, 1, 1};
    if (did == CardIds::Sentry)            return {1, 1, 0};
    if (did == CardIds::Poacher)           return {1, 1, 1};
    if (did == CardIds::Merchant)          return {1, 1, 0};
    if (did == CardIds::Harbinger)         return {1, 1, 0};
    if (did == CardIds::Cellar)            return {1, 0, 0}; // draw depends on discard
    if (did == CardIds::Hamlet)            return {1, 1, 0}; // base only, discards are optional
    if (did == CardIds::Lookout)           return {1, 0, 0};
    if (did == CardIds::Oasis)             return {1, 1, 1};
    if (did == CardIds::Menagerie)         return {1, 1, 0}; // conservative (could be 3)
    if (did == CardIds::Conspirator)       return {0, 0, 2}; // conditional +1 Card +1 Action
    // Terminal draw
    if (did == CardIds::Smithy)            return {0, 3, 0};
    if (did == CardIds::Witch)             return {0, 2, 0};
    if (did == CardIds::CouncilRoom)       return {0, 4, 0};
    if (did == CardIds::Moat)              return {0, 2, 0};
    if (did == CardIds::Scholar)           return {0, 7, 0}; // net depends on hand, but huge draw
    if (did == CardIds::Courtyard)         return {0, 3, 0}; // net +2 (topdecks 1)
    if (did == CardIds::Library)           return {0, 2, 0}; // draws to 7, estimate +2
    // Terminal utility
    if (did == CardIds::Militia)           return {0, 0, 2};
    if (did == CardIds::Swindler)          return {0, 0, 2};
    // TR/KC multiply another card — only count a small bonus since they need
    // a target to be useful. The real value is in the build priority, not the yield.
    if (did == CardIds::ThroneRoom)        return {0, 0, 0}; // multiplier, no standalone yield
    if (did == CardIds::KingsCourt)        return {0, 0, 0};
    if (did == CardIds::Chapel)            return {0, 0, 0};
    if (did == CardIds::Remodel)           return {0, 0, 0};
    if (did == CardIds::Mine)              return {0, 0, 0};
    if (did == CardIds::Moneylender)       return {0, 0, 3}; // if trashing Copper
    if (did == CardIds::Workshop)          return {0, 0, 0};
    if (did == CardIds::Artisan)           return {0, 0, 0};
    if (did == CardIds::Bandit)            return {0, 0, 0};
    if (did == CardIds::Bureaucrat)        return {0, 0, 0};
    if (did == CardIds::Vassal)            return {0, 0, 2};
    if (did == CardIds::Storeroom)         return {0, 0, 0};
    return {0, 0, 0};
}

static DeckProfile analyze_deck(const GameState& state, int pid) {
    DeckProfile p = {};
    const Player& player = state.get_player(pid);
    p.deck_size = player.total_card_count();
    player.for_each_card([&](int cid) {
        const Card* card = state.card_def(cid);
        if (!card) return;

        int did = card->def_id;
        p.counts[did]++;
        if (card->is_action()) {
            p.total_action_cards++;
            auto y = card_yield(did);
            p.total_plus_actions += y.actions;
            p.total_plus_cards += y.cards;
            p.total_plus_coins += y.coins;
        }
        if (card->is_treasure()) { p.treasures++; p.total_money += card->coin_value; }
        if (did == CardIds::Village || did == CardIds::Festival || did == CardIds::WorkersVillage) p.villages++;
        if (did == CardIds::Smithy || did == CardIds::CouncilRoom || did == CardIds::Witch ||
            did == CardIds::Moat || did == CardIds::Library || did == CardIds::Scholar ||
            did == CardIds::Courtyard) p.terminal_draw++;
        if (did == CardIds::Laboratory || did == CardIds::Market || did == CardIds::Sentry ||
            did == CardIds::Merchant || did == CardIds::Harbinger || did == CardIds::Poacher ||
            did == CardIds::Cellar || did == CardIds::Hamlet || did == CardIds::Lookout) p.cantrips++;
        if (did == CardIds::Chapel) p.chapels++;
        if (did == GameState::def_copper() || did == GameState::def_estate() ||
            did == GameState::def_curse()) p.junk++;
    });
    return p;
}

static int engine_green_buy(const DecisionPoint& dp, const GameState& state, int /*pid*/) {
    int provinces_left = state.get_supply().count("Province");
    int coins = state.coins();

    auto find = [&](int target_def_id) -> int {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].def_id == target_def_id) return i;
        return -1;
    };

    if (coins >= 8) { int idx = find(CardIds::Province); if (idx >= 0) return idx; }
    if (coins >= 6) {
        if (provinces_left <= 4) { int idx = find(CardIds::Duchy); if (idx >= 0) return idx; }
        int idx = find(CardIds::Gold); if (idx >= 0) return idx;
    }
    if (coins == 5) {
        if (provinces_left <= 5) { int idx = find(CardIds::Duchy); if (idx >= 0) return idx; }
        for (int n : {CardIds::Laboratory, CardIds::Market, CardIds::Festival}) {
            if (n < 0) continue;
            int idx = find(n); if (idx >= 0) return idx;
        }
        int idx = find(CardIds::Silver); if (idx >= 0) return idx;
    }
    if (coins >= 3) {
        if (provinces_left <= 2) { int idx = find(CardIds::Estate); if (idx >= 0) return idx; }
        int idx = find(CardIds::Silver); if (idx >= 0) return idx;
    }
    if (coins == 2 && provinces_left <= 3) {
        int idx = find(CardIds::Estate); if (idx >= 0) return idx;
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
                int total_money = 0;
                state.get_player(pid_s).for_each_card([&](int cid) {
                    const Card* c = state.card_def(cid);
                    if (c && c->is_treasure()) total_money += c->coin_value;
                });
                if (total_money > 6) return {2};
                if (total_money > 4) return {1};
                return {0};
            }
            // Always discard Duchies and Provinces (don't trash VP we bought)
            if (card->def_id == CardIds::Duchy) return {1};
            if (card->def_id == CardIds::Province) return {1};
            // Keep everything else
            return {0};
        }
    }
    return {0};
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
            // Check if this is a Swindler gain (attacker choosing for opponent).
            // Swindler: all options have the same cost. Self-gain cards (Workshop etc)
            // offer different costs, so same-cost options means Swindler.
            bool all_same_cost = true;
            int first_cost = -1;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                const Card* card = state.card_def(dp.options[i].card_id);
                int c = card ? card->cost : 0;
                if (first_cost < 0) first_cost = c;
                else if (c != first_cost) { all_same_cost = false; break; }
            }

            if (all_same_cost && dp.options.size() > 1) {
                // Swindler: give opponent the worst card at this cost
                int cost = first_cost;

                // Cost 0: always Curse
                if (cost == 0) {
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        if (card && card->name == "Curse") return {i};
                    }
                }
                // Cost 2: always Estate
                if (cost == 2) {
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        if (card && card->name == "Estate") return {i};
                    }
                }
                // Cost 3-5: give worst action (reverse of best_action ranking)
                if (cost >= 3 && cost <= 5) {
                    // Worst actions first (reverse of BM+X best_action ranking)
                    static const std::vector<std::string> worst_first = {
                        "Moneylender", "Scholar", "Library", "Moat", "Festival",
                        "Swindler", "Market", "Courtyard", "Sentry", "Council Room",
                        "Militia", "Laboratory", "Smithy", "Bandit", "Witch"
                    };
                    // Pick worst action available, never give them a treasure
                    for (const auto& worst : worst_first) {
                        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                            const Card* card = state.card_def(dp.options[i].card_id);
                            if (card && card->name == worst) return {i};
                        }
                    }
                    // No action found — pick any non-treasure (Victory card)
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        if (card && !card->is_treasure()) return {i};
                    }
                }
                // Cost 6: give action if available, otherwise Gold
                if (cost == 6) {
                    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                        const Card* card = state.card_def(dp.options[i].card_id);
                        if (card && card->is_action()) return {i};
                    }
                }
                // Cost 7+: just give them the same type of card (first option)
                return {0};
            }

            // Normal self-gain: pick most expensive
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
                if (dp.options[i].value != 0) return {i};
            return {static_cast<int>(dp.options.size()) - 1};
        }
        case ChoiceType::PLAY_CARD: {
            // TR/KC target selection: pick best action but never throne Chapel
            int best_idx = 0, best_prio = 999;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                int did = dp.options[i].def_id;
                if (did >= 0) {
                    if (did == CardIds::Chapel) continue; // never throne Chapel
                    int prio = action_priority(did);
                    if (prio < best_prio) { best_prio = prio; best_idx = i; }
                }
            }
            return {best_idx};
        }
        case ChoiceType::MULTI_FATE: {
            return sentry_fate_decide(dp, state);
        }
        case ChoiceType::ORDER: {
            // Sentry: choose which kept card goes on top of deck.
            // Prefer actions on top (we can play them), treasures below.
            int cid0 = state.get_turn_flag(TurnFlag::SentryOrderCard0);
            int cid1 = state.get_turn_flag(TurnFlag::SentryOrderCard1);
            const Card* c0 = (cid0 > 0) ? state.card_def(cid0) : nullptr;
            const Card* c1 = (cid1 > 0) ? state.card_def(cid1) : nullptr;
            bool c0_action = c0 && c0->is_action();
            bool c1_action = c1 && c1->is_action();
            // If one is action and other isn't, put action on top
            if (c0_action && !c1_action) return {0};
            if (c1_action && !c0_action) return {1};
            return {0}; // tie: keep original order
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

// ═══════════════════════════════════════════════════════════════════
// Sentry-Vassal strategy: when both are in the kingdom, build a
// deck of Sentries + Vassals. Sentry trashes junk and topdecks
// Vassals; Vassal chains off the top of deck for free +$2 each.
// ═══════════════════════════════════════════════════════════════════

static bool has_sentry_vassal(const GameState& state) {
    bool has_sentry = false, has_vassal = false;
    for (const auto& pile : state.get_supply().piles()) {
        if (pile.pile_name == "Sentry") has_sentry = true;
        if (pile.pile_name == "Vassal") has_vassal = true;
    }
    return has_sentry && has_vassal;
}

std::vector<int> EngineBot::sentry_vassal_decide(
    const DecisionPoint& dp, const GameState& state) {

    int pid = dp.player_id;

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

    // Count Vassals and Sentries in deck
    int vassal_count = 0, sentry_count = 0;
    for (int cid : state.get_player(pid).all_cards()) {
        const std::string& name = state.card_name(cid);
        if (name == "Vassal") vassal_count++;
        if (name == "Sentry") sentry_count++;
    }

    // PLAY_ACTION: Sentry first (sets up deck), then Vassal chains
    if (dp.type == DecisionType::PLAY_ACTION) {
        // Priority: Sentry > Vassal > other actions > pass
        for (auto& want : {"Sentry", "Vassal"}) {
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].card_name == want) return {i};
            }
        }
        // Play any other non-pass action
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (!dp.options[i].is_pass) return {i};
        }
        return pass();
    }

    // PLAY_TREASURE
    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].label == "Play all Treasures") return {i};
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (!dp.options[i].is_pass) return {i};
        return pass();
    }

    // BUY_CARD
    if (dp.type == DecisionType::BUY_CARD) {
        int coins = state.coins();
        int provinces_left = state.get_supply().count("Province");

        // Green when we have enough Vassals to chain $8
        bool greening = (vassal_count >= 4);

        if (greening) {
            // Province at $8+
            if (coins >= 8) {
                int idx = find("Province");
                if (idx >= 0) return {idx};
            }
            // Late game VP
            if (coins >= 5 && provinces_left <= 4) {
                int idx = find("Duchy");
                if (idx >= 0) return {idx};
            }
            if (coins >= 2 && provinces_left <= 2) {
                int idx = find("Estate");
                if (idx >= 0) return {idx};
            }
        }

        // Build: Sentry at $5+, Vassal at $3-4
        if (coins >= 5) {
            int idx = find("Sentry");
            if (idx >= 0) return {idx};
        }
        if (coins >= 3) {
            int idx = find("Vassal");
            if (idx >= 0) return {idx};
        }

        // Nothing useful affordable — pass (don't buy Copper/Silver/Gold)
        return pass();
    }

    // SUB-DECISIONS
    // Sentry MULTI_FATE: trash Copper/Estate, discard Duchy/Province, keep rest
    if (dp.sub_choice_type == ChoiceType::MULTI_FATE) {
        int sentry_cid = state.get_turn_flag(TurnFlag::SentryCardId);
        if (sentry_cid > 0) {
            const Card* card = state.card_def(sentry_cid);
            if (card) {
                if (card->name == "Copper") return {2};   // trash
                if (card->name == "Estate") return {2};    // trash
                if (card->is_curse()) return {2};          // trash
                if (card->name == "Duchy") return {1};     // discard
                if (card->name == "Province") return {1};  // discard
                return {0};                                 // keep (Vassal, Sentry, Gold, etc.)
            }
        }
        return {0};
    }

    // Sentry ORDER: when keeping 2 cards, put Vassal on top > Sentry > other action > treasure
    if (dp.sub_choice_type == ChoiceType::ORDER) {
        int cid0 = state.get_turn_flag(TurnFlag::SentryOrderCard0);
        int cid1 = state.get_turn_flag(TurnFlag::SentryOrderCard1);
        const Card* c0 = (cid0 > 0) ? state.card_def(cid0) : nullptr;
        const Card* c1 = (cid1 > 0) ? state.card_def(cid1) : nullptr;

        auto order_prio = [](const Card* c) -> int {
            if (!c) return 99;
            if (c->name == "Vassal") return 0;    // Vassal on top (chains)
            if (c->name == "Sentry") return 1;    // Sentry next
            if (c->is_action()) return 2;          // other actions
            return 10;                              // treasures last
        };

        int p0 = order_prio(c0);
        int p1 = order_prio(c1);
        return {(p0 <= p1) ? 0 : 1};  // lower prio = goes on top
    }

    // Vassal YES/NO (play revealed action?): always yes
    if (dp.sub_choice_type == ChoiceType::YES_NO) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].label == "Yes") return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    }

    // Default: first non-pass option
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
        if (!dp.options[i].is_pass) return {i};
    return {0};
}

// ═══════════════════════════════════════════════════════════════════

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
        // After first Chapel play, only play it if it's the only terminal available.
        bool has_other_terminal = false;
        if (chapel_plays_ >= 1) {
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                if (dp.options[i].is_pass) continue;
                int odid = dp.options[i].def_id;
                if (odid == CardIds::Chapel) continue;
                auto y = card_yield(odid);
                if (y.actions == 0) { has_other_terminal = true; break; }
            }
        }

        int best_idx = -1, best_prio = 999;
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
            if (dp.options[i].is_pass) continue;
            // After first play, skip Chapel if another terminal is available
            if (dp.options[i].def_id == CardIds::Chapel && chapel_plays_ >= 1 && has_other_terminal)
                continue;
            int prio = action_priority(dp.options[i].def_id);
            if (prio < best_prio) { best_prio = prio; best_idx = i; }
        }
        // Track Chapel plays
        if (best_idx >= 0 && dp.options[best_idx].def_id == CardIds::Chapel)
            chapel_plays_++;
        if (best_idx >= 0) return {best_idx};
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].is_pass) return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    }

    // PLAY_TREASURE
    if (dp.type == DecisionType::PLAY_TREASURE) {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].is_play_all) return {i};
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (!dp.options[i].is_pass) return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    }

    // SUB-DECISIONS
    if (dp.type != DecisionType::BUY_CARD) {
        // Chapel-specific trash logic: max_choices == 4 is Chapel's signature.
        // First play: trash all junk aggressively with no money floor.
        // Subsequent plays: preserve $3 in hand for buying.
        if (dp.sub_choice_type == ChoiceType::TRASH && dp.max_choices == 4) {
            bool first_play = (chapel_plays_ <= 1);

            int hand_money = 0;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                const Card* c = state.card_def(dp.options[i].card_id);
                if (c && c->is_treasure()) hand_money += c->coin_value;
            }

            std::vector<std::pair<int, int>> scored;
            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                const Card* c = state.card_def(dp.options[i].card_id);
                scored.push_back({trash_priority(c), i});
            }
            std::sort(scored.begin(), scored.end());

            std::vector<int> result;
            int money_left = hand_money;

            for (int i = 0; i < dp.max_choices && i < static_cast<int>(scored.size()); i++) {
                if (scored[i].first >= 50) break;

                const Card* c = state.card_def(dp.options[scored[i].second].card_id);

                if (!first_play && c && c->is_treasure()) {
                    if (money_left - c->coin_value < 3)
                        continue;
                    money_left -= c->coin_value;
                }

                result.push_back(scored[i].second);
            }
            return result;
        }
        return engine_sub_decide(dp, state);
    }

    // BUY_CARD
    auto prof = analyze_deck(state, pid);
    int coins = state.coins();

    auto find = [&](int target_def_id) -> int {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].def_id == target_def_id) return i;
        return -1;
    };

    auto pass = [&]() -> std::vector<int> {
        for (int i = 0; i < static_cast<int>(dp.options.size()); i++)
            if (dp.options[i].is_pass) return {i};
        return {static_cast<int>(dp.options.size()) - 1};
    };

    // PURE_BM — exact wiki Big Money optimized strategy
    if (strategy == EngineStrategy::PURE_BM) {
        int provinces_left = state.get_supply().count("Province");

        if (coins >= 8) {
            // Early game exception: no Gold and <5 Silvers → buy Gold
            int gold_count = 0, silver_count = 0;
            int gold_def = GameState::def_gold();
            int silver_def = GameState::def_silver();
            state.get_player(pid).for_each_card([&](int cid) {
                int did = state.card_def_id(cid);
                if (did == gold_def) gold_count++;
                else if (did == silver_def) silver_count++;
            });
            if (gold_count == 0 && silver_count < 5) {
                int idx = find(CardIds::Gold);
                if (idx >= 0) return {idx};
            }
            int idx = find(CardIds::Province);
            if (idx >= 0) return {idx};
        }
        if (coins >= 6) {
            if (provinces_left <= 4) { int idx = find(CardIds::Duchy); if (idx >= 0) return {idx}; }
            int idx = find(CardIds::Gold);
            if (idx >= 0) return {idx};
        }
        if (coins == 5) {
            if (provinces_left <= 5) { int idx = find(CardIds::Duchy); if (idx >= 0) return {idx}; }
            int idx = find(CardIds::Silver);
            if (idx >= 0) return {idx};
        }
        if (coins >= 3) {
            if (provinces_left <= 2) { int idx = find(CardIds::Estate); if (idx >= 0) return {idx}; }
            int idx = find(CardIds::Silver);
            if (idx >= 0) return {idx};
        }
        if (coins == 2) {
            if (provinces_left <= 3) { int idx = find(CardIds::Estate); if (idx >= 0) return {idx}; }
        }
        return pass();
    }

    // BM_PLUS_X
    if (strategy == EngineStrategy::BM_PLUS_X) {
        double ds = static_cast<double>(prof.deck_size);
        double money_density = (prof.total_money + prof.total_plus_coins) / ds;
        int my_turns = (state.turn_number() + 1) / state.num_players();
        int sentry_count = prof.counts.count(CardIds::Sentry) ? prof.counts.at(CardIds::Sentry) : 0;
        bool has_trasher = (prof.chapels >= 1 || sentry_count >= 1);

        // Green when we have enough buying power or have thinned the deck.
        // P2 greens earlier to compensate for tempo disadvantage.
        bool is_p2 = (pid == 1);
        double money_thresh = is_p2 ? 1.1 : 1.2;
        bool greening = (money_density >= money_thresh);
        if (my_turns > 5) greening = true;

        if (greening) {
            int idx = engine_green_buy(dp, state, pid);
            if (idx >= 0) return {idx};
            return pass();
        }

        // Count terminals and total actions in deck for density limits
        int terminal_count = 0;
        int total_action_count = 0;
        state.get_player(pid).for_each_card([&](int cid) {
            const Card* c = state.card_def(cid);
            if (!c || !c->is_action()) return;
            int cdid = c->def_id;
            if (cdid == CardIds::Chapel || cdid == CardIds::Sentry) return; // trashers, not actions we're limiting
            if (cdid == CardIds::ThroneRoom || cdid == CardIds::KingsCourt) return; // multipliers, don't count
            total_action_count++;
            auto y = card_yield(cdid);
            if (y.actions == 0) terminal_count++;
        });
        double terminal_density = terminal_count / ds;
        double action_density = total_action_count / ds;
        // Buy Chapel only with Militia or Bandit -- attacks benefit most from
        // a thin deck (draw them more often + opponent is disrupted).
        bool has_militia_or_bandit = false;
        for (const auto& pile : state.get_supply().piles()) {
            if (pile.card_ids.empty()) continue;
            const Card* pc = state.card_def(pile.card_ids.back());
            if (pc && (pc->def_id == CardIds::Militia || pc->def_id == CardIds::Bandit))
                has_militia_or_bandit = true;
        }
        if (kingdom.has_chapel && !has_trasher && has_militia_or_bandit) {
            if (my_turns <= 2 && (coins == 2 || coins == 3)) {
                int idx = find(CardIds::Chapel);
                if (idx >= 0) return {idx};
            }
            int idx = find(CardIds::Sentry);
            if (idx >= 0) return {idx};
        }
        // Buy the best action in the kingdom (not necessarily affordable this turn --
        // saving the terminal slot for the strongest card is worth buying Silver now).
        int best = (kingdom.best_action >= 0) ? kingdom.best_action : kingdom.best_terminal;
        if (best >= 0 && best != CardIds::Sentry) {
            auto by = card_yield(best);
            bool is_terminal = (by.actions == 0);
            // Witch gets a special exemption: always buy up to 2 copies.
            // Double Witch wins 96% vs Big Money -- Curses are devastating.
            bool under_limit;
            if (best == CardIds::Witch) {
                int witch_count = prof.counts.count(CardIds::Witch) ? prof.counts.at(CardIds::Witch) : 0;
                under_limit = (witch_count < 2);
            } else {
                under_limit = is_terminal
                    ? (terminal_density < EngineBot::terminal_density_limit)
                    : (action_density < EngineBot::action_density_limit);
            }
            if (under_limit) {
                int idx = find(best);
                if (idx >= 0) return {idx};
            }
        }
        // Buy cantrips if under total action density limit
        if (coins >= 5 && action_density < EngineBot::action_density_limit) {
            for (int n : {CardIds::Laboratory, CardIds::Market}) {
                if (n < 0) continue;
                int idx = find(n); if (idx >= 0) return {idx};
            }
        }
        if (coins >= 6) { int idx = find(CardIds::Gold); if (idx >= 0) return {idx}; }
        if (coins >= 3) { int idx = find(CardIds::Silver); if (idx >= 0) return {idx}; }
        return pass();
    }

    // ==================== FULL_ENGINE ====================
    // Green when the engine can reliably draw most of the deck and produce $8.
    // Three metrics (all relative to deck size):
    //   draw_ratio: total +Cards / deck_size — can we draw our deck?
    //   action_ratio: total +Actions / deck_size — can we chain?
    //   money_density: (treasure_money + action_coins) / deck_size — can we buy Province?
    //
    // Build phase: buy whichever metric is the bottleneck.

    int my_turns = (state.turn_number() + 1) / state.num_players();
    double ds = static_cast<double>(prof.deck_size);
    double draw_ratio = prof.total_plus_cards / ds;
    double action_ratio = prof.total_plus_actions / ds;
    double money_density = (prof.total_money + prof.total_plus_coins) / ds;

    // P2 needs to green earlier to avoid falling behind on tempo.
    // P1 can afford to build one more turn.
    bool is_p2 = (pid == 1);
    // Tuned via parameter sweep (best of 30+ configs tested)
    double draw_thresh    = is_p2 ? 0.3  : 0.45;
    double action_thresh  = is_p2 ? 0.15 : 0.25;
    double money_thresh   = is_p2 ? 1.0  : 1.2;
    int failsafe_turn     = is_p2 ? 4    : 5;

    bool greening = false;
    if (my_turns > failsafe_turn) greening = true;
    if (draw_ratio >= draw_thresh && action_ratio >= action_thresh &&
        money_density >= money_thresh) greening = true;
    // Chapel-thinned deck: green early if lean enough
    if (prof.chapels >= 1 && prof.junk <= 1 && prof.deck_size <= 10 &&
        money_density >= 1.0) greening = true;

    if (greening) {
        int idx = engine_green_buy(dp, state, pid);
        if (idx >= 0) return {idx};
        return pass();
    }

    // --- BUILD PHASE: buy whichever metric is the bottleneck ---

    // Always buy Chapel first if we don't have one (trashing is #1 priority)
    if (kingdom.has_chapel && prof.chapels == 0) {
        int idx = find(CardIds::Chapel);
        if (idx >= 0) return {idx};
    }

    // Determine the bottleneck
    enum class Bottleneck { ACTIONS, DRAW, MONEY, BALANCED };
    Bottleneck bottleneck = Bottleneck::BALANCED;

    if (action_ratio < 0.15 && prof.total_action_cards >= 1) {
        bottleneck = Bottleneck::ACTIONS;  // need villages
    } else if (draw_ratio < 0.3) {
        bottleneck = Bottleneck::DRAW;     // need draw cards
    } else if (money_density < 0.8) {
        bottleneck = Bottleneck::MONEY;    // need treasure/coin-producing actions
    }

    // Component limits
    // TR/KC are never limited -- always good to buy more
    auto is_limited = [&](int did) -> bool {
        if ((did == CardIds::Chapel || did == CardIds::Lookout) && prof.chapels >= 1) return true;
        if (did == CardIds::Sentry && prof.chapels >= 1) return true; // don't need Sentry + Chapel
        if ((did == CardIds::Smithy || did == CardIds::CouncilRoom || did == CardIds::Witch ||
             did == CardIds::Moat || did == CardIds::Library || did == CardIds::Scholar ||
             did == CardIds::Courtyard) && prof.terminal_draw >= 2) return true;
        if ((did == CardIds::Village || did == CardIds::Festival || did == CardIds::WorkersVillage)
            && prof.villages >= 3) return true;
        return false;
    };

    // Bottleneck-driven priority adjustments
    int best_idx = -1, best_prio = 999;
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        if (dp.options[i].is_pass) continue;
        int did = dp.options[i].def_id;
        if (did < 0) continue;
        if (did == CardIds::Province || did == CardIds::Duchy || did == CardIds::Estate ||
            did == CardIds::Gardens || did == CardIds::Curse || did == CardIds::Copper) continue;
        if (is_limited(did)) continue;

        int prio = engine_build_priority(did);
        auto y = card_yield(did);

        // Boost cards that address the bottleneck
        if (bottleneck == Bottleneck::ACTIONS && y.actions >= 2) {
            prio = std::min(prio, 3); // urgent: need villages
        } else if (bottleneck == Bottleneck::DRAW && y.cards >= 2) {
            prio = std::min(prio, 5); // urgent: need draw
        } else if (bottleneck == Bottleneck::MONEY) {
            if (did == CardIds::Gold && coins >= 6) prio = 2;
            else if (did == CardIds::Silver && coins >= 3) prio = 8;
            else if (y.coins >= 2) prio = std::min(prio, 10); // Festival, Militia, etc.
        }

        // Witch is always high priority if available (attack + draw)
        if (did == CardIds::Witch && prof.terminal_draw < 2) prio = std::min(prio, 15);

        if (prio < best_prio) { best_prio = prio; best_idx = i; }
    }
    if (best_idx >= 0) return {best_idx};
    return pass();
}

std::string EngineBot::strategy_for(const std::vector<std::string>& kingdom) {
    KingdomAnalysis k = {};
    for (const auto& name : kingdom) {
        if (name == "Village" || name == "Festival" || name == "Worker's Village") k.has_village = true;
        if (name == "Chapel" || name == "Lookout" || name == "Sentry") k.has_chapel = true;
        if (name == "Chapel") k.has_fast_trasher = true;
        if (name == "Witch") { k.has_witch = true; k.has_terminal_draw = true; k.has_strong_draw = true; }
        if (name == "Smithy" || name == "Council Room" || name == "Library" ||
            name == "Moat" || name == "Scholar" || name == "Courtyard") k.has_terminal_draw = true;
        if (name == "Smithy" || name == "Scholar" || name == "Council Room") k.has_strong_draw = true;
        if (name == "Laboratory" || name == "Market") k.has_cantrip_draw = true;
        if (name == "Laboratory") k.has_lab = true;
        if (name == "Witch" || name == "Militia" || name == "Bandit") k.has_attack = true;
        for (auto& a : {"Witch", "Scholar", "Smithy", "Laboratory", "Bandit",
                         "Council Room", "Militia", "Sentry", "Courtyard", "Market",
                         "Swindler", "Festival", "Moat", "Library", "Moneylender"}) {
            if (name == a) k.has_good_action = true;
        }
    }
    auto s = pick_strategy(k);
    switch (s) {
        case EngineStrategy::FULL_ENGINE: return "FULL_ENGINE";
        case EngineStrategy::BM_PLUS_X:  return "BM_PLUS_X";
        case EngineStrategy::PURE_BM:    return "PURE_BM";
    }
    return "???";
}
