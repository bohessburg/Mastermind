#include "game_runner.h"
#include "rules.h"

#include <algorithm>
#include <sstream>

// --- GameRunner ---

GameRunner::GameRunner(int num_players, const std::vector<std::string>& kingdom_cards)
    : state_(num_players)
    , kingdom_cards_(kingdom_cards)
    , total_decisions_(0)
{
}

GameRunner::GameRunner(GameState state)
    : state_(std::move(state))
    , kingdom_cards_()
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
        if (observer_ && choice_type == ChoiceType::MULTI_FATE) {
            int sentry_cid = state_.get_turn_flag(TurnFlag::SentryCardId);
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
            opt.def_id = -1;
            opt.is_pass = false;
            opt.is_play_all = false;
            opt.value = options[i];

            switch (choice_type) {
                case ChoiceType::DISCARD:
                case ChoiceType::TRASH:
                case ChoiceType::TOPDECK:
                case ChoiceType::PLAY_CARD:
                case ChoiceType::REVEAL: {
                    int idx = options[i];
                    if (idx < static_cast<int>(hand.size())) {
                        opt.card_id = hand[idx];
                        opt.def_id = state_.card_def_id(opt.card_id);
                    }
                    break;
                }
                case ChoiceType::SELECT_FROM_DISCARD: {
                    int idx = options[i];
                    if (idx < static_cast<int>(discard.size())) {
                        opt.card_id = discard[idx];
                        opt.def_id = state_.card_def_id(opt.card_id);
                    }
                    break;
                }
                case ChoiceType::GAIN: {
                    opt.card_id = options[i];
                    opt.def_id = state_.card_def_id(opt.card_id);
                    break;
                }
                case ChoiceType::YES_NO: {
                    if (observer_) {
                        opt.label = (options[i] == 0) ? "No" : "Yes";
                    }
                    break;
                }
                case ChoiceType::MULTI_FATE: {
                    if (observer_) {
                        if (options[i] == 0) opt.label = "Keep (put back on deck)";
                        else if (options[i] == 1) opt.label = "Discard";
                        else if (options[i] == 2) opt.label = "Trash";
                        else opt.label = "Option " + std::to_string(options[i]);
                    }
                    break;
                }
                case ChoiceType::ORDER: {
                    opt.card_id = options[i];
                    opt.def_id = state_.card_def_id(opt.card_id);
                    if (observer_) {
                        opt.label = state_.card_name(options[i]);
                    }
                    break;
                }
                default: {
                    opt.card_id = options[i];
                    if (opt.card_id >= 0) {
                        opt.def_id = state_.card_def_id(opt.card_id);
                    }
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

            if (n < min_choices || n > max_choices) {
                if (observer_) {
                    observe("  [engine] Invalid: need " + std::to_string(min_choices) +
                            "-" + std::to_string(max_choices) + " choices, got " +
                            std::to_string(n) + ". Retrying.");
                }
                continue;
            }

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
                if (observer_) {
                    observe("  [engine] Invalid indices. Retrying.");
                }
                continue;
            }

            return result;
        }
    };
}

GameResult GameRunner::run(std::vector<Agent*> agents) {
    agents_ = agents;
    // Route card-level logs through the observer
    if (observer_) {
        state_.set_log([this](const std::string& msg) { observe(msg); });
    }
    BaseCards::setup_supply(state_, kingdom_cards_);
    BaseCards::setup_starting_decks(state_);
    state_.start_game();
    total_decisions_ = 0;

    static constexpr int MAX_TURNS = 80;
    static constexpr int MAX_DECISIONS = 5000;

    while (!state_.is_game_over() && state_.turn_number() < MAX_TURNS
           && total_decisions_ < MAX_DECISIONS) {
        int pid = state_.current_player_id();
        if (observer_) {
            observe("--- Player " + std::to_string(pid) + "'s turn (Turn " +
                    std::to_string(state_.turn_number()) + ") ---");
        }
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

GameResult GameRunner::resume(std::vector<Agent*> agents, Phase start_phase) {
    agents_ = agents;
    // Rollouts run silent — do not install a log handler.

    static constexpr int MAX_TURNS = 80;
    static constexpr int MAX_DECISIONS = 5000;

    if (!state_.is_game_over()) {
        // Finish the current player's in-progress turn from the given phase.
        // The engine does not track ACTION→TREASURE→BUY transitions in
        // state_.phase_, so the caller (MCTS) tells us where to pick up.
        int pid = state_.current_player_id();
        if (start_phase == Phase::ACTION) {
            run_action_phase(pid);
            run_treasure_phase(pid);
            run_buy_phase(pid);
        } else if (start_phase == Phase::TREASURE) {
            run_treasure_phase(pid);
            run_buy_phase(pid);
        } else if (start_phase == Phase::BUY) {
            run_buy_phase(pid);
        }
        state_.set_phase(Phase::CLEANUP);
        state_.advance_phase();
    }

    while (!state_.is_game_over() && state_.turn_number() < MAX_TURNS
           && total_decisions_ < MAX_DECISIONS) {
        int pid = state_.current_player_id();
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
        if (!Rules::has_playable_action(state_, pid)) break;

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

        if (observer_) {
            const Card* card_info = state_.card_def(card_id);
            observe("  Player " + std::to_string(pid) + " plays " +
                    (card_info ? card_info->name : "???"));
        }

        state_.add_actions(-1);
        DecisionFn decide_fn = make_decision_fn();
        state_.play_card_from_hand(pid, hand_idx, decide_fn);
    }
}

void GameRunner::run_treasure_phase(int pid) {
    Agent* agent = agents_[pid];
    int silver_def = GameState::def_silver();

    while (true) {
        if (!Rules::has_playable_treasure(state_, pid)) break;

        DecisionPoint dp = build_treasure_decision(state_);
        auto choice = agent->decide(dp, state_);
        total_decisions_++;
        if (choice.empty()) break;

        int chosen_idx = choice[0];
        if (chosen_idx >= static_cast<int>(dp.options.size()) || dp.options[chosen_idx].is_pass) break;

        if (dp.options[chosen_idx].is_play_all) {
            Player& player = state_.get_player(pid);
            DecisionFn decide_fn = make_decision_fn();

            std::vector<int> treasure_indices;
            for (int i = 0; i < player.hand_size(); i++) {
                const Card* card = state_.card_def(player.get_hand()[i]);
                if (card && card->is_treasure()) treasure_indices.push_back(i);
            }

            if (observer_) {
                std::ostringstream oss;
                oss << "  Player " << pid << " plays all Treasures:";
                for (int i : treasure_indices) {
                    oss << " " << state_.card_name(player.get_hand()[i]);
                }
                observe(oss.str());
            }

            std::sort(treasure_indices.begin(), treasure_indices.end(), std::greater<int>());

            int merchant_count = state_.get_turn_flag(TurnFlag::MerchantCount);
            bool silver_triggered = state_.get_turn_flag(TurnFlag::MerchantSilverTriggered) != 0;

            for (int idx : treasure_indices) {
                int cid = player.get_hand()[idx];
                const Card* card = state_.card_def(cid);
                player.play_from_hand(idx);
                if (card && card->on_play) {
                    card->on_play(state_, pid, decide_fn);
                }
                if (card && card->def_id == silver_def && merchant_count > 0 && !silver_triggered) {
                    state_.add_coins(merchant_count);
                    state_.set_turn_flag(TurnFlag::MerchantSilverTriggered, 1);
                    silver_triggered = true;
                }
            }
            break;
        }

        int card_id = dp.options[chosen_idx].card_id;
        Player& player = state_.get_player(pid);
        int hand_idx = player.find_in_hand(card_id);
        if (hand_idx == -1) break;

        if (observer_) {
            const Card* card = state_.card_def(card_id);
            observe("  Player " + std::to_string(pid) + " plays " +
                    (card ? card->name : "???"));
        }

        DecisionFn decide_fn = make_decision_fn();
        player.play_from_hand(hand_idx);
        const Card* card = state_.card_def(card_id);
        if (card && card->on_play) {
            card->on_play(state_, pid, decide_fn);
        }

        int merchant_count = state_.get_turn_flag(TurnFlag::MerchantCount);
        bool silver_triggered = state_.get_turn_flag(TurnFlag::MerchantSilverTriggered) != 0;
        if (card && card->def_id == silver_def && merchant_count > 0 && !silver_triggered) {
            state_.add_coins(merchant_count);
            state_.set_turn_flag(TurnFlag::MerchantSilverTriggered, 1);
        }
    }

    if (observer_) {
        observe("  Player " + std::to_string(pid) + " has " +
                std::to_string(state_.coins()) + " coins");
    }
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

        int pile_index = dp.options[chosen_idx].pile_index;
        const Card* card = state_.card_def(dp.options[chosen_idx].card_id);
        if (!card) break;

        if (observer_) {
            observe("  Player " + std::to_string(pid) + " buys " + card->name);
        }

        state_.add_coins(-card->cost);
        state_.add_buys(-1);
        state_.gain_card(pid, pile_index);
    }
}
