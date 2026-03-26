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
                case ChoiceType::DISCARD:
                case ChoiceType::TRASH:
                case ChoiceType::TOPDECK:
                case ChoiceType::PLAY_CARD:
                case ChoiceType::REVEAL: {
                    int idx = options[i];
                    opt.card_id = (idx < static_cast<int>(hand.size())) ? hand[idx] : -1;
                    break;
                }
                case ChoiceType::SELECT_FROM_DISCARD: {
                    int idx = options[i];
                    opt.card_id = (idx < static_cast<int>(discard.size())) ? discard[idx] : -1;
                    break;
                }
                case ChoiceType::GAIN: {
                    opt.card_id = options[i];
                    break;
                }
                case ChoiceType::YES_NO: {
                    opt.label = (options[i] == 0) ? "No" : "Yes";
                    break;
                }
                case ChoiceType::MULTI_FATE: {
                    if (options[i] == 0) opt.label = "Keep (put back on deck)";
                    else if (options[i] == 1) opt.label = "Discard";
                    else if (options[i] == 2) opt.label = "Trash";
                    else opt.label = "Option " + std::to_string(options[i]);
                    break;
                }
                case ChoiceType::ORDER: {
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

            if (n < min_choices || n > max_choices) {
                observe("  [engine] Invalid: need " + std::to_string(min_choices) +
                        "-" + std::to_string(max_choices) + " choices, got " +
                        std::to_string(n) + ". Retrying.");
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

    static constexpr int MAX_TURNS = 120;
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

        if (dp.options[chosen_idx].label == "Play all Treasures") {
            Player& player = state_.get_player(pid);
            DecisionFn decide_fn = make_decision_fn();

            std::vector<int> treasure_indices;
            for (int i = 0; i < player.hand_size(); i++) {
                const Card* card = state_.card_def(player.get_hand()[i]);
                if (card && card->is_treasure()) treasure_indices.push_back(i);
            }

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
