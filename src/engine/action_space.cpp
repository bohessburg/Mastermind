#include "action_space.h"

DecisionPoint build_action_decision(const GameState& state) {
    DecisionPoint dp;
    dp.player_id = state.current_player_id();
    dp.type = DecisionType::PLAY_ACTION;
    dp.min_choices = 1;
    dp.max_choices = 1;

    const Player& player = state.get_player(dp.player_id);
    const auto& hand = player.get_hand();

    int idx = 0;
    for (int i = 0; i < static_cast<int>(hand.size()); i++) {
        const Card* card = state.card_def(hand[i]);
        if (card && card->is_action()) {
            ActionOption opt;
            opt.local_id = idx++;
            opt.card_id = hand[i];
            opt.def_id = card->def_id;
            opt.is_pass = false;
            dp.options.push_back(opt);
        }
    }
    ActionOption pass_opt;
    pass_opt.local_id = idx;
    pass_opt.card_id = -1;
    pass_opt.is_pass = true;
    dp.options.push_back(pass_opt);

    return dp;
}

DecisionPoint build_treasure_decision(const GameState& state) {
    DecisionPoint dp;
    dp.player_id = state.current_player_id();
    dp.type = DecisionType::PLAY_TREASURE;
    dp.min_choices = 1;
    dp.max_choices = 1;

    const Player& player = state.get_player(dp.player_id);
    const auto& hand = player.get_hand();

    int copper_def = GameState::def_copper();
    int silver_def = GameState::def_silver();
    int gold_def = GameState::def_gold();

    bool all_basic = true;
    bool has_treasure = false;
    for (int card_id : hand) {
        const Card* card = state.card_def(card_id);
        if (card && card->is_treasure()) {
            has_treasure = true;
            int did = card->def_id;
            if (did != copper_def && did != silver_def && did != gold_def) {
                all_basic = false;
            }
        }
    }

    int idx = 0;
    if (has_treasure && all_basic) {
        ActionOption opt;
        opt.local_id = idx++;
        opt.card_id = -1;
        opt.is_pass = false;
        opt.is_play_all = true;
        dp.options.push_back(opt);
    }

    for (int i = 0; i < static_cast<int>(hand.size()); i++) {
        const Card* card = state.card_def(hand[i]);
        if (card && card->is_treasure()) {
            ActionOption opt;
            opt.local_id = idx++;
            opt.card_id = hand[i];
            opt.def_id = card->def_id;
            opt.is_pass = false;
            dp.options.push_back(opt);
        }
    }
    ActionOption pass_opt;
    pass_opt.local_id = idx;
    pass_opt.card_id = -1;
    pass_opt.is_pass = true;
    dp.options.push_back(pass_opt);

    return dp;
}

DecisionPoint build_buy_decision(const GameState& state) {
    DecisionPoint dp;
    dp.player_id = state.current_player_id();
    dp.type = DecisionType::BUY_CARD;
    dp.min_choices = 1;
    dp.max_choices = 1;

    int coins = state.coins();
    int idx = 0;

    // Iterate piles directly instead of allocating all_pile_names()
    const auto& piles = state.get_supply().piles();
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i].card_ids.empty()) continue;
        int top_id = piles[i].card_ids.back();
        const Card* card = state.card_def(top_id);
        if (card && card->cost <= coins) {
            ActionOption opt;
            opt.local_id = idx++;
            opt.card_id = top_id;
            opt.def_id = card->def_id;
            opt.pile_index = i;
            opt.is_pass = false;
            dp.options.push_back(opt);
        }
    }
    ActionOption pass_opt;
    pass_opt.local_id = idx;
    pass_opt.card_id = -1;
    pass_opt.is_pass = true;
    dp.options.push_back(pass_opt);

    return dp;
}
