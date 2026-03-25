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
            dp.options.push_back({idx++, hand[i], card->name, "Play " + card->name, false});
        }
    }
    dp.options.push_back({idx, -1, "", "End Actions", true});

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

    bool all_basic = true;
    bool has_treasure = false;
    for (int card_id : hand) {
        const Card* card = state.card_def(card_id);
        if (card && card->is_treasure()) {
            has_treasure = true;
            if (card->name != "Copper" && card->name != "Silver" && card->name != "Gold") {
                all_basic = false;
            }
        }
    }

    int idx = 0;
    if (has_treasure && all_basic) {
        dp.options.push_back({idx++, -1, "", "Play all Treasures", false});
    }

    for (int i = 0; i < static_cast<int>(hand.size()); i++) {
        const Card* card = state.card_def(hand[i]);
        if (card && card->is_treasure()) {
            dp.options.push_back({idx++, hand[i], card->name, "Play " + card->name, false});
        }
    }
    dp.options.push_back({idx, -1, "", "Stop playing Treasures", true});

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

    for (const auto& pile_name : state.get_supply().all_pile_names()) {
        if (state.get_supply().is_pile_empty(pile_name)) continue;
        int top_id = state.get_supply().top_card(pile_name);
        if (top_id == -1) continue;
        const Card* card = state.card_def(top_id);
        if (card && card->cost <= coins) {
            dp.options.push_back({idx++, top_id, card->name, "Buy " + card->name, false});
        }
    }
    dp.options.push_back({idx, -1, "", "End Buys", true});

    return dp;
}
