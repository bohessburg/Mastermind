#include "rules.h"

namespace Rules {

bool can_play_action(const GameState& state, int /*player_id*/, int card_id) {
    if (state.current_phase() != Phase::ACTION) return false;
    if (state.actions() <= 0) return false;
    const Card* card = state.card_def(card_id);
    return card && card->is_action();
}

bool can_play_treasure(const GameState& state, int /*player_id*/, int card_id) {
    if (state.current_phase() != Phase::TREASURE) return false;
    const Card* card = state.card_def(card_id);
    return card && card->is_treasure();
}

bool can_buy(const GameState& state, int /*player_id*/, int pile_index) {
    if (state.current_phase() != Phase::BUY) return false;
    if (state.buys() <= 0) return false;
    if (state.get_supply().is_pile_empty_index(pile_index)) return false;
    int top_id = state.get_supply().top_card_index(pile_index);
    if (top_id == -1) return false;
    const Card* card = state.card_def(top_id);
    return card && card->cost <= state.coins();
}

bool has_playable_action(const GameState& state, int player_id) {
    if (state.current_phase() != Phase::ACTION || state.actions() <= 0) return false;
    const auto& hand = state.get_player(player_id).get_hand();
    for (int card_id : hand) {
        const Card* card = state.card_def(card_id);
        if (card && card->is_action()) return true;
    }
    return false;
}

bool has_playable_treasure(const GameState& state, int player_id) {
    const auto& hand = state.get_player(player_id).get_hand();
    for (int card_id : hand) {
        const Card* card = state.card_def(card_id);
        if (card && card->is_treasure()) return true;
    }
    return false;
}

std::vector<int> playable_actions(const GameState& state, int player_id) {
    std::vector<int> result;
    if (state.current_phase() != Phase::ACTION || state.actions() <= 0) return result;
    const auto& hand = state.get_player(player_id).get_hand();
    for (int card_id : hand) {
        const Card* card = state.card_def(card_id);
        if (card && card->is_action()) result.push_back(card_id);
    }
    return result;
}

std::vector<int> playable_treasures(const GameState& state, int player_id) {
    std::vector<int> result;
    const auto& hand = state.get_player(player_id).get_hand();
    for (int card_id : hand) {
        const Card* card = state.card_def(card_id);
        if (card && card->is_treasure()) result.push_back(card_id);
    }
    return result;
}

std::vector<int> buyable_piles(const GameState& state, int /*player_id*/) {
    std::vector<int> result;
    if (state.buys() <= 0) return result;
    int coins = state.coins();
    const auto& piles = state.get_supply().piles();
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i].card_ids.empty()) continue;
        const Card* card = state.card_def(piles[i].card_ids.back());
        if (card && card->cost <= coins) result.push_back(i);
    }
    return result;
}

} // namespace Rules
