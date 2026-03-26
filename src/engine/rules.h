#pragma once

#include "../game/game_state.h"

#include <string>
#include <vector>

namespace Rules {
    bool can_play_action(const GameState& state, int player_id, int card_id);
    bool can_play_treasure(const GameState& state, int player_id, int card_id);
    bool can_buy(const GameState& state, int player_id, const std::string& pile_name);
    std::vector<int> playable_actions(const GameState& state, int player_id);
    std::vector<int> playable_treasures(const GameState& state, int player_id);
    std::vector<std::string> buyable_piles(const GameState& state, int player_id);
}
