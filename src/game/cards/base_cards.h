#pragma once

#include "../game_state.h"

#include <vector>
#include <string>

namespace BaseCards {
    // Register all base card definitions with the CardRegistry
    void register_all();

    // Set up a standard supply for a given number of players.
    // kingdom_cards: the 10 kingdom card pile names to include.
    // All card IDs are created through the GameState.
    void setup_supply(GameState& state, const std::vector<std::string>& kingdom_cards);

    // Give each player their starting deck (7 Copper + 3 Estate) and draw 5
    void setup_starting_decks(GameState& state);
}
