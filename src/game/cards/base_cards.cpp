#include "base_cards.h"
#include "../card.h"

namespace BaseCards {

void register_all() {
    // --- Treasures ---

    CardRegistry::register_card({
        .name = "Copper",
        .cost = 0,
        .types = CardType::Treasure,
        .text = "+1 Coin",
        .victory_points = 0,
        .coin_value = 1,
        .tags = {},
        .on_play = [](GameState& state, int /*player_id*/, DecisionFn) {
            state.add_coins(1);
        },
        .on_react = nullptr,
        .vp_fn = nullptr,
        .on_gain = nullptr,
        .on_trash = nullptr,
        .on_duration = nullptr,
    });

    CardRegistry::register_card({
        .name = "Silver",
        .cost = 3,
        .types = CardType::Treasure,
        .text = "+2 Coins",
        .victory_points = 0,
        .coin_value = 2,
        .tags = {},
        .on_play = [](GameState& state, int /*player_id*/, DecisionFn) {
            state.add_coins(2);
        },
        .on_react = nullptr,
        .vp_fn = nullptr,
        .on_gain = nullptr,
        .on_trash = nullptr,
        .on_duration = nullptr,
    });

    CardRegistry::register_card({
        .name = "Gold",
        .cost = 6,
        .types = CardType::Treasure,
        .text = "+3 Coins",
        .victory_points = 0,
        .coin_value = 3,
        .tags = {},
        .on_play = [](GameState& state, int /*player_id*/, DecisionFn) {
            state.add_coins(3);
        },
        .on_react = nullptr,
        .vp_fn = nullptr,
        .on_gain = nullptr,
        .on_trash = nullptr,
        .on_duration = nullptr,
    });

    // --- Victory cards ---

    CardRegistry::register_card({
        .name = "Estate",
        .cost = 2,
        .types = CardType::Victory,
        .text = "1 VP",
        .victory_points = 1,
        .coin_value = 0,
        .tags = {},
        .on_play = nullptr,
        .on_react = nullptr,
        .vp_fn = nullptr,
        .on_gain = nullptr,
        .on_trash = nullptr,
        .on_duration = nullptr,
    });

    CardRegistry::register_card({
        .name = "Duchy",
        .cost = 5,
        .types = CardType::Victory,
        .text = "3 VP",
        .victory_points = 3,
        .coin_value = 0,
        .tags = {},
        .on_play = nullptr,
        .on_react = nullptr,
        .vp_fn = nullptr,
        .on_gain = nullptr,
        .on_trash = nullptr,
        .on_duration = nullptr,
    });

    CardRegistry::register_card({
        .name = "Province",
        .cost = 8,
        .types = CardType::Victory,
        .text = "6 VP",
        .victory_points = 6,
        .coin_value = 0,
        .tags = {},
        .on_play = nullptr,
        .on_react = nullptr,
        .vp_fn = nullptr,
        .on_gain = nullptr,
        .on_trash = nullptr,
        .on_duration = nullptr,
    });

    // --- Curse ---

    CardRegistry::register_card({
        .name = "Curse",
        .cost = 0,
        .types = CardType::Curse,
        .text = "-1 VP",
        .victory_points = -1,
        .coin_value = 0,
        .tags = {},
        .on_play = nullptr,
        .on_react = nullptr,
        .vp_fn = nullptr,
        .on_gain = nullptr,
        .on_trash = nullptr,
        .on_duration = nullptr,
    });
}

// Helper: create N cards of a given name and return their IDs
static std::vector<int> make_pile(GameState& state, const std::string& card_name, int count) {
    std::vector<int> ids;
    ids.reserve(count);
    for (int i = 0; i < count; i++) {
        ids.push_back(state.create_card(card_name));
    }
    return ids;
}

void setup_supply(GameState& state, const std::vector<std::string>& kingdom_cards) {
    int num_players = state.num_players();

    // Victory card counts scale with player count
    int victory_count = (num_players <= 2) ? 8 : 12;
    int curse_count = (num_players - 1) * 10;

    // Treasure piles
    state.get_supply().add_pile("Copper", make_pile(state, "Copper", 60 - (7 * num_players)));
    state.get_supply().add_pile("Silver", make_pile(state, "Silver", 40));
    state.get_supply().add_pile("Gold",   make_pile(state, "Gold", 30));

    // Victory piles
    state.get_supply().add_pile("Estate",   make_pile(state, "Estate", victory_count));
    state.get_supply().add_pile("Duchy",    make_pile(state, "Duchy", victory_count));
    state.get_supply().add_pile("Province", make_pile(state, "Province", victory_count));

    // Curse pile
    state.get_supply().add_pile("Curse", make_pile(state, "Curse", curse_count));

    // Kingdom card piles (10 cards each by default)
    for (const auto& name : kingdom_cards) {
        const Card* card = CardRegistry::get(name);
        if (!card) continue;

        int pile_size = 10;
        // Victory kingdom cards get the same count as other victory piles
        if (card->is_victory()) {
            pile_size = victory_count;
        }

        state.get_supply().add_pile(name, make_pile(state, name, pile_size));
    }
}

void setup_starting_decks(GameState& state) {
    for (int p = 0; p < state.num_players(); p++) {
        Player& player = state.get_player(p);

        // 7 Coppers
        for (int i = 0; i < 7; i++) {
            int id = state.create_card("Copper");
            player.add_to_discard(id);
        }

        // 3 Estates
        for (int i = 0; i < 3; i++) {
            int id = state.create_card("Estate");
            player.add_to_discard(id);
        }

        // Shuffle and draw opening hand of 5
        player.shuffle_discard_into_deck();
        player.draw_cards(5);
    }
}

} // namespace BaseCards
