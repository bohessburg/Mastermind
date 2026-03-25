#pragma once

#include "card.h"
#include "player.h"
#include "supply.h"

#include <string>
#include <unordered_map>
#include <vector>

enum class Phase {
    ACTION,
    BUY,
    CLEANUP,
    GAME_OVER
};

class GameState {
public:
    GameState(int num_players);

    // --- Card ID system ---
    // Every physical card in the game gets a unique int ID.
    // This is the only way to create cards — supply, starting decks, everything goes through here.
    int create_card(const std::string& card_name);
    const std::string& card_name(int card_id) const;
    const Card* card_def(int card_id) const;

    // --- Players ---
    int num_players() const;
    Player& get_player(int player_id);
    const Player& get_player(int player_id) const;
    int current_player_id() const;

    // --- Supply ---
    Supply& get_supply();
    const Supply& get_supply() const;

    // --- Turn state ---
    Phase current_phase() const;
    int actions() const;
    int buys() const;
    int coins() const;
    int turn_number() const;

    void add_actions(int n);
    void add_buys(int n);
    void add_coins(int n);

    // --- Game actions ---
    // Gain a card from supply to a player's discard pile. Returns the card_id, or -1 if pile empty.
    int gain_card(int player_id, const std::string& pile_name);

    // Gain a card to a specific zone
    int gain_card_to_hand(int player_id, const std::string& pile_name);
    int gain_card_to_deck_top(int player_id, const std::string& pile_name);

    // Trash a card (from anywhere — caller removes it from its current zone first)
    void trash_card(int card_id);
    const std::vector<int>& get_trash() const;

    // --- Turn lifecycle ---
    void start_game();
    void start_turn();
    void advance_phase();  // ACTION -> BUY -> CLEANUP -> next player's ACTION

    // --- Game end ---
    bool is_game_over() const;
    std::vector<int> calculate_scores() const;
    int winner() const;  // returns player_id of winner (-1 if tie)

private:
    // Players
    std::vector<Player> players_;
    int current_player_;

    // Supply and trash
    Supply supply_;
    std::vector<int> trash_;

    // Turn state
    Phase phase_;
    int actions_;
    int buys_;
    int coins_;
    int turn_number_;

    // Card ID tracking
    int next_card_id_;
    std::unordered_map<int, std::string> card_names_;  // card_id -> card_name
};
