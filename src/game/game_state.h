#pragma once

#include "card.h"
#include "player.h"
#include "supply.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

enum class Phase {
    ACTION,
    TREASURE,
    BUY,
    NIGHT,
    CLEANUP,
    GAME_OVER
};

class GameState {
public:
    GameState(int num_players);

    // --- Card ID system ---
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
    void set_phase(Phase phase);
    int actions() const;
    int buys() const;
    int coins() const;
    int turn_number() const;
    int actions_played() const;

    void add_actions(int n);
    void add_buys(int n);
    void add_coins(int n);

    // --- Game actions ---
    int gain_card(int player_id, const std::string& pile_name);
    int gain_card_to_hand(int player_id, const std::string& pile_name);
    int gain_card_to_deck_top(int player_id, const std::string& pile_name);

    void trash_card(int card_id);
    const std::vector<int>& get_trash() const;

    int play_card_from_hand(int player_id, int hand_index, DecisionFn decide);
    void play_card_effect(int card_id, int player_id, DecisionFn decide);

    void resolve_attack(
        int attacker_id,
        std::function<void(GameState&, int target_id, DecisionFn)> attack_effect,
        DecisionFn decide);

    // Supply queries
    std::vector<std::string> gainable_piles(int max_cost) const;
    std::vector<std::string> gainable_piles(int max_cost, CardType required_type) const;
    int effective_cost(const std::string& card_name) const;
    int total_cards_owned(int player_id) const;

    // --- Turn lifecycle ---
    void start_game();
    void start_turn();
    void advance_phase();

    // --- Game end ---
    bool is_game_over() const;
    std::vector<int> calculate_scores() const;
    int winner() const;

    // --- Turn flags (for cards like Merchant) ---
    int get_turn_flag(const std::string& key) const;
    void set_turn_flag(const std::string& key, int value);

private:
    std::vector<Player> players_;
    int current_player_;

    Supply supply_;
    std::vector<int> trash_;

    Phase phase_;
    int actions_;
    int buys_;
    int coins_;
    int turn_number_;
    int actions_played_;
    std::vector<int> turns_taken_;

    // Card ID system: sequential IDs, vector-indexed for O(1) lookup
    int next_card_id_;
    std::vector<std::string> card_names_;       // card_id → card_name
    std::vector<const Card*> card_defs_;        // card_id → cached Card*

    std::unordered_map<std::string, int> turn_flags_;
};
