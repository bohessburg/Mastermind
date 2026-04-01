#pragma once

#include "card.h"
#include "player.h"
#include "supply.h"

#include <array>
#include <cstring>
#include <functional>
#include <string>
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
    int card_def_id(int card_id) const;  // O(1) card instance → card definition ID

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

    // --- Game actions (pile-index-based, no string lookups) ---
    int gain_card(int player_id, int pile_index);
    int gain_card_to_hand(int player_id, int pile_index);
    int gain_card_to_deck_top(int player_id, int pile_index);

    void trash_card(int card_id);
    const std::vector<int>& get_trash() const;

    int play_card_from_hand(int player_id, int hand_index, DecisionFn decide);
    void play_card_effect(int card_id, int player_id, DecisionFn decide);

    void resolve_attack(
        int attacker_id,
        std::function<void(GameState&, int target_id, DecisionFn)> attack_effect,
        DecisionFn decide);

    // Supply queries (return pile indices, no string allocation)
    std::vector<int> gainable_piles(int max_cost) const;
    std::vector<int> gainable_piles(int max_cost, CardType required_type) const;
    int effective_cost(int pile_index) const;
    int total_cards_owned(int player_id) const;

    // Well-known pile indices (cached after supply setup)
    int pile_copper() const   { return pile_copper_; }
    int pile_silver() const   { return pile_silver_; }
    int pile_gold() const     { return pile_gold_; }
    int pile_estate() const   { return pile_estate_; }
    int pile_duchy() const    { return pile_duchy_; }
    int pile_province() const { return pile_province_; }
    int pile_curse() const    { return pile_curse_; }
    void cache_well_known_piles();  // call after supply setup

    // Well-known card definition IDs (cached after cards registered)
    static int def_copper()   { return def_copper_; }
    static int def_silver()   { return def_silver_; }
    static int def_gold()     { return def_gold_; }
    static int def_estate()   { return def_estate_; }
    static int def_duchy()    { return def_duchy_; }
    static int def_province() { return def_province_; }
    static int def_curse()    { return def_curse_; }
    static void cache_well_known_def_ids();  // call after base cards registered

    // --- Turn lifecycle ---
    void start_game();
    void start_turn();
    void advance_phase();

    // --- Game end ---
    bool is_game_over() const;
    std::vector<int> calculate_scores() const;
    int winner() const;

    // --- Turn flags (int-indexed, no string hashing) ---
    int get_turn_flag(TurnFlag flag) const;
    void set_turn_flag(TurnFlag flag, int value);

    // --- Game log (cards can emit events) ---
    using LogFn = std::function<void(const std::string&)>;
    void set_log(LogFn fn);
    void log(const std::string& msg) const;

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
    std::vector<int> card_def_ids_;             // card_id → def_id

    // Well-known pile indices (instance, set after supply setup)
    int pile_copper_   = -1;
    int pile_silver_   = -1;
    int pile_gold_     = -1;
    int pile_estate_   = -1;
    int pile_duchy_    = -1;
    int pile_province_ = -1;
    int pile_curse_    = -1;

    // Well-known card def IDs (static, set once after base cards registered)
    static int def_copper_;
    static int def_silver_;
    static int def_gold_;
    static int def_estate_;
    static int def_duchy_;
    static int def_province_;
    static int def_curse_;

    std::array<int, static_cast<int>(TurnFlag::COUNT)> turn_flags_;
    LogFn log_fn_;
};
