#pragma once

#include "game/game_state.h"
#include "game/card.h"
#include "game/cards/base_cards.h"
#include "game/cards/level_1_cards.h"

#include <memory>
#include <queue>
#include <string>
#include <vector>

class TestGame {
public:
    explicit TestGame(int num_players = 2);

    void set_hand(int pid, const std::vector<std::string>& card_names);
    void set_deck(int pid, const std::vector<std::string>& card_names);
    void set_discard(int pid, const std::vector<std::string>& card_names);
    void set_seed(uint64_t seed);
    void setup_supply(const std::vector<std::string>& kingdom_cards = {});

    static DecisionFn scripted_decisions(std::vector<std::vector<int>> choices);
    static DecisionFn minimal_decisions();
    static DecisionFn greedy_decisions();

    void play_card(int pid, const std::string& card_name, DecisionFn decide);

    GameState& state();
    const GameState& state() const;

    std::vector<std::string> hand_names(int pid) const;
    std::vector<std::string> deck_names(int pid) const;
    std::vector<std::string> discard_names(int pid) const;
    std::vector<std::string> in_play_names(int pid) const;

private:
    GameState state_;

    void ensure_cards_registered();
    void clear_hand(int pid);
    void clear_deck(int pid);
    void clear_discard(int pid);
};
