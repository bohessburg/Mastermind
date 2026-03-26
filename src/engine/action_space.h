#pragma once

#include "../game/game_state.h"

#include <cstdint>
#include <string>
#include <vector>

enum class DecisionType : uint8_t {
    PLAY_ACTION,
    PLAY_TREASURE,
    PLAY_ALL_TREASURES,
    BUY_CARD,
    CHOOSE_CARDS_IN_HAND,
    CHOOSE_SUPPLY_PILE,
    CHOOSE_OPTION,
    REACT,
    PLAY_NIGHT,
};

struct ActionOption {
    int local_id;
    int card_id;
    std::string card_name;
    std::string label;
    bool is_pass;
};

struct DecisionPoint {
    int player_id;
    DecisionType type;
    std::vector<ActionOption> options;
    std::string source_card;
    ChoiceType sub_choice_type;
    int min_choices;
    int max_choices;

    int num_options() const { return static_cast<int>(options.size()); }
};

DecisionPoint build_action_decision(const GameState& state);
DecisionPoint build_treasure_decision(const GameState& state);
DecisionPoint build_buy_decision(const GameState& state);
