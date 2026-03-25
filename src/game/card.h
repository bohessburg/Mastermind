#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

// Forward declaration - GameState defined later
class GameState;

// Bitmask for types that affect game mechanics (phase/turn structure)
enum class CardType : uint16_t {
    None     = 0,
    Action   = 1 << 0,
    Treasure = 1 << 1,
    Victory  = 1 << 2,
    Curse    = 1 << 3,
    Attack   = 1 << 4,
    Reaction = 1 << 5,
    Duration = 1 << 6,
    Night    = 1 << 7,
    Command  = 1 << 8
};

// Bitwise operators for combining CardTypes
inline CardType operator|(CardType a, CardType b) {
    return static_cast<CardType>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

inline CardType operator&(CardType a, CardType b) {
    return static_cast<CardType>(static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
}

inline bool has_type(CardType types, CardType flag) {
    return (types & flag) != CardType::None;
}

// What kind of decision a player needs to make mid-card-resolution
enum class ChoiceType {
    DISCARD,
    TRASH,
    GAIN,
    TOPDECK,
    EXILE
};

// Callback for player decisions during card resolution
// Returns indices chosen from the options vector
using DecisionFn = std::function<std::vector<int>(
    int player_id,
    ChoiceType choice_type,
    const std::vector<int>& options,
    int min_choices,
    int max_choices
)>;

struct Card {
    std::string name;
    int cost;
    CardType types;
    std::string text;
    int victory_points;
    int coin_value;
    std::unordered_set<std::string> tags;  // Types that don't really alter game mechanics: "Knight", "Looter", "Spirit", etc.
    std::function<void(GameState&, int, DecisionFn)> on_play;

    bool is_action() const   { return has_type(types, CardType::Action); }
    bool is_treasure() const { return has_type(types, CardType::Treasure); }
    bool is_victory() const  { return has_type(types, CardType::Victory); }
    bool is_curse() const    { return has_type(types, CardType::Curse); }
    bool is_attack() const   { return has_type(types, CardType::Attack); }
    bool is_reaction() const { return has_type(types, CardType::Reaction); }
    bool is_duration() const { return has_type(types, CardType::Duration); }
    bool is_night() const    { return has_type(types, CardType::Night); }
    bool is_command() const  { return has_type(types, CardType::Command); }
    bool has_tag(const std::string& tag) const { return tags.count(tag) > 0; }
};

// Global registry for looking up card definitions by name
namespace CardRegistry {
    void register_card(const Card& card);
    const Card* get(const std::string& name);
    std::vector<std::string> all_names();
};
