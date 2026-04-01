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
    EXILE,
    PLAY_CARD,          // Choose a card to play (Throne Room)
    YES_NO,             // Binary choice (reveal Moat? set aside with Library?)
    REVEAL,             // Reveal from hand (Bureaucrat defense)
    MULTI_FATE,         // Per-card trash/discard/keep (Sentry)
    ORDER,              // Choose ordering (topdeck order for Sentry put-back)
    SELECT_FROM_DISCARD // Choose card from own discard (Harbinger)
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

// Turn flags: fixed enum keys replacing string-keyed map
enum class TurnFlag : int {
    MerchantCount,
    MerchantSilverTriggered,
    SentryCardId,
    SentryOrderCard0,
    SentryOrderCard1,
    COUNT  // must be last
};

struct Card {
    int def_id = -1;  // assigned by CardRegistry at registration time
    std::string name;
    int cost;
    CardType types;
    std::string text;
    int victory_points;
    int coin_value;
    std::unordered_set<std::string> tags;  // "Knight", "Looter", "Spirit", etc.

    // Play ability: resolved when card is played
    std::function<void(GameState&, int, DecisionFn)> on_play;

    // Reaction ability: called when an Attack targets this player and they have
    // this card in hand. Returns true if the attack is blocked for this player.
    std::function<bool(GameState&, int player_id, int attacker_id, DecisionFn)> on_react;

    // Dynamic VP calculation (e.g. Gardens = cards_owned / 10).
    // If set, called during scoring instead of using static victory_points.
    std::function<int(const GameState&, int player_id)> vp_fn;

    // When-gain trigger: called when this card is gained by a player
    std::function<void(GameState&, int player_id, DecisionFn)> on_gain;

    // When-trash trigger: called when this card is trashed
    std::function<void(GameState&, int player_id, DecisionFn)> on_trash;

    // Duration next-turn effect: called at start of owning player's next turn
    std::function<void(GameState&, int player_id, DecisionFn)> on_duration;

    // Discard effect: called when a card with "when discard" is discarded
    std::function<void(GameState&, int player_id, DecisionFn)> on_discard;

    std::function<void(GameState&, int player_id, DecisionFn)> on_reveal;

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
    const Card* get(int def_id);
    int def_id_of(const std::string& name);  // -1 if not found
    std::vector<std::string> all_names();
    int count();
}
