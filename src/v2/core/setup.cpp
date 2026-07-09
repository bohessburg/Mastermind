#include "v2/core/setup.h"

#include "v2/core/defs.h"
#include "v2/core/interp.h"
#include "v2/core/turns.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace {

constexpr std::uint8_t kCopperSupply = 60;
constexpr std::uint8_t kSilverSupply = 40;
constexpr std::uint8_t kGoldSupply = 30;
constexpr std::uint8_t kPlatinumSupply = 12;
constexpr std::uint8_t kPotionSupply = 16;
constexpr std::uint8_t kKingdomSupply = 10;

[[nodiscard]] PlayerId sanitize_player_count(PlayerId count) noexcept {
    if (count < 2U) {
        return 2U;
    }
    if (count > static_cast<PlayerId>(MAX_PLAYERS)) {
        return static_cast<PlayerId>(MAX_PLAYERS);
    }
    return count;
}

[[nodiscard]] std::uint8_t victory_supply_count(PlayerId players) noexcept {
    return players == 2U ? 8U : 12U;
}

[[nodiscard]] bool setup_needs_potion(const Setup& setup) noexcept {
    constexpr Cost kBudgetWithoutPotion{127, 0, 32767};
    for (std::uint8_t i = 0; i < setup.kingdom_count && i < MAX_KINGDOM_DEFS; ++i) {
        const DefId def = setup.kingdom[i];
        if (def < card_def_count() && !card_def(def).cost.fits_within(kBudgetWithoutPotion)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] Slot add_slot(GameState& state, DefId def) noexcept {
    const Slot existing = slot_of(state, def);
    if (existing != NONE) {
        return existing;
    }

    assert(state.num_slots < MAX_SLOTS);
    const Slot slot = state.num_slots;
    state.slot_to_def[slot] = def;
    ++state.num_slots;
    return slot;
}

void add_uniform_pile(GameState& state, DefId def, std::uint8_t count) noexcept {
    assert(state.num_piles < MAX_PILES);
    Pile& pile = state.piles[state.num_piles];
    pile.base = add_slot(state, def);
    pile.count = count;
    ++state.num_piles;
}

void add_card_to_deck(PlayerState& player, Slot slot) noexcept {
    assert(player.deck.size < MAX_DECK_CARDS);
    player.deck.cards[player.deck.size] = slot;
    ++player.deck.size;
}

void shuffle_zone(OrderedZone& zone, Xoshiro256pp& rng) noexcept {
    for (int i = static_cast<int>(zone.size); i > 1; --i) {
        const int last = i - 1;
        const std::uint32_t swap_index = rng.uniform(static_cast<std::uint32_t>(i));
        const Slot tmp = zone.cards[last];
        zone.cards[last] = zone.cards[swap_index];
        zone.cards[swap_index] = tmp;
    }
}

void init_starting_deck(GameState& state, PlayerId player_id, Slot copper, Slot estate) noexcept {
    PlayerState& player = state.players[player_id];
    for (int i = 0; i < 7; ++i) {
        add_card_to_deck(player, copper);
    }
    for (int i = 0; i < 3; ++i) {
        add_card_to_deck(player, estate);
    }
    shuffle_zone(player.deck, state.rng);
    draw_cards(state, player_id, 5U);
}

void add_kingdom_piles(GameState& state, const Setup& setup, PlayerId players) noexcept {
    for (std::uint8_t i = 0; i < setup.kingdom_count && i < MAX_KINGDOM_DEFS; ++i) {
        const DefId def = setup.kingdom[i];
        if (def >= card_def_count() || has_slot(state, def)) {
            continue;
        }
        const CardDef& defn = card_def(def);
        const std::uint8_t count = (defn.types & TYPE_VICTORY) != 0U
            ? victory_supply_count(players)
            : kKingdomSupply;
        add_uniform_pile(state, def, count);
    }
}

} // namespace

GameState new_game(const Setup& setup, std::uint64_t seed) noexcept {
    GameState state{};
    std::memset(&state, 0, sizeof(state));
    state.num_players = sanitize_player_count(setup.num_players);
    state.rng = Xoshiro256pp::seeded(seed);
    state.trigger_table.dirty = 1U;

    for (int i = 0; i < NUM_ARTIFACTS; ++i) {
        state.artifact_holder[i] = NONE;
    }

    const PlayerId players = state.num_players;
    add_uniform_pile(state, DEF_COPPER, static_cast<std::uint8_t>(kCopperSupply - (7U * players)));
    add_uniform_pile(state, DEF_SILVER, kSilverSupply);
    add_uniform_pile(state, DEF_GOLD, kGoldSupply);
    if (setup.use_colony_platinum) {
        add_uniform_pile(state, DEF_PLATINUM, kPlatinumSupply);
    }
    if (setup_needs_potion(setup)) {
        add_uniform_pile(state, DEF_POTION, kPotionSupply);
    }
    add_uniform_pile(state, DEF_ESTATE, victory_supply_count(players));
    add_uniform_pile(state, DEF_DUCHY, victory_supply_count(players));
    add_uniform_pile(state, DEF_PROVINCE, victory_supply_count(players));
    if (setup.use_colony_platinum) {
        add_uniform_pile(state, DEF_COLONY, victory_supply_count(players));
    }
    add_uniform_pile(state, DEF_CURSE, static_cast<std::uint8_t>((players - 1U) * 10U));
    add_kingdom_piles(state, setup, players);

    const Slot copper = slot_of(state, DEF_COPPER);
    const Slot estate = slot_of(state, DEF_ESTATE);
    for (PlayerId player = 0; player < players; ++player) {
        init_starting_deck(state, player, copper, estate);
    }

    turn_queue_clear(state.turn_queue);
    const bool queued = turn_queue_push(state.turn_queue, 0U, TurnKind::Normal);
    (void)queued;
    assert(queued);
    start_turn(state, 0U);
    return state;
}

bool has_slot(const GameState& state, DefId def) noexcept {
    return slot_of(state, def) != NONE;
}

Slot slot_of(const GameState& state, DefId def) noexcept {
    for (std::uint8_t i = 0; i < state.num_slots; ++i) {
        if (state.slot_to_def[i] == def) {
            return i;
        }
    }
    return NONE;
}
