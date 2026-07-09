#include "v2/core/determinize.h"

#include "v2/core/rng.h"

#include <cassert>
#include <cstdint>

namespace {

constexpr int MAX_HIDDEN_POOL = MAX_DECK_CARDS * 2;

void shuffle_slots(Slot* cards, std::uint16_t count, Xoshiro256pp& rng) noexcept {
    for (std::uint16_t i = count; i > 1U; --i) {
        const std::uint16_t last = static_cast<std::uint16_t>(i - 1U);
        const std::uint16_t swap_index =
            static_cast<std::uint16_t>(rng.uniform(static_cast<std::uint32_t>(i)));
        const Slot tmp = cards[last];
        cards[last] = cards[swap_index];
        cards[swap_index] = tmp;
    }
}

void shuffle_zone(OrderedZone& zone, Xoshiro256pp& rng) noexcept {
    shuffle_slots(zone.cards, zone.size, rng);
}

[[nodiscard]] std::uint16_t hand_count(const PlayerState& player, std::uint8_t num_slots) noexcept {
    std::uint16_t total = 0;
    for (std::uint8_t slot = 0; slot < num_slots; ++slot) {
        total = static_cast<std::uint16_t>(total + player.hand[slot]);
    }
    return total;
}

void clear_hand(PlayerState& player, std::uint8_t num_slots) noexcept {
    for (std::uint8_t slot = 0; slot < num_slots; ++slot) {
        player.hand[slot] = 0;
    }
}

void determinize_opponent(PlayerState& player, std::uint8_t num_slots, Xoshiro256pp& rng) noexcept {
    Slot pool[MAX_HIDDEN_POOL]{};
    std::uint16_t pool_size = 0;
    const std::uint16_t target_hand = hand_count(player, num_slots);
    const std::uint8_t target_deck = player.deck.size;

    for (std::uint8_t slot = 0; slot < num_slots; ++slot) {
        for (std::uint8_t count = 0; count < player.hand[slot]; ++count) {
            assert(pool_size < MAX_HIDDEN_POOL);
            pool[pool_size] = slot;
            ++pool_size;
        }
    }
    for (std::uint8_t i = 0; i < player.deck.size; ++i) {
        assert(pool_size < MAX_HIDDEN_POOL);
        pool[pool_size] = player.deck.cards[i];
        ++pool_size;
    }

    shuffle_slots(pool, pool_size, rng);
    clear_hand(player, num_slots);

    std::uint16_t index = 0;
    for (; index < target_hand && index < pool_size; ++index) {
        const Slot slot = pool[index];
        assert(slot < num_slots);
        ++player.hand[slot];
    }

    player.deck.size = 0;
    for (; index < pool_size && player.deck.size < target_deck; ++index) {
        player.deck.cards[player.deck.size] = pool[index];
        ++player.deck.size;
    }
    for (std::uint8_t i = player.deck.size; i < MAX_DECK_CARDS; ++i) {
        player.deck.cards[i] = 0;
    }
}

} // namespace

void determinize(GameState& state, PlayerId perspective, std::uint64_t seed) noexcept {
    if (perspective >= state.num_players) {
        return;
    }

    Xoshiro256pp rng = Xoshiro256pp::seeded(seed);
    shuffle_zone(state.players[perspective].deck, rng);

    for (PlayerId player = 0; player < state.num_players; ++player) {
        if (player == perspective) {
            continue;
        }
        determinize_opponent(state.players[player], state.num_slots, rng);
    }
}
