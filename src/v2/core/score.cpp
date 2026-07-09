#include "v2/core/score.h"

#include "v2/core/defs.h"

#include <cstdint>

namespace {

[[nodiscard]] std::int16_t vp_tokens(const PlayerState& player) noexcept {
    return static_cast<std::int16_t>(
        static_cast<std::uint16_t>(player.vp_tokens_lo)
        | (static_cast<std::uint16_t>(player.vp_tokens_hi) << 8U));
}

[[nodiscard]] std::int16_t card_vp(const GameState& state, Slot slot, std::uint8_t count) noexcept {
    const DefId def = state.slot_to_def[slot];
    return static_cast<std::int16_t>(static_cast<std::int16_t>(card_def(def).vp) * count);
}

[[nodiscard]] std::int16_t count_ordered_zone_vp(const GameState& state, const OrderedZone& zone) noexcept {
    std::int16_t total = 0;
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        total = static_cast<std::int16_t>(total + card_vp(state, zone.cards[i], 1U));
    }
    return total;
}

[[nodiscard]] std::int16_t count_zone_vp(
    const GameState& state,
    const std::uint8_t (&zone)[MAX_SLOTS]) noexcept {
    std::int16_t total = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        total = static_cast<std::int16_t>(total + card_vp(state, slot, zone[slot]));
    }
    return total;
}

[[nodiscard]] std::int16_t count_in_play_vp(const GameState& state, const PlayerState& player) noexcept {
    std::int16_t total = 0;
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        total = static_cast<std::int16_t>(total + card_vp(state, player.in_play[i].slot, 1U));
    }
    return total;
}

} // namespace

std::int16_t score(const GameState& state, PlayerId player_id) noexcept {
    const PlayerState& player = state.players[player_id];
    std::int16_t total = vp_tokens(player);
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player.hand));
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player.exile));
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player.tavern));
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player.island_mat));
    total = static_cast<std::int16_t>(total + count_ordered_zone_vp(state, player.deck));
    total = static_cast<std::int16_t>(total + count_ordered_zone_vp(state, player.discard));
    total = static_cast<std::int16_t>(total + count_in_play_vp(state, player));
    return total;
}
