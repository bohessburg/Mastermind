#include "v2/core/score.h"

#include "v2/core/defs.h"

#include <cstdint>

namespace {

[[nodiscard]] std::int16_t vp_tokens(const PlayerState& player) noexcept {
    return static_cast<std::int16_t>(
        static_cast<std::uint16_t>(player.vp_tokens_lo)
        | (static_cast<std::uint16_t>(player.vp_tokens_hi) << 8U));
}

[[nodiscard]] std::int16_t count_ordered_zone_vp(
    const GameState& state,
    PlayerId player_id,
    const OrderedZone& zone) noexcept {
    std::int16_t total = 0;
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        const DefId def = state.slot_to_def[zone.cards[i]];
        const CardDef& defn = card_def(def);
        const std::int16_t per_card = defn.score_hook == nullptr
            ? defn.vp
            : defn.score_hook(state, player_id, def);
        total = static_cast<std::int16_t>(total + per_card);
    }
    return total;
}

[[nodiscard]] std::int16_t count_zone_vp(
    const GameState& state,
    PlayerId player_id,
    const std::uint8_t (&zone)[MAX_SLOTS]) noexcept {
    std::int16_t total = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        const DefId def = state.slot_to_def[slot];
        const CardDef& defn = card_def(def);
        const std::int16_t per_card = defn.score_hook == nullptr
            ? defn.vp
            : defn.score_hook(state, player_id, def);
        total = static_cast<std::int16_t>(total + static_cast<std::int16_t>(per_card * zone[slot]));
    }
    return total;
}

[[nodiscard]] std::int16_t count_in_play_vp(
    const GameState& state,
    PlayerId player_id,
    const PlayerState& player) noexcept {
    std::int16_t total = 0;
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        const DefId def = state.slot_to_def[player.in_play[i].slot];
        const CardDef& defn = card_def(def);
        const std::int16_t per_card = defn.score_hook == nullptr
            ? defn.vp
            : defn.score_hook(state, player_id, def);
        total = static_cast<std::int16_t>(total + per_card);
    }
    return total;
}

} // namespace

std::int16_t score(const GameState& state, PlayerId player_id) noexcept {
    const PlayerState& player = state.players[player_id];
    std::int16_t total = vp_tokens(player);
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player_id, player.hand));
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player_id, player.exile));
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player_id, player.tavern));
    total = static_cast<std::int16_t>(total + count_zone_vp(state, player_id, player.island_mat));
    total = static_cast<std::int16_t>(total + count_ordered_zone_vp(state, player_id, player.deck));
    total = static_cast<std::int16_t>(total + count_ordered_zone_vp(state, player_id, player.discard));
    total = static_cast<std::int16_t>(total + count_in_play_vp(state, player_id, player));
    return total;
}
