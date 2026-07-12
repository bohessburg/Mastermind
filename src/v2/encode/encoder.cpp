#include "v2/encode/encoder.h"

#include "v2/core/defs.h"

#include <cstddef>
#include <cstdint>

namespace {

[[nodiscard]] float id_or_zero(std::uint16_t id, std::uint16_t limit) noexcept {
    return id < limit ? static_cast<float>(id + 1U) : 0.0F;
}

[[nodiscard]] PlayerId current_player(const GameState& state) noexcept {
    if (state.turn_queue.size == 0U) {
        return 0U;
    }
    return state.turn_queue.entries[state.turn_queue.head].player;
}

[[nodiscard]] std::int16_t vp_tokens(const PlayerState& player) noexcept {
    return static_cast<std::int16_t>(
        static_cast<std::uint16_t>(player.vp_tokens_lo)
        | (static_cast<std::uint16_t>(player.vp_tokens_hi) << 8U));
}

[[nodiscard]] Slot pile_top_slot(const Pile& pile) noexcept {
    if (pile.mixed_len > 0U) {
        return pile.mixed[pile.mixed_len - 1U];
    }
    if (pile.count > 0U) {
        return pile.base;
    }
    return NONE;
}

[[nodiscard]] float pile_top_def_id(const GameState& state, const Pile& pile) noexcept {
    const Slot slot = pile_top_slot(pile);
    if (slot == NONE || slot >= state.num_slots) {
        return 0.0F;
    }
    return id_or_zero(state.slot_to_def[slot], card_def_count());
}

[[nodiscard]] float zone_top_def_id(const GameState& state, const OrderedZone& zone) noexcept {
    if (zone.size == 0U) {
        return 0.0F;
    }
    const Slot slot = zone.cards[zone.size - 1U];
    if (slot >= state.num_slots) {
        return 0.0F;
    }
    return id_or_zero(state.slot_to_def[slot], card_def_count());
}

void clear(float* out) noexcept {
    for (std::size_t i = 0; i < OBS_SIZE; ++i) {
        out[i] = 0.0F;
    }
}

void add_ordered_composition(const GameState& state, const OrderedZone& zone, float* out) noexcept {
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        const Slot slot = zone.cards[i];
        if (slot < state.num_slots) {
            out[slot] += 1.0F;
        }
    }
}

void add_in_play_composition(const PlayerState& player, float* out) noexcept {
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        const Slot slot = player.in_play[i].slot;
        if (slot < MAX_SLOTS) {
            out[slot] += 1.0F;
        }
    }
}

void encode_meta(const GameState& state, PlayerId perspective, float* out) noexcept {
    out[OBS_META_OFFSET + 0U] = static_cast<float>(OBS_VERSION);
    out[OBS_META_OFFSET + 1U] = static_cast<float>(OBS_SIZE);
    out[OBS_META_OFFSET + 2U] = static_cast<float>(perspective);
    out[OBS_META_OFFSET + 3U] = static_cast<float>(state.num_slots);
}

void encode_own(const GameState& state, PlayerId perspective, float* out) noexcept {
    if (perspective >= state.num_players) {
        return;
    }

    const PlayerState& player = state.players[perspective];
    float* section = out + OBS_OWN_OFFSET;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        section[slot] = static_cast<float>(player.hand[slot]);
    }
    add_ordered_composition(state, player.deck, section + MAX_SLOTS);
    add_ordered_composition(state, player.discard, section + (2U * MAX_SLOTS));
    add_in_play_composition(player, section + (3U * MAX_SLOTS));
    add_ordered_composition(state, player.set_aside, section + (4U * MAX_SLOTS));
}

void encode_opponent_block(
    const GameState& state,
    PlayerId perspective,
    std::uint8_t block_index,
    PlayerId player_id,
    float* out) noexcept {
    float* block = out + OBS_OPPONENT_OFFSET + (block_index * OBS_OPPONENT_BLOCK_SIZE);
    if (player_id >= state.num_players || player_id == perspective) {
        return;
    }

    const PlayerState& player = state.players[player_id];
    block[0] = 1.0F;
    block[1] = static_cast<float>(player_id);
    block[2] = static_cast<float>(player.deck.size);
    block[3] = static_cast<float>(player.discard.size);
    block[4] = zone_top_def_id(state, player.discard);
    std::uint16_t hand_count = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        hand_count = static_cast<std::uint16_t>(hand_count + player.hand[slot]);
    }
    block[5] = static_cast<float>(hand_count);
    add_in_play_composition(player, block + 6U);
    block[70] = static_cast<float>(vp_tokens(player));
    block[71] = static_cast<float>(player.debt);
    block[72] = static_cast<float>(player.coffers);
    block[73] = static_cast<float>(player.villagers);
    block[74] = static_cast<float>(player.favors);
}

void encode_opponents(const GameState& state, PlayerId perspective, float* out) noexcept {
    if (state.num_players == 0U || perspective >= state.num_players) {
        return;
    }

    std::uint8_t write = 0;
    for (std::uint8_t offset = 1U; offset < state.num_players && write < (MAX_PLAYERS - 1); ++offset) {
        const PlayerId player_id = static_cast<PlayerId>((perspective + offset) % state.num_players);
        encode_opponent_block(state, perspective, write, player_id, out);
        ++write;
    }
}

void encode_supply(const GameState& state, float* out) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        float* block = out + OBS_SUPPLY_OFFSET + (i * OBS_PILE_BLOCK_SIZE);
        block[0] = static_cast<float>(pile.mixed_len > 0U ? pile.mixed_len : pile.count);
        block[1] = pile_top_def_id(state, pile);
        block[2] = pile.base < state.num_slots
            ? id_or_zero(state.slot_to_def[pile.base], card_def_count())
            : 0.0F;
        block[3] = static_cast<float>(pile.mixed_len);
        block[4] = pile.trait == NO_LANDSCAPE ? 0.0F : static_cast<float>(pile.trait + 1U);
        block[5] = static_cast<float>(pile.embargo);
        block[6] = static_cast<float>(pile.gain_counter);
        for (PlayerId player = 0; player < MAX_PLAYERS; ++player) {
            block[7U + player] = static_cast<float>(pile.adv_tokens[player]);
        }
    }
}

void encode_landscapes(const GameState& state, float* out) noexcept {
    float* section = out + OBS_LANDSCAPE_OFFSET;
    std::size_t index = 0;
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.events[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.events[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.ways[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.ways[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.landmarks[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.landmarks[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.projects[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.projects[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = static_cast<float>(state.project_bought[i]);
        ++index;
    }
    section[index] = state.prophecy == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.prophecy + 1U);
    ++index;
    section[index] = static_cast<float>(state.sun_tokens);
    ++index;
    for (std::uint8_t i = 0; i < NUM_ARTIFACTS; ++i) {
        section[index] = state.artifact_holder[i] == NONE ? 0.0F : static_cast<float>(state.artifact_holder[i] + 1U);
        ++index;
    }
}

void encode_resources(const GameState& state, PlayerId perspective, float* out) noexcept {
    float* section = out + OBS_RESOURCE_OFFSET;
    const PlayerState* player = perspective < state.num_players ? &state.players[perspective] : nullptr;
    section[0] = static_cast<float>(state.actions);
    section[1] = static_cast<float>(state.buys);
    section[2] = static_cast<float>(state.coins);
    section[3] = static_cast<float>(state.potion_coins);
    section[4] = player == nullptr ? 0.0F : static_cast<float>(player->debt);
    section[5] = player == nullptr ? 0.0F : static_cast<float>(player->coffers);
    section[6] = player == nullptr ? 0.0F : static_cast<float>(player->villagers);
    section[7] = player == nullptr ? 0.0F : static_cast<float>(player->favors);
    section[8] = player == nullptr ? 0.0F : static_cast<float>(vp_tokens(*player));
    section[9] = player == nullptr ? 0.0F : static_cast<float>(player->journey_up);
    section[10] = player == nullptr ? 0.0F : static_cast<float>(player->minus_card);
    section[11] = player == nullptr ? 0.0F : static_cast<float>(player->minus_coin);
}

void encode_turn(const GameState& state, PlayerId perspective, float* out) noexcept {
    float* section = out + OBS_TURN_OFFSET;
    const std::uint8_t phase = state.phase;
    if (phase < OBS_PHASE_COUNT) {
        section[phase] = 1.0F;
    }
    const PlayerId active = current_player(state);
    section[OBS_PHASE_COUNT + 0U] = static_cast<float>(active);
    section[OBS_PHASE_COUNT + 1U] = active == perspective ? 1.0F : 0.0F;
    section[OBS_PHASE_COUNT + 2U] = static_cast<float>(state.turn_counter);
    section[OBS_PHASE_COUNT + 3U] = static_cast<float>(state.truncated);
    section[OBS_PHASE_COUNT + 4U] = static_cast<float>(state.effect_depth);
}

void encode_decision(const GameState& state, PlayerId perspective, float* out) noexcept {
    float* section = out + OBS_DECISION_OFFSET;
    const std::uint8_t kind = state.decision.kind;
    if (kind < OBS_DECISION_KIND_COUNT) {
        section[kind] = 1.0F;
    }
    const std::size_t base = OBS_DECISION_KIND_COUNT;
    section[base + 0U] = static_cast<float>(state.decision.player);
    section[base + 1U] = state.decision.player == perspective ? 1.0F : 0.0F;
    section[base + 2U] = kind == 0U ? 0.0F : id_or_zero(state.decision.source, card_def_count());
    section[base + 3U] = static_cast<float>(state.decision.min_left);
    section[base + 4U] = static_cast<float>(state.decision.max_left);
}

void clear_v2(float* out) noexcept {
    for (std::size_t i = 0; i < OBS_SIZE_V2; ++i) {
        out[i] = 0.0F;
    }
}

void add_hand_composition(const PlayerState& player, std::uint8_t num_slots, float* out) noexcept {
    for (std::uint8_t slot = 0; slot < num_slots; ++slot) {
        out[slot] += static_cast<float>(player.hand[slot]);
    }
}

void encode_meta_v2(const GameState& state, PlayerId perspective, float* out) noexcept {
    out[OBS_V2_META_OFFSET + 0U] = static_cast<float>(ObsVersion::V2);
    out[OBS_V2_META_OFFSET + 1U] = static_cast<float>(OBS_SIZE_V2);
    out[OBS_V2_META_OFFSET + 2U] = static_cast<float>(perspective);
    out[OBS_V2_META_OFFSET + 3U] = static_cast<float>(state.num_slots);
}

void encode_own_v2(const GameState& state, PlayerId perspective, float* out) noexcept {
    if (perspective >= state.num_players) {
        return;
    }

    const PlayerState& player = state.players[perspective];
    float* section = out + OBS_V2_OWN_OFFSET;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        section[slot] = static_cast<float>(player.hand[slot]);
    }
    add_ordered_composition(state, player.deck, section + MAX_SLOTS);
    add_ordered_composition(state, player.discard, section + (2U * MAX_SLOTS));
    add_in_play_composition(player, section + (3U * MAX_SLOTS));
    add_ordered_composition(state, player.set_aside, section + (4U * MAX_SLOTS));
}

void encode_opponent_block_v2(
    const GameState& state,
    PlayerId perspective,
    std::uint8_t block_index,
    PlayerId player_id,
    float* out) noexcept {
    float* block = out + OBS_V2_OPPONENT_OFFSET + (block_index * OBS_OPPONENT_BLOCK_SIZE_V2);
    if (player_id >= state.num_players || player_id == perspective) {
        return;
    }

    const PlayerState& player = state.players[player_id];
    // Preserve the complete v1 block prefix for layout continuity.
    block[0] = 1.0F;
    block[1] = static_cast<float>(player_id);
    block[2] = static_cast<float>(player.deck.size);
    block[3] = static_cast<float>(player.discard.size);
    block[4] = zone_top_def_id(state, player.discard);
    std::uint16_t hand_count = 0;
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        hand_count = static_cast<std::uint16_t>(hand_count + player.hand[slot]);
    }
    block[5] = static_cast<float>(hand_count);
    add_in_play_composition(player, block + 6U);
    block[70] = static_cast<float>(vp_tokens(player));
    block[71] = static_cast<float>(player.debt);
    block[72] = static_cast<float>(player.coffers);
    block[73] = static_cast<float>(player.villagers);
    block[74] = static_cast<float>(player.favors);

    float* collection = block + OBS_V2_OPPONENT_COLLECTION_OFFSET;
    add_hand_composition(player, state.num_slots, collection);
    add_ordered_composition(state, player.deck, collection);
    add_ordered_composition(state, player.discard, collection);
    add_in_play_composition(player, collection);
    add_ordered_composition(state, player.set_aside, collection);

    add_ordered_composition(state, player.discard, block + OBS_V2_OPPONENT_DISCARD_OFFSET);
    add_ordered_composition(state, player.set_aside, block + OBS_V2_OPPONENT_SET_ASIDE_OFFSET);
}

void encode_opponents_v2(const GameState& state, PlayerId perspective, float* out) noexcept {
    if (state.num_players == 0U || perspective >= state.num_players) {
        return;
    }

    std::uint8_t write = 0;
    for (std::uint8_t offset = 1U; offset < state.num_players && write < (MAX_PLAYERS - 1); ++offset) {
        const PlayerId player_id = static_cast<PlayerId>((perspective + offset) % state.num_players);
        encode_opponent_block_v2(state, perspective, write, player_id, out);
        ++write;
    }
}

void encode_supply_v2(const GameState& state, float* out) noexcept {
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        float* block = out + OBS_V2_SUPPLY_OFFSET + (i * OBS_PILE_BLOCK_SIZE);
        block[0] = static_cast<float>(pile.mixed_len > 0U ? pile.mixed_len : pile.count);
        block[1] = pile_top_def_id(state, pile);
        block[2] = pile.base < state.num_slots
            ? id_or_zero(state.slot_to_def[pile.base], card_def_count())
            : 0.0F;
        block[3] = static_cast<float>(pile.mixed_len);
        block[4] = pile.trait == NO_LANDSCAPE ? 0.0F : static_cast<float>(pile.trait + 1U);
        block[5] = static_cast<float>(pile.embargo);
        block[6] = static_cast<float>(pile.gain_counter);
        for (PlayerId player = 0; player < MAX_PLAYERS; ++player) {
            block[7U + player] = static_cast<float>(pile.adv_tokens[player]);
        }
    }
}

void encode_landscapes_v2(const GameState& state, float* out) noexcept {
    float* section = out + OBS_V2_LANDSCAPE_OFFSET;
    std::size_t index = 0;
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.events[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.events[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.ways[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.ways[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.landmarks[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.landmarks[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = state.projects[i] == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.projects[i] + 1U);
        ++index;
    }
    for (std::uint8_t i = 0; i < MAX_LANDSCAPES; ++i) {
        section[index] = static_cast<float>(state.project_bought[i]);
        ++index;
    }
    section[index] = state.prophecy == NO_LANDSCAPE ? 0.0F : static_cast<float>(state.prophecy + 1U);
    ++index;
    section[index] = static_cast<float>(state.sun_tokens);
    ++index;
    for (std::uint8_t i = 0; i < NUM_ARTIFACTS; ++i) {
        section[index] = state.artifact_holder[i] == NONE ? 0.0F : static_cast<float>(state.artifact_holder[i] + 1U);
        ++index;
    }
}

void encode_resources_v2(const GameState& state, PlayerId perspective, float* out) noexcept {
    float* section = out + OBS_V2_RESOURCE_OFFSET;
    const PlayerState* player = perspective < state.num_players ? &state.players[perspective] : nullptr;
    section[0] = static_cast<float>(state.actions);
    section[1] = static_cast<float>(state.buys);
    section[2] = static_cast<float>(state.coins);
    section[3] = static_cast<float>(state.potion_coins);
    section[4] = player == nullptr ? 0.0F : static_cast<float>(player->debt);
    section[5] = player == nullptr ? 0.0F : static_cast<float>(player->coffers);
    section[6] = player == nullptr ? 0.0F : static_cast<float>(player->villagers);
    section[7] = player == nullptr ? 0.0F : static_cast<float>(player->favors);
    section[8] = player == nullptr ? 0.0F : static_cast<float>(vp_tokens(*player));
    section[9] = player == nullptr ? 0.0F : static_cast<float>(player->journey_up);
    section[10] = player == nullptr ? 0.0F : static_cast<float>(player->minus_card);
    section[11] = player == nullptr ? 0.0F : static_cast<float>(player->minus_coin);
}

void encode_turn_v2(const GameState& state, PlayerId perspective, float* out) noexcept {
    float* section = out + OBS_V2_TURN_OFFSET;
    const std::uint8_t phase = state.phase;
    if (phase < OBS_PHASE_COUNT) {
        section[phase] = 1.0F;
    }
    const PlayerId active = current_player(state);
    section[OBS_PHASE_COUNT + 0U] = static_cast<float>(active);
    section[OBS_PHASE_COUNT + 1U] = active == perspective ? 1.0F : 0.0F;
    section[OBS_PHASE_COUNT + 2U] = static_cast<float>(state.turn_counter);
    section[OBS_PHASE_COUNT + 3U] = static_cast<float>(state.truncated);
    section[OBS_PHASE_COUNT + 4U] = static_cast<float>(state.effect_depth);
}

void encode_decision_v2(const GameState& state, PlayerId perspective, float* out) noexcept {
    float* section = out + OBS_V2_DECISION_OFFSET;
    const std::uint8_t kind = state.decision.kind;
    if (kind < OBS_DECISION_KIND_COUNT) {
        section[kind] = 1.0F;
    }
    const std::size_t base = OBS_DECISION_KIND_COUNT;
    section[base + 0U] = static_cast<float>(state.decision.player);
    section[base + 1U] = state.decision.player == perspective ? 1.0F : 0.0F;
    section[base + 2U] = kind == 0U ? 0.0F : id_or_zero(state.decision.source, card_def_count());
    section[base + 3U] = static_cast<float>(state.decision.min_left);
    section[base + 4U] = static_cast<float>(state.decision.max_left);
}

} // namespace

void encode_v1(const GameState& state, PlayerId perspective, float* out) noexcept {
    if (out == nullptr) {
        return;
    }

    clear(out);
    encode_meta(state, perspective, out);
    encode_own(state, perspective, out);
    encode_opponents(state, perspective, out);
    encode_supply(state, out);
    encode_landscapes(state, out);
    encode_resources(state, perspective, out);
    encode_turn(state, perspective, out);
    encode_decision(state, perspective, out);
}

void encode_v2(const GameState& state, PlayerId perspective, float* out) noexcept {
    if (out == nullptr) {
        return;
    }

    clear_v2(out);
    encode_meta_v2(state, perspective, out);
    encode_own_v2(state, perspective, out);
    encode_opponents_v2(state, perspective, out);
    encode_supply_v2(state, out);
    encode_landscapes_v2(state, out);
    encode_resources_v2(state, perspective, out);
    encode_turn_v2(state, perspective, out);
    encode_decision_v2(state, perspective, out);
}

void encode(const GameState& state, PlayerId perspective, float* out, ObsVersion version) noexcept {
    if (version == ObsVersion::V2) {
        encode_v2(state, perspective, out);
        return;
    }
    encode_v1(state, perspective, out);
}
