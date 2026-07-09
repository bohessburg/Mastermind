#include "v2/core/triggers.h"

#include "v2/core/interp.h"

#include <cassert>
#include <cstdint>

namespace {

[[nodiscard]] EffectSpan trigger_span_for(const CardDef& def, TriggerKind kind) noexcept {
    switch (kind) {
    case TriggerKind::OnGain:
        return def.on_gain;
    case TriggerKind::OnTrash:
        return def.on_trash;
    case TriggerKind::OnDiscard:
        return def.on_discard;
    case TriggerKind::OnFirstPlay:
        return def.on_gain;
    default:
        return NO_EFFECT;
    }
}

[[nodiscard]] bool trigger_payload_matches(
    const Subscription& sub,
    TriggerKind kind,
    const TriggerPayload& payload) noexcept {
    if (kind == TriggerKind::OnFirstPlay) {
        return sub.owner == payload.player && payload.def == DEF_SILVER;
    }
    return true;
}

void rebuild_trigger_table(GameState& state) noexcept {
    TriggerTable& table = state.trigger_table;
    table.count = 0;

    for (PlayerId player = 0; player < state.num_players; ++player) {
        const PlayerState& player_state = state.players[player];
        for (std::uint8_t i = 0; i < player_state.in_play_size; ++i) {
            const InPlayEntry& entry = player_state.in_play[i];
            const DefId def = state.slot_to_def[entry.behaves_as];
            if (card_def(def).trigger_mask == 0U) {
                continue;
            }
            assert(table.count < MAX_TRIGGER_SUBS);
            if (table.count >= MAX_TRIGGER_SUBS) {
                break;
            }
            table.subs[table.count] = Subscription{player, def, 0U};
            ++table.count;
        }
    }

    table.dirty = 0U;
}

void move_owner_frames_to_top(
    GameState& state,
    std::uint8_t match_count,
    PlayerId owner) noexcept {
    const std::uint8_t base = static_cast<std::uint8_t>(state.effect_depth - match_count);
    EffectFrame ordered[MAX_TRIGGER_SUBS]{};
    std::uint8_t write = 0;

    for (std::uint8_t i = 0; i < match_count; ++i) {
        const EffectFrame& frame = state.effect_stack[static_cast<std::uint8_t>(base + i)];
        if (frame.player != owner) {
            ordered[write] = frame;
            ++write;
        }
    }
    for (std::uint8_t i = 0; i < match_count; ++i) {
        const EffectFrame& frame = state.effect_stack[static_cast<std::uint8_t>(base + i)];
        if (frame.player == owner) {
            ordered[write] = frame;
            ++write;
        }
    }
    for (std::uint8_t i = 0; i < match_count; ++i) {
        state.effect_stack[static_cast<std::uint8_t>(base + i)] = ordered[i];
    }
}

void push_order_window_if_needed(
    GameState& state,
    const Subscription (&matches)[MAX_TRIGGER_SUBS],
    std::uint8_t match_count,
    DefId source) noexcept {
    for (PlayerId player = 0; player < state.num_players; ++player) {
        std::uint8_t count = 0;
        for (std::uint8_t i = 0; i < match_count; ++i) {
            if (matches[i].owner == player) {
                ++count;
            }
        }
        if (count > 1U) {
            move_owner_frames_to_top(state, match_count, player);
            const bool pushed = push_trigger_order_frame(state, player, source, count);
            (void)pushed;
            assert(pushed);
            return;
        }
    }
}

} // namespace

void mark_trigger_table_dirty(GameState& state) noexcept {
    state.trigger_table.dirty = 1U;
}

void emit(GameState& state, TriggerKind kind, TriggerPayload payload) noexcept {
    if (state.trigger_table.dirty != 0U) {
        rebuild_trigger_table(state);
    }

    Subscription matches[MAX_TRIGGER_SUBS]{};
    std::uint8_t match_count = 0;
    const std::uint32_t mask = trigger_mask(kind);

    for (std::uint8_t i = 0; i < state.trigger_table.count; ++i) {
        const Subscription& sub = state.trigger_table.subs[i];
        const CardDef& def = card_def(sub.source);
        if ((def.trigger_mask & mask) == 0U || !trigger_payload_matches(sub, kind, payload)) {
            continue;
        }
        const EffectSpan span = trigger_span_for(def, kind);
        if (span.len == 0U) {
            continue;
        }
        matches[match_count] = sub;
        ++match_count;
    }

    if (match_count == 0U) {
        return;
    }

    for (std::uint8_t reverse = match_count; reverse > 0U; --reverse) {
        const Subscription& sub = matches[reverse - 1U];
        const EffectSpan span = trigger_span_for(card_def(sub.source), kind);
        const bool pushed = push_effect_span(state, sub.source, sub.owner, span, FRAME_ABSOLUTE_PROGRAM);
        (void)pushed;
        assert(pushed);
    }

    push_order_window_if_needed(state, matches, match_count, payload.def);
}
