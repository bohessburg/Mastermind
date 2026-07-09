#include "v2/core/interp.h"

#include <cassert>
#include <cstdint>

namespace {

void shuffle_zone(OrderedZone& zone, Xoshiro256pp& rng) noexcept {
    for (int i = static_cast<int>(zone.size); i > 1; --i) {
        const int last = i - 1;
        const std::uint32_t swap_index = rng.uniform(static_cast<std::uint32_t>(i));
        const Slot tmp = zone.cards[last];
        zone.cards[last] = zone.cards[swap_index];
        zone.cards[swap_index] = tmp;
    }
}

void reshuffle_discard_into_deck(PlayerState& player, Xoshiro256pp& rng) noexcept {
    if (player.discard.size == 0U) {
        return;
    }

    for (std::uint8_t i = 0; i < player.discard.size; ++i) {
        player.deck.cards[i] = player.discard.cards[i];
    }
    player.deck.size = player.discard.size;
    player.discard.size = 0;
    shuffle_zone(player.deck, rng);
}

[[nodiscard]] bool draw_one(GameState& state, PlayerId player_id) noexcept {
    PlayerState& player = state.players[player_id];
    if (player.deck.size == 0U) {
        reshuffle_discard_into_deck(player, state.rng);
    }
    if (player.deck.size == 0U) {
        return false;
    }

    --player.deck.size;
    const Slot card = player.deck.cards[player.deck.size];
    assert(card < MAX_SLOTS);
    ++player.hand[card];
    return true;
}

void pop_frame(GameState& state) noexcept {
    assert(state.effect_depth > 0U);
    --state.effect_depth;
}

void add_uint8(std::uint8_t& value, std::int16_t delta) noexcept {
    int next = static_cast<int>(value) + static_cast<int>(delta);
    if (next < 0) {
        next = 0;
    }
    if (next > 255) {
        next = 255;
    }
    value = static_cast<std::uint8_t>(next);
}

} // namespace

void draw_cards(GameState& state, PlayerId player, std::uint8_t count) noexcept {
    for (std::uint8_t i = 0; i < count; ++i) {
        if (!draw_one(state, player)) {
            return;
        }
    }
}

bool push_effect(GameState& state, DefId source, PlayerId player) noexcept {
    assert(source < card_def_count());
    if (source >= card_def_count()) {
        return false;
    }
    const CardDef& def = card_def(source);
    if (def.on_play.len == 0U && def.custom == nullptr) {
        return true;
    }
    assert(state.effect_depth < MAX_EFFECT_DEPTH);
    if (state.effect_depth >= MAX_EFFECT_DEPTH) {
        return false;
    }

    EffectFrame frame{};
    frame.source = source;
    frame.player = player;
    state.effect_stack[state.effect_depth] = frame;
    ++state.effect_depth;
    return true;
}

RunResult interp_run(GameState& state) noexcept {
    while (state.effect_depth > 0U) {
        EffectFrame& frame = state.effect_stack[state.effect_depth - 1U];
        assert(frame.source < card_def_count());
        const CardDef& def = card_def(frame.source);

        if (def.custom != nullptr) {
            const RunResult result = def.custom(state, frame);
            if (result == RunResult::NeedDecision) {
                return result;
            }
            if (result == RunResult::FrameDone) {
                pop_frame(state);
            }
            continue;
        }

        if (frame.pc >= def.on_play.len) {
            pop_frame(state);
            continue;
        }

        const std::uint16_t instr_offset = static_cast<std::uint16_t>(def.on_play.offset + frame.pc);
        assert(instr_offset < effect_instr_count());
        const Instr& instr = effect_instr(instr_offset);

        switch (instr.op) {
        case Op::PlusCards:
            if (instr.arg > 0) {
                draw_cards(state, frame.player, static_cast<std::uint8_t>(instr.arg));
            }
            ++frame.pc;
            break;
        case Op::PlusActions:
            add_uint8(state.actions, instr.arg);
            ++frame.pc;
            break;
        case Op::PlusBuys:
            add_uint8(state.buys, instr.arg);
            ++frame.pc;
            break;
        case Op::PlusCoins:
            state.coins = static_cast<std::int16_t>(state.coins + instr.arg);
            ++frame.pc;
            break;
        case Op::End:
            pop_frame(state);
            break;
        case Op::Choose:
        case Op::ChooseGain:
        case Op::ChooseOption:
        case Op::ChooseOrder:
            assert(false);
            return RunResult::NeedDecision;
        default:
            assert(false);
            pop_frame(state);
            break;
        }
    }

    return RunResult::FrameDone;
}

void interp_resume(GameState& state, Action action) noexcept {
    (void)action;
    (void)interp_run(state);
}
