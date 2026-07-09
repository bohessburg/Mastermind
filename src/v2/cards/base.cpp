#include "v2/core/defs.h"

namespace {

constexpr CardDef kBaseCards[] = {
    DOMINION_V2_DEF(Copper, (Cost{0, 0, 0}), TYPE_TREASURE, 0, 1),
    DOMINION_V2_DEF(Silver, (Cost{3, 0, 0}), TYPE_TREASURE, 0, 2),
    DOMINION_V2_DEF(Gold, (Cost{6, 0, 0}), TYPE_TREASURE, 0, 3),
    DOMINION_V2_DEF(Platinum, (Cost{9, 0, 0}), TYPE_TREASURE, 0, 5),
    DOMINION_V2_DEF(Potion, (Cost{4, 0, 0}), TYPE_TREASURE, 0, 0),
    DOMINION_V2_DEF(Estate, (Cost{2, 0, 0}), TYPE_VICTORY, 1, 0),
    DOMINION_V2_DEF(Duchy, (Cost{5, 0, 0}), TYPE_VICTORY, 3, 0),
    DOMINION_V2_DEF(Province, (Cost{8, 0, 0}), TYPE_VICTORY, 6, 0),
    DOMINION_V2_DEF(Colony, (Cost{11, 0, 0}), TYPE_VICTORY, 10, 0),
    DOMINION_V2_DEF(Curse, (Cost{0, 0, 0}), TYPE_CURSE, -1, 0),
};

constexpr Instr kEffectInstrs[] = {
    Instr{Op::End, 0, 0, 0, 0},
};

static_assert(sizeof(kBaseCards) / sizeof(kBaseCards[0]) == BASIC_CARD_COUNT);

} // namespace

const CardDef* card_defs() noexcept {
    return kBaseCards;
}

std::uint16_t card_def_count() noexcept {
    return BASIC_CARD_COUNT;
}

const CardDef& card_def(DefId id) noexcept {
    return kBaseCards[id];
}

const Instr& effect_instr(std::uint16_t offset) noexcept {
    return kEffectInstrs[offset];
}

std::uint16_t effect_instr_count() noexcept {
    return static_cast<std::uint16_t>(sizeof(kEffectInstrs) / sizeof(kEffectInstrs[0]));
}

const CardDef* base_card_defs() noexcept {
    return kBaseCards;
}

std::uint16_t base_card_def_count() noexcept {
    return BASIC_CARD_COUNT;
}
