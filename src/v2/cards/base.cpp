#include "v2/core/defs.h"

namespace {

enum FilterId : std::uint8_t {
    FILTER_HAND_ANY = 0,
    FILTER_HAND_TREASURE = 1,
    FILTER_SUPPLY_COST_4 = 2,
    FILTER_SUPPLY_LAST_PLUS_2 = 3,
    FILTER_SUPPLY_TREASURE_LAST_PLUS_3 = 4,
};

constexpr std::uint8_t CHOOSE_MAX_ALL = 0xFFU;

constexpr EffectSpan SPAN_CELLAR{0, 5};
constexpr EffectSpan SPAN_CHAPEL{5, 2};
constexpr EffectSpan SPAN_VILLAGE{7, 3};
constexpr EffectSpan SPAN_SMITHY{10, 2};
constexpr EffectSpan SPAN_WORKSHOP{12, 2};
constexpr EffectSpan SPAN_REMODEL{14, 4};
constexpr EffectSpan SPAN_MINE{18, 4};
constexpr EffectSpan SPAN_EXACT_TWO_TEST{22, 2};
constexpr EffectSpan SPAN_REPEAT_CHOOSE_TEST{24, 4};

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
    DOMINION_V2_DEF_EFFECT(Cellar, (Cost{2, 0, 0}), TYPE_ACTION, 0, 0, SPAN_CELLAR),
    DOMINION_V2_DEF_EFFECT(Chapel, (Cost{2, 0, 0}), TYPE_ACTION, 0, 0, SPAN_CHAPEL),
    DOMINION_V2_DEF_EFFECT(Village, (Cost{3, 0, 0}), TYPE_ACTION, 0, 0, SPAN_VILLAGE),
    DOMINION_V2_DEF_EFFECT(Smithy, (Cost{4, 0, 0}), TYPE_ACTION, 0, 0, SPAN_SMITHY),
    DOMINION_V2_DEF_EFFECT(Workshop, (Cost{3, 0, 0}), TYPE_ACTION, 0, 0, SPAN_WORKSHOP),
    DOMINION_V2_DEF_EFFECT(Remodel, (Cost{4, 0, 0}), TYPE_ACTION, 0, 0, SPAN_REMODEL),
    DOMINION_V2_DEF_EFFECT(Mine, (Cost{5, 0, 0}), TYPE_ACTION, 0, 0, SPAN_MINE),
    DOMINION_V2_DEF_EFFECT(ExactTwoTest, (Cost{0, 0, 0}), TYPE_ACTION, 0, 0, SPAN_EXACT_TWO_TEST),
    DOMINION_V2_DEF_EFFECT(RepeatChooseTest, (Cost{0, 0, 0}), TYPE_ACTION, 0, 0, SPAN_REPEAT_CHOOSE_TEST),
};

constexpr Instr kEffectInstrs[] = {
    Instr{Op::PlusActions, 0, 0, 0, 1},
    Instr{Op::Choose, FILTER_HAND_ANY, 0, CHOOSE_MAX_ALL, static_cast<std::int16_t>(Then::Discard)},
    Instr{Op::PerChosen, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 1},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_ANY, 0, 4, static_cast<std::int16_t>(Then::Trash)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 1},
    Instr{Op::PlusActions, 0, 0, 0, 2},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 3},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::ChooseGain, FILTER_SUPPLY_COST_4, 1, 1, static_cast<std::int16_t>(GainDestination::Discard)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_ANY, 1, 1, static_cast<std::int16_t>(Then::Trash)},
    Instr{Op::IfElse, static_cast<std::uint8_t>(PredicateId::ChosenAny), 1, 2, 0},
    Instr{Op::ChooseGain, FILTER_SUPPLY_LAST_PLUS_2, 1, 1, static_cast<std::int16_t>(GainDestination::Discard)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_TREASURE, 1, 1, static_cast<std::int16_t>(Then::Trash)},
    Instr{Op::IfElse, static_cast<std::uint8_t>(PredicateId::ChosenAny), 1, 2, 0},
    Instr{Op::ChooseGain, FILTER_SUPPLY_TREASURE_LAST_PLUS_3, 1, 1, static_cast<std::int16_t>(GainDestination::Hand)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_ANY, 2, 2, static_cast<std::int16_t>(Then::Trash)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_ANY, 0, 1, static_cast<std::int16_t>(Then::Discard)},
    Instr{Op::PlusCoins, 0, 0, 0, 1},
    Instr{Op::Repeat, 0, 2, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
};

constexpr Filter kFilters[] = {
    Filter{ZoneSelector::Hand, 0, CostLimitKind::None, Cost{}, 0},
    Filter{ZoneSelector::Hand, TYPE_TREASURE, CostLimitKind::None, Cost{}, 0},
    Filter{ZoneSelector::Supply, 0, CostLimitKind::Fixed, Cost{4, 0, 0}, 0},
    Filter{ZoneSelector::Supply, 0, CostLimitKind::LastChosenPlus, Cost{}, 2},
    Filter{ZoneSelector::Supply, TYPE_TREASURE, CostLimitKind::LastChosenPlus, Cost{}, 3},
};

static_assert(sizeof(kBaseCards) / sizeof(kBaseCards[0]) == BASIC_CARD_COUNT);
static_assert(sizeof(kFilters) / sizeof(kFilters[0]) == 5U);

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

const Filter& filter_def(std::uint8_t id) noexcept {
    return kFilters[id];
}

std::uint8_t filter_count() noexcept {
    return static_cast<std::uint8_t>(sizeof(kFilters) / sizeof(kFilters[0]));
}

const CardDef* base_card_defs() noexcept {
    return kBaseCards;
}

std::uint16_t base_card_def_count() noexcept {
    return BASIC_CARD_COUNT;
}
