#include "v2/core/defs.h"
#include "v2/core/triggers.h"

namespace {

enum FilterId : std::uint8_t {
    FILTER_HAND_ANY = 0,
    FILTER_HAND_TREASURE = 1,
    FILTER_SUPPLY_COST_4 = 2,
    FILTER_SUPPLY_LAST_PLUS_2 = 3,
    FILTER_SUPPLY_TREASURE_LAST_PLUS_3 = 4,
    FILTER_HAND_VICTORY = 5,
    FILTER_HAND_COPPER = 6,
    FILTER_HAND_ACTION = 7,
    FILTER_DISCARD_ANY = 8,
    FILTER_SUPPLY_COST_5 = 9,
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
constexpr EffectSpan SPAN_MERCHANT{28, 3};
constexpr EffectSpan SPAN_MERCHANT_ON_FIRST_PLAY{31, 2};
constexpr EffectSpan SPAN_MILITIA{33, 3};
constexpr EffectSpan SPAN_WITCH{38, 3};
constexpr EffectSpan SPAN_MOAT{43, 2};
constexpr EffectSpan SPAN_BUREAUCRAT{45, 3};
constexpr EffectSpan SPAN_ORDER_ALPHA_ON_FIRST_PLAY{50, 2};
constexpr EffectSpan SPAN_ORDER_BETA_ON_FIRST_PLAY{52, 5};
constexpr EffectSpan SPAN_ORDER_GAMMA_ON_FIRST_PLAY{57, 5};
constexpr EffectSpan SPAN_MARKET{62, 5};
constexpr EffectSpan SPAN_FESTIVAL{67, 4};
constexpr EffectSpan SPAN_LABORATORY{71, 3};
constexpr EffectSpan SPAN_MONEYLENDER{77, 4};
constexpr EffectSpan SPAN_POACHER{81, 5};
constexpr EffectSpan SPAN_VASSAL{86, 7};
constexpr EffectSpan SPAN_HARBINGER{93, 4};
constexpr EffectSpan SPAN_THRONE_ROOM{97, 3};
constexpr EffectSpan SPAN_COUNCIL_ROOM{100, 4};
constexpr EffectSpan SPAN_ARTISAN{106, 3};
constexpr EffectSpan SPAN_BANDIT{109, 3};

[[nodiscard]] int count_ordered(const OrderedZone& zone) noexcept {
    return zone.size;
}

[[nodiscard]] int count_zone(const std::uint8_t (&zone)[MAX_SLOTS]) noexcept {
    int total = 0;
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        total += zone[slot];
    }
    return total;
}

[[nodiscard]] std::int16_t gardens_score(const GameState& state, PlayerId player_id, DefId) noexcept {
    const PlayerState& player = state.players[player_id];
    int total = 0;
    total += count_zone(player.hand);
    total += count_zone(player.exile);
    total += count_zone(player.tavern);
    total += count_zone(player.island_mat);
    total += count_ordered(player.deck);
    total += count_ordered(player.discard);
    total += player.in_play_size;
    return static_cast<std::int16_t>(total / 10);
}

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
    CardDef{
        "Merchant",
        Cost{3, 0, 0},
        TYPE_ACTION,
        0,
        0,
        SPAN_MERCHANT,
        SPAN_MERCHANT_ON_FIRST_PLAY,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        trigger_mask(TriggerKind::OnFirstPlay),
        nullptr,
        nullptr,
    },
    DOMINION_V2_DEF_EFFECT(Militia, (Cost{4, 0, 0}), static_cast<std::uint16_t>(TYPE_ACTION | TYPE_ATTACK), 0, 0, SPAN_MILITIA),
    DOMINION_V2_DEF_EFFECT(Witch, (Cost{5, 0, 0}), static_cast<std::uint16_t>(TYPE_ACTION | TYPE_ATTACK), 0, 0, SPAN_WITCH),
    DOMINION_V2_DEF_EFFECT(Moat, (Cost{2, 0, 0}), static_cast<std::uint16_t>(TYPE_ACTION | TYPE_REACTION), 0, 0, SPAN_MOAT),
    DOMINION_V2_DEF_EFFECT(Bureaucrat, (Cost{4, 0, 0}), static_cast<std::uint16_t>(TYPE_ACTION | TYPE_ATTACK), 0, 0, SPAN_BUREAUCRAT),
    CardDef{
        "OrderAlphaTest",
        Cost{0, 0, 0},
        TYPE_ACTION,
        0,
        0,
        NO_EFFECT,
        SPAN_ORDER_ALPHA_ON_FIRST_PLAY,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        trigger_mask(TriggerKind::OnFirstPlay),
        nullptr,
        nullptr,
    },
    CardDef{
        "OrderBetaTest",
        Cost{0, 0, 0},
        TYPE_ACTION,
        0,
        0,
        NO_EFFECT,
        SPAN_ORDER_BETA_ON_FIRST_PLAY,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        trigger_mask(TriggerKind::OnFirstPlay),
        nullptr,
        nullptr,
    },
    CardDef{
        "OrderGammaTest",
        Cost{0, 0, 0},
        TYPE_ACTION,
        0,
        0,
        NO_EFFECT,
        SPAN_ORDER_GAMMA_ON_FIRST_PLAY,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        trigger_mask(TriggerKind::OnFirstPlay),
        nullptr,
        nullptr,
    },
    DOMINION_V2_DEF_EFFECT(Market, (Cost{5, 0, 0}), TYPE_ACTION, 0, 0, SPAN_MARKET),
    DOMINION_V2_DEF_EFFECT(Festival, (Cost{5, 0, 0}), TYPE_ACTION, 0, 0, SPAN_FESTIVAL),
    DOMINION_V2_DEF_EFFECT(Laboratory, (Cost{5, 0, 0}), TYPE_ACTION, 0, 0, SPAN_LABORATORY),
    CardDef{
        "Gardens",
        Cost{4, 0, 0},
        TYPE_VICTORY,
        0,
        0,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        0U,
        nullptr,
        gardens_score,
    },
    DOMINION_V2_DEF_EFFECT(Moneylender, (Cost{4, 0, 0}), TYPE_ACTION, 0, 0, SPAN_MONEYLENDER),
    DOMINION_V2_DEF_EFFECT(Poacher, (Cost{4, 0, 0}), TYPE_ACTION, 0, 0, SPAN_POACHER),
    DOMINION_V2_DEF_EFFECT(Vassal, (Cost{3, 0, 0}), TYPE_ACTION, 0, 0, SPAN_VASSAL),
    DOMINION_V2_DEF_EFFECT(Harbinger, (Cost{3, 0, 0}), TYPE_ACTION, 0, 0, SPAN_HARBINGER),
    CardDef{
        "Throne Room",
        Cost{4, 0, 0},
        TYPE_ACTION,
        0,
        0,
        SPAN_THRONE_ROOM,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        0U,
        nullptr,
        nullptr,
    },
    CardDef{
        "Council Room",
        Cost{5, 0, 0},
        TYPE_ACTION,
        0,
        0,
        SPAN_COUNCIL_ROOM,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        NO_EFFECT,
        0U,
        nullptr,
        nullptr,
    },
    DOMINION_V2_DEF_EFFECT(Artisan, (Cost{6, 0, 0}), TYPE_ACTION, 0, 0, SPAN_ARTISAN),
    DOMINION_V2_DEF_EFFECT(Bandit, (Cost{5, 0, 0}), static_cast<std::uint16_t>(TYPE_ACTION | TYPE_ATTACK), 0, 0, SPAN_BANDIT),
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
    Instr{Op::PlusCards, 0, 0, 0, 1},
    Instr{Op::PlusActions, 0, 0, 0, 1},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCoins, 0, 0, 0, 1},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCoins, 0, 0, 0, 2},
    Instr{Op::Attack, 36, 0, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::DiscardDownTo, 0, 0, 0, 3},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 2},
    Instr{Op::Attack, 41, 0, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::GainCurse, static_cast<std::uint8_t>(GainDestination::Discard), 0, 0, DEF_CURSE},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 2},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::GainSpecific, static_cast<std::uint8_t>(GainDestination::Topdeck), 0, 0, DEF_SILVER},
    Instr{Op::Attack, 48, 0, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_VICTORY, 1, 1, static_cast<std::int16_t>(Then::Topdeck)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCoins, 0, 0, 0, 2},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::IfElse, static_cast<std::uint8_t>(PredicateId::CoinsAtLeastArg), 1, 3, 3},
    Instr{Op::PlusCoins, 0, 0, 0, 10},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCoins, 0, 0, 0, 1},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::IfElse, static_cast<std::uint8_t>(PredicateId::CoinsAtLeastArg), 1, 3, 10},
    Instr{Op::PlusCoins, 0, 0, 0, 100},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCoins, 0, 0, 0, 5},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 1},
    Instr{Op::PlusActions, 0, 0, 0, 1},
    Instr{Op::PlusBuys, 0, 0, 0, 1},
    Instr{Op::PlusCoins, 0, 0, 0, 1},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusActions, 0, 0, 0, 2},
    Instr{Op::PlusBuys, 0, 0, 0, 1},
    Instr{Op::PlusCoins, 0, 0, 0, 2},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 2},
    Instr{Op::PlusActions, 0, 0, 0, 1},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusBuys, 0, 0, 0, 1},
    Instr{Op::PlusCoins, 0, 0, 0, 2},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_COPPER, 0, 1, static_cast<std::int16_t>(Then::Trash)},
    Instr{Op::IfElse, static_cast<std::uint8_t>(PredicateId::ChosenAny), 1, 2, 0},
    Instr{Op::PlusCoins, 0, 0, 0, 3},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 1},
    Instr{Op::PlusActions, 0, 0, 0, 1},
    Instr{Op::PlusCoins, 0, 0, 0, 1},
    Instr{Op::DiscardPerEmptySupply, FILTER_HAND_ANY, 0, 0, static_cast<std::int16_t>(Then::Discard)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCoins, 0, 0, 0, 2},
    Instr{Op::DiscardDeckTop, 0, 0, 0, 0},
    Instr{Op::IfElse, static_cast<std::uint8_t>(PredicateId::LastChosenIsAction), 1, 4, 0},
    Instr{Op::ChooseOption, 2, 0, 0, 0},
    Instr{Op::IfElse, static_cast<std::uint8_t>(PredicateId::LastOptionEqualsArg), 1, 2, 1},
    Instr{Op::PlayLastFromDiscard, 0, 0, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 1},
    Instr{Op::PlusActions, 0, 0, 0, 1},
    Instr{Op::Choose, FILTER_DISCARD_ANY, 0, 1, static_cast<std::int16_t>(Then::Topdeck)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::Choose, FILTER_HAND_ACTION, 0, 1, static_cast<std::int16_t>(Then::Play)},
    Instr{Op::PlayChosenRepeated, 0, 0, 0, 2},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 4},
    Instr{Op::PlusBuys, 0, 0, 0, 1},
    Instr{Op::EachOtherPlayer, 104, 0, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::PlusCards, 0, 0, 0, 1},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::ChooseGain, FILTER_SUPPLY_COST_5, 1, 1, static_cast<std::int16_t>(GainDestination::Hand)},
    Instr{Op::Choose, FILTER_HAND_ANY, 1, 1, static_cast<std::int16_t>(Then::Topdeck)},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::GainSpecific, static_cast<std::uint8_t>(GainDestination::Discard), 0, 0, DEF_GOLD},
    Instr{Op::Attack, 112, 0, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
    Instr{Op::BanditAttack, 0, 0, 0, 0},
    Instr{Op::End, 0, 0, 0, 0},
};

constexpr Filter kFilters[] = {
    Filter{ZoneSelector::Hand, 0, CostLimitKind::None, Cost{}, 0, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Hand, TYPE_TREASURE, CostLimitKind::None, Cost{}, 0, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Supply, 0, CostLimitKind::Fixed, Cost{4, 0, 0}, 0, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Supply, 0, CostLimitKind::LastChosenPlus, Cost{}, 2, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Supply, TYPE_TREASURE, CostLimitKind::LastChosenPlus, Cost{}, 3, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Hand, TYPE_VICTORY, CostLimitKind::None, Cost{}, 0, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Hand, 0, CostLimitKind::None, Cost{}, 0, DEF_COPPER, ANY_DEF},
    Filter{ZoneSelector::Hand, TYPE_ACTION, CostLimitKind::None, Cost{}, 0, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Discard, 0, CostLimitKind::None, Cost{}, 0, ANY_DEF, ANY_DEF},
    Filter{ZoneSelector::Supply, 0, CostLimitKind::Fixed, Cost{5, 0, 0}, 0, ANY_DEF, ANY_DEF},
};

static_assert(sizeof(kBaseCards) / sizeof(kBaseCards[0]) == BASIC_CARD_COUNT);
static_assert(sizeof(kFilters) / sizeof(kFilters[0]) == 10U);

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
