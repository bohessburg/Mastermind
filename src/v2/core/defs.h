#pragma once

#include "v2/core/state.h"
#include "v2/core/types.h"

#include <cstdint>

inline constexpr std::uint16_t TYPE_ACTION = static_cast<std::uint16_t>(1U << 0);
inline constexpr std::uint16_t TYPE_TREASURE = static_cast<std::uint16_t>(1U << 1);
inline constexpr std::uint16_t TYPE_VICTORY = static_cast<std::uint16_t>(1U << 2);
inline constexpr std::uint16_t TYPE_CURSE = static_cast<std::uint16_t>(1U << 3);
inline constexpr std::uint16_t TYPE_ATTACK = static_cast<std::uint16_t>(1U << 4);
inline constexpr std::uint16_t TYPE_REACTION = static_cast<std::uint16_t>(1U << 5);
inline constexpr std::uint16_t TYPE_DURATION = static_cast<std::uint16_t>(1U << 6);
inline constexpr std::uint16_t TYPE_NIGHT = static_cast<std::uint16_t>(1U << 7);
inline constexpr std::uint16_t TYPE_RESERVE = static_cast<std::uint16_t>(1U << 8);
inline constexpr std::uint16_t TYPE_COMMAND = static_cast<std::uint16_t>(1U << 9);

enum class Op : std::uint8_t {
    PlusCards,
    PlusActions,
    PlusBuys,
    PlusCoins,
    PlusCoffers,
    PlusVillagers,
    PlusFavors,
    PlusVP,
    TakeDebt,
    RepayDebtFree,
    DrawTo,
    GainSpecific,
    TrashSelf,
    DiscardSelf,
    TopdeckSelf,
    ExileSelf,
    ReturnToPile,
    Choose,
    ChooseGain,
    ChooseOption,
    ChooseOrder,
    Attack,
    EachOtherPlayer,
    Repeat,
    PerChosen,
    IfElse,
    EmitTrigger,
    CallCustom,
    End,
};

struct Instr {
    Op op = Op::End;
    std::uint8_t a = 0;
    std::uint8_t b = 0;
    std::uint8_t c = 0;
    std::int16_t arg = 0;
};

struct EffectSpan {
    std::uint16_t offset = 0;
    std::uint8_t len = 0;
};

enum class RunResult : std::uint8_t {
    NeedDecision,
    Continue,
    FrameDone,
};

using CustomStepFn = RunResult (*)(GameState&, EffectFrame&);
using ScoreHookFn = std::int16_t (*)(const GameState&, PlayerId, DefId);

struct CardDef {
    const char* name = "";
    Cost cost{};
    std::uint16_t types = 0;
    std::int8_t vp = 0;
    std::int8_t coin_value = 0;
    EffectSpan on_play{};
    EffectSpan on_gain{};
    EffectSpan on_trash{};
    EffectSpan on_discard{};
    EffectSpan on_reveal{};
    std::uint32_t trigger_mask = 0;
    CustomStepFn custom = nullptr;
    ScoreHookFn score_hook = nullptr;
};

enum class LandscapeKind : std::uint8_t {
    Event,
    Way,
    Landmark,
    Project,
    Trait,
    Ally,
    Prophecy,
};

struct LandscapeDef {
    const char* name = "";
    LandscapeKind kind = LandscapeKind::Event;
    Cost cost{};
    EffectSpan effect{};
    std::uint32_t trigger_mask = 0;
    CustomStepFn custom = nullptr;
    ScoreHookFn score_hook = nullptr;
};

inline constexpr EffectSpan NO_EFFECT{0, 0};

inline constexpr DefId DEF_COPPER = 0;
inline constexpr DefId DEF_SILVER = 1;
inline constexpr DefId DEF_GOLD = 2;
inline constexpr DefId DEF_PLATINUM = 3;
inline constexpr DefId DEF_POTION = 4;
inline constexpr DefId DEF_ESTATE = 5;
inline constexpr DefId DEF_DUCHY = 6;
inline constexpr DefId DEF_PROVINCE = 7;
inline constexpr DefId DEF_COLONY = 8;
inline constexpr DefId DEF_CURSE = 9;
inline constexpr std::uint16_t BASIC_CARD_COUNT = 10;

[[nodiscard]] const CardDef* card_defs() noexcept;
[[nodiscard]] std::uint16_t card_def_count() noexcept;
[[nodiscard]] const CardDef& card_def(DefId id) noexcept;

[[nodiscard]] const CardDef* base_card_defs() noexcept;
[[nodiscard]] std::uint16_t base_card_def_count() noexcept;

#define DOMINION_V2_DEF(Name, CostValue, TypeMask, VpValue, CoinValue) \
    CardDef {                                                          \
        #Name, CostValue, TypeMask, VpValue, CoinValue, NO_EFFECT,     \
            NO_EFFECT, NO_EFFECT, NO_EFFECT, NO_EFFECT, 0U, nullptr,   \
            nullptr                                                    \
    }
