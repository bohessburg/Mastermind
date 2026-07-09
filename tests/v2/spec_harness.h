#pragma once

#include "v2/core/actions.h"
#include "v2/core/game.h"
#include "v2/core/setup.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#define CARD_SPEC(Name) TEST_CASE_METHOD(SpecHarness, Name, "[v2][spec]")

class SpecHarness {
public:
    class GivenBuilder {
    public:
        explicit GivenBuilder(SpecHarness& harness);

        template <typename... Names>
        GivenBuilder& hand(Names... names) {
            (harness_.add_hand(names), ...);
            harness_.refresh_baseline();
            return *this;
        }

        template <typename... Names>
        GivenBuilder& deck(Names... names) {
            (harness_.add_deck(names), ...);
            harness_.refresh_baseline();
            return *this;
        }

    private:
        SpecHarness& harness_;
    };

    class ExpectBuilder {
    public:
        explicit ExpectBuilder(SpecHarness& harness);

        ExpectBuilder& hand_has(const char* name);
        ExpectBuilder& hand_lacks(const char* name);
        ExpectBuilder& discard_has(const char* name);
        ExpectBuilder& discard_lacks(const char* name);
        ExpectBuilder& trash_has(const char* name);
        ExpectBuilder& trash_count(const char* name, std::uint8_t count);
        ExpectBuilder& coins(std::int16_t coins);

    private:
        SpecHarness& harness_;
    };

    SpecHarness();

    GivenBuilder& given();
    ExpectBuilder& expect();

    void play(const char* name);
    void choose(const char* name);
    void gain(const char* name);
    void pass();
    void option(std::uint8_t option);
    void empty_supply_up_to(std::int8_t coins);
    void expect_conservation() const;

    [[nodiscard]] GameState& state();
    [[nodiscard]] const GameState& state() const;
    [[nodiscard]] DefId def_named(const char* name) const;
    [[nodiscard]] int total_cards() const;

private:
    void add_hand(const char* name);
    void add_deck(const char* name);
    void clear_player_zero();
    void refresh_baseline();
    void step_checked(Action action);
    [[nodiscard]] Slot slot_named(const char* name) const;
    [[nodiscard]] std::uint8_t hand_count(const char* name) const;
    [[nodiscard]] std::uint8_t discard_count(const char* name) const;
    [[nodiscard]] std::uint8_t trash_count_for(const char* name) const;

    GameState state_;
    int baseline_ = 0;
    GivenBuilder given_;
    ExpectBuilder expect_;
};
