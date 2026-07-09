#include "diff_bridge.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

namespace {

using dz_diff::buy;
using dz_diff::play;
using dz_diff::Scenario;
using dz_diff::select;
using dz_diff::Snapshot;
using dz_diff::start_buy;

[[nodiscard]] std::vector<std::string> base_kingdom() {
    return {
        "Village",
        "Smithy",
        "Market",
        "Militia",
        "Moat",
        "Mine",
        "Bureaucrat",
        "Remodel",
        "Throne Room",
        "Witch",
    };
}

void require_equal_snapshots(const Scenario& scenario) {
    const Snapshot v1 = dz_diff::run_v1(scenario);
    const Snapshot v2 = dz_diff::run_v2(scenario);
    INFO(dz_diff::mismatch_dump(scenario, v1, v2));
    REQUIRE(v1.scores == v2.scores);
    REQUIRE(v1.owned == v2.owned);
    REQUIRE(v1.supply == v2.supply);
    REQUIRE(v1.trash == v2.trash);
    REQUIRE(v1.completed_turns == v2.completed_turns);
}

} // namespace

TEST_CASE("v2 differential BigMoney-only turn matches v1", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "BigMoney-only";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Copper", "Copper", "Silver"};
    scenario.players[0].deck = {"Estate", "Copper", "Copper", "Copper", "Copper"};
    scenario.players[1].hand = {"Copper"};
    scenario.steps = {
        start_buy(),
        play("Copper"),
        play("Copper"),
        play("Silver"),
        buy("Silver"),
    };

    require_equal_snapshots(scenario);
}

TEST_CASE("v2 differential Village Smithy Market chain matches v1", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "Village/Smithy/Market";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Village", "Smithy", "Market"};
    scenario.players[0].deck = {"Duchy", "Estate", "Gold", "Silver", "Copper"};
    scenario.steps = {
        play("Village"),
        play("Smithy"),
        play("Market"),
    };

    require_equal_snapshots(scenario);
}

TEST_CASE("v2 differential Militia Moat interaction matches v1", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "Militia/Moat";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Militia"};
    scenario.players[1].hand = {"Moat", "Copper", "Copper", "Estate", "Silver"};
    scenario.steps = {
        play("Militia", {select("Moat")}),
    };

    require_equal_snapshots(scenario);
}

TEST_CASE("v2 differential Mine matches fixed v1 oracle", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "Mine";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Mine", "Estate", "Copper", "Silver"};
    scenario.steps = {
        play("Mine", {select("Silver"), select("Gold")}),
    };

    require_equal_snapshots(scenario);
}

TEST_CASE("v2 differential Bureaucrat matches fixed v1 oracle", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "Bureaucrat";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Bureaucrat"};
    scenario.players[1].hand = {"Copper", "Estate", "Duchy"};
    scenario.steps = {
        play("Bureaucrat", {select("Duchy")}),
    };

    require_equal_snapshots(scenario);
}

TEST_CASE("v2 differential Remodel matches v1", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "Remodel";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Remodel", "Estate"};
    scenario.steps = {
        play("Remodel", {select("Estate"), select("Silver")}),
    };

    require_equal_snapshots(scenario);
}

TEST_CASE("v2 differential Throne Room on Smithy matches v1", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "Throne Room/Smithy";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Throne Room", "Smithy"};
    scenario.players[0].deck = {"Estate", "Duchy", "Province", "Copper", "Silver", "Gold"};
    scenario.steps = {
        play("Throne Room", {select("Smithy")}),
    };

    require_equal_snapshots(scenario);
}

TEST_CASE("v2 differential Witch curses match v1", "[v2][differential]") {
    Scenario scenario{};
    scenario.name = "Witch";
    scenario.kingdom = base_kingdom();
    scenario.players[0].hand = {"Witch"};
    scenario.players[0].deck = {"Silver", "Gold"};
    scenario.players[1].hand = {"Copper"};
    scenario.steps = {
        play("Witch"),
    };

    require_equal_snapshots(scenario);
}
