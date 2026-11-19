#pragma once

#include "v2/core/game.h"
#include "v2/core/setup.h"

#include <cstdint>
#include <cstdio>

struct InvariantBaseline {
    int total_cards = 0;
    std::uint8_t num_piles = 0;
    std::uint8_t pile_counts[MAX_PILES]{};
    std::uint8_t num_nonsupply = 0;
    std::uint8_t nonsupply_counts[MAX_NONSUPPLY]{};
};

struct InvariantViolation {
    bool ok = true;
    char message[192]{};
};

struct FuzzConfig {
    std::uint64_t seed_start = 0;
    std::uint64_t seeds = 50;
    std::uint64_t steps_budget = 250000;
    PlayerId players = 0; // 0 means sample 2-4 players per seed.
};

struct FuzzResult {
    std::uint64_t games = 0;
    std::uint64_t steps = 0;
    std::uint64_t violations = 0;
    std::uint64_t failing_seed = 0;
    std::uint64_t failing_step = 0;
    Setup failing_setup{};
    GameState failing_state{};
    InvariantViolation violation{};
};

[[nodiscard]] InvariantBaseline capture_baseline(const GameState& state) noexcept;
[[nodiscard]] InvariantViolation check_invariants(
    const GameState& state,
    const InvariantBaseline& baseline) noexcept;
[[nodiscard]] Setup random_kingdom_setup(std::uint64_t seed, PlayerId players) noexcept;
[[nodiscard]] FuzzResult run_fuzz(const FuzzConfig& config) noexcept;

void print_fuzz_summary(std::FILE* file, const FuzzResult& result) noexcept;
void print_fuzz_failure(std::FILE* file, const FuzzResult& result) noexcept;
