#include "fuzz_harness.h"

#include "v2/core/types.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void print_usage(const char* argv0) noexcept {
    (void)std::fprintf(
        stderr,
        "usage: %s [--seeds N] [--steps-budget M] [--players P]\n"
        "  --players 0 samples 2-4 players per seed; otherwise use 2, 3, or 4.\n",
        argv0);
}

[[nodiscard]] bool parse_u64(const char* text, std::uint64_t& out) noexcept {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint64_t>(parsed);
    return true;
}

[[nodiscard]] bool next_arg(int argc, char** argv, int& index, const char*& value) noexcept {
    if (index + 1 >= argc) {
        print_usage(argv[0]);
        return false;
    }
    ++index;
    value = argv[index];
    return true;
}

} // namespace

int main(int argc, char** argv) {
    FuzzConfig config{};
    config.seeds = 1000;
    config.steps_budget = 100000;
    config.players = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }

        const char* value = nullptr;
        std::uint64_t parsed = 0;
        if (std::strcmp(argv[i], "--seeds") == 0) {
            if (!next_arg(argc, argv, i, value) || !parse_u64(value, parsed)) {
                print_usage(argv[0]);
                return 2;
            }
            config.seeds = parsed;
        } else if (std::strcmp(argv[i], "--steps-budget") == 0) {
            if (!next_arg(argc, argv, i, value) || !parse_u64(value, parsed)) {
                print_usage(argv[0]);
                return 2;
            }
            config.steps_budget = parsed;
        } else if (std::strcmp(argv[i], "--players") == 0) {
            if (!next_arg(argc, argv, i, value) || !parse_u64(value, parsed)) {
                print_usage(argv[0]);
                return 2;
            }
            if (parsed != 0U && (parsed < 2U || parsed > static_cast<std::uint64_t>(MAX_PLAYERS))) {
                print_usage(argv[0]);
                return 2;
            }
            config.players = static_cast<PlayerId>(parsed);
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    const FuzzResult result = run_fuzz(config);
    if (result.violations != 0U) {
        print_fuzz_failure(stderr, result);
        print_fuzz_summary(stdout, result);
        return 1;
    }
    print_fuzz_summary(stdout, result);
    return 0;
}
