#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

namespace dz_diff {

enum class ChoiceKind {
    Select,
    Option,
    Pass,
};

struct Choice {
    ChoiceKind kind = ChoiceKind::Pass;
    std::string name;
    int option = 0;
};

enum class StepKind {
    StartBuy,
    Play,
    Buy,
    EndTurn,
};

struct Step {
    StepKind kind = StepKind::Play;
    std::string name;
    std::vector<Choice> choices;
};

struct PlayerSetup {
    std::vector<std::string> hand;
    std::vector<std::string> deck;
    std::vector<std::string> discard;
};

struct Scenario {
    std::string name;
    std::vector<std::string> kingdom;
    std::array<PlayerSetup, 2> players;
    std::vector<Step> steps;
};

struct Snapshot {
    std::vector<std::map<std::string, int>> owned;
    std::map<std::string, int> supply;
    std::map<std::string, int> trash;
    std::vector<int> scores;
    int completed_turns = 0;
    std::string dump;
};

[[nodiscard]] Choice select(std::string name);
[[nodiscard]] Choice option(int value);
[[nodiscard]] Choice pass();
[[nodiscard]] Step start_buy();
[[nodiscard]] Step play(std::string name, std::vector<Choice> choices = {});
[[nodiscard]] Step buy(std::string name);
[[nodiscard]] Step end_turn();

[[nodiscard]] Snapshot run_v1(const Scenario& scenario);
[[nodiscard]] Snapshot run_v2(const Scenario& scenario);
[[nodiscard]] std::string mismatch_dump(
    const Scenario& scenario,
    const Snapshot& v1,
    const Snapshot& v2);

} // namespace dz_diff
