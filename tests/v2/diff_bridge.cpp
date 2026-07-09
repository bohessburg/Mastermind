#include "diff_bridge.h"

#include <ostream>
#include <sstream>
#include <utility>

namespace dz_diff {

Choice select(std::string name) {
    Choice choice{};
    choice.kind = ChoiceKind::Select;
    choice.name = std::move(name);
    return choice;
}

Choice option(int value) {
    Choice choice{};
    choice.kind = ChoiceKind::Option;
    choice.option = value;
    return choice;
}

Choice pass() {
    return Choice{};
}

Step start_buy() {
    Step step{};
    step.kind = StepKind::StartBuy;
    return step;
}

Step play(std::string name, std::vector<Choice> choices) {
    Step step{};
    step.kind = StepKind::Play;
    step.name = std::move(name);
    step.choices = std::move(choices);
    return step;
}

Step buy(std::string name) {
    Step step{};
    step.kind = StepKind::Buy;
    step.name = std::move(name);
    return step;
}

Step end_turn() {
    Step step{};
    step.kind = StepKind::EndTurn;
    return step;
}

namespace {

void write_map(std::ostream& out, const std::map<std::string, int>& values) {
    out << "{";
    bool first = true;
    for (const auto& [name, count] : values) {
        if (!first) {
            out << ", ";
        }
        first = false;
        out << name << ":" << count;
    }
    out << "}";
}

void write_snapshot(std::ostream& out, const Snapshot& snapshot) {
    out << "scores=[";
    for (std::size_t i = 0; i < snapshot.scores.size(); ++i) {
        if (i != 0U) {
            out << ", ";
        }
        out << snapshot.scores[i];
    }
    out << "] turns=" << snapshot.completed_turns << "\n";
    out << "supply=";
    write_map(out, snapshot.supply);
    out << "\ntrash=";
    write_map(out, snapshot.trash);
    out << "\n";
    for (std::size_t i = 0; i < snapshot.owned.size(); ++i) {
        out << "p" << i << "=";
        write_map(out, snapshot.owned[i]);
        out << "\n";
    }
    out << snapshot.dump;
}

} // namespace

std::string mismatch_dump(
    const Scenario& scenario,
    const Snapshot& v1,
    const Snapshot& v2) {
    std::ostringstream out;
    out << "Differential mismatch: " << scenario.name << "\n";
    out << "-- v1 --\n";
    write_snapshot(out, v1);
    out << "-- v2 --\n";
    write_snapshot(out, v2);
    return out.str();
}

} // namespace dz_diff
