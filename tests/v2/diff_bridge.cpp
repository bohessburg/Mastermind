#include "diff_bridge.h"

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

} // namespace dz_diff
