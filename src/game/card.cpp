#include "card.h"

#include <unordered_map>

namespace CardRegistry {

static std::unordered_map<std::string, Card>& registry() {
    static std::unordered_map<std::string, Card> instance;
    return instance;
}

void register_card(const Card& card) {
    registry()[card.name] = card;
}

const Card* get(const std::string& name) {
    auto& reg = registry();
    auto it = reg.find(name);
    if (it == reg.end()) return nullptr;
    return &it->second;
}

std::vector<std::string> all_names() {
    std::vector<std::string> names;
    for (const auto& [name, card] : registry()) {
        names.push_back(name);
    }
    return names;
}

} // namespace CardRegistry
