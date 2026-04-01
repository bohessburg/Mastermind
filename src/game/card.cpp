#include "card.h"

#include <unordered_map>
#include <vector>

namespace CardRegistry {

static std::vector<Card>& cards() {
    static std::vector<Card> instance;
    return instance;
}

static std::unordered_map<std::string, int>& name_index() {
    static std::unordered_map<std::string, int> instance;
    return instance;
}

void register_card(const Card& card) {
    auto& vec = cards();
    auto& idx = name_index();
    int id = static_cast<int>(vec.size());
    vec.push_back(card);
    vec.back().def_id = id;
    idx[card.name] = id;
}

const Card* get(const std::string& name) {
    auto& idx = name_index();
    auto it = idx.find(name);
    if (it == idx.end()) return nullptr;
    return &cards()[it->second];
}

const Card* get(int def_id) {
    auto& vec = cards();
    if (def_id < 0 || def_id >= static_cast<int>(vec.size())) return nullptr;
    return &vec[def_id];
}

int def_id_of(const std::string& name) {
    auto& idx = name_index();
    auto it = idx.find(name);
    if (it == idx.end()) return -1;
    return it->second;
}

std::vector<std::string> all_names() {
    std::vector<std::string> names;
    for (const auto& card : cards()) {
        names.push_back(card.name);
    }
    return names;
}

int count() {
    return static_cast<int>(cards().size());
}

} // namespace CardRegistry
