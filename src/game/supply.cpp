#include "supply.h"

void Supply::add_pile(const std::string& pile_name, std::vector<int> card_ids) {
    int index = static_cast<int>(piles_.size());
    int count = static_cast<int>(card_ids.size());
    piles_.push_back({pile_name, std::move(card_ids), count});
    pile_index_[pile_name] = index;
}

int Supply::gain_from(const std::string& pile_name) {
    SupplyPile* pile = get_pile(pile_name);
    if (!pile || pile->card_ids.empty()) return -1;
    int card_id = pile->card_ids.back();
    pile->card_ids.pop_back();
    return card_id;
}

int Supply::top_card(const std::string& pile_name) const {
    const SupplyPile* pile = get_pile(pile_name);
    if (!pile || pile->card_ids.empty()) return -1;
    return pile->card_ids.back();
}

int Supply::count(const std::string& pile_name) const {
    const SupplyPile* pile = get_pile(pile_name);
    if (!pile) return 0;
    return static_cast<int>(pile->card_ids.size());
}

bool Supply::is_pile_empty(const std::string& pile_name) const {
    return count(pile_name) == 0;
}

std::vector<std::string> Supply::available_piles() const {
    std::vector<std::string> result;
    for (const auto& pile : piles_) {
        if (!pile.card_ids.empty()) {
            result.push_back(pile.pile_name);
        }
    }
    return result;
}

std::vector<std::string> Supply::all_pile_names() const {
    std::vector<std::string> result;
    for (const auto& pile : piles_) {
        result.push_back(pile.pile_name);
    }
    return result;
}

int Supply::empty_pile_count() const {
    int count = 0;
    for (const auto& pile : piles_) {
        if (pile.card_ids.empty()) count++;
    }
    return count;
}

bool Supply::is_game_over() const {
    if (is_pile_empty("Province")) return true;
    if (empty_pile_count() >= 3) return true;
    return false;
}

SupplyPile* Supply::get_pile(const std::string& pile_name) {
    auto it = pile_index_.find(pile_name);
    if (it == pile_index_.end()) return nullptr;
    return &piles_[it->second];
}

const SupplyPile* Supply::get_pile(const std::string& pile_name) const {
    auto it = pile_index_.find(pile_name);
    if (it == pile_index_.end()) return nullptr;
    return &piles_[it->second];
}
