#include "supply.h"

void Supply::add_pile(const std::string& pile_name, std::vector<int> card_ids) {
    int index = static_cast<int>(piles_.size());
    int count = static_cast<int>(card_ids.size());
    piles_.push_back({pile_name, std::move(card_ids), count});
    pile_index_[pile_name] = index;
    if (pile_name == "Province") province_pile_idx_ = index;
}

// --- String-based API ---

int Supply::gain_from(const std::string& pile_name) {
    auto it = pile_index_.find(pile_name);
    if (it == pile_index_.end()) return -1;
    return gain_from_index(it->second);
}

int Supply::top_card(const std::string& pile_name) const {
    auto it = pile_index_.find(pile_name);
    if (it == pile_index_.end()) return -1;
    return top_card_index(it->second);
}

int Supply::count(const std::string& pile_name) const {
    auto it = pile_index_.find(pile_name);
    if (it == pile_index_.end()) return 0;
    return count_index(it->second);
}

bool Supply::is_pile_empty(const std::string& pile_name) const {
    return count(pile_name) == 0;
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

// --- Index-based API ---

int Supply::num_piles() const {
    return static_cast<int>(piles_.size());
}

const SupplyPile& Supply::pile_at(int index) const {
    return piles_[index];
}

int Supply::gain_from_index(int index) {
    auto& pile = piles_[index];
    if (pile.card_ids.empty()) return -1;
    int card_id = pile.card_ids.back();
    pile.card_ids.pop_back();
    return card_id;
}

int Supply::top_card_index(int index) const {
    const auto& pile = piles_[index];
    if (pile.card_ids.empty()) return -1;
    return pile.card_ids.back();
}

int Supply::count_index(int index) const {
    return static_cast<int>(piles_[index].card_ids.size());
}

bool Supply::is_pile_empty_index(int index) const {
    return piles_[index].card_ids.empty();
}

int Supply::pile_index_of(const std::string& pile_name) const {
    auto it = pile_index_.find(pile_name);
    if (it == pile_index_.end()) return -1;
    return it->second;
}

const std::vector<SupplyPile>& Supply::piles() const {
    return piles_;
}

// --- Legacy API ---

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

// --- Aggregate queries ---

int Supply::empty_pile_count() const {
    int count = 0;
    for (const auto& pile : piles_) {
        if (pile.card_ids.empty()) count++;
    }
    return count;
}

bool Supply::is_game_over() const {
    if (province_pile_idx_ >= 0 && piles_[province_pile_idx_].card_ids.empty()) return true;
    if (empty_pile_count() >= 3) return true;
    return false;
}
