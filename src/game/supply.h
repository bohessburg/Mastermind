#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct SupplyPile {
    std::string pile_name;       // e.g. "Knights", "Encampment/Plunder"
    std::vector<int> card_ids;   // stack of card IDs — gain from the back (top)
    int starting_count;
};

class Supply {
public:
    // Add a new pile to the supply
    void add_pile(const std::string& pile_name, std::vector<int> card_ids);

    // --- String-based API (convenience, used by card code) ---
    int gain_from(const std::string& pile_name);
    int top_card(const std::string& pile_name) const;
    int count(const std::string& pile_name) const;
    bool is_pile_empty(const std::string& pile_name) const;
    SupplyPile* get_pile(const std::string& pile_name);
    const SupplyPile* get_pile(const std::string& pile_name) const;

    // --- Index-based API (fast, used by engine hot path) ---
    int num_piles() const;
    const SupplyPile& pile_at(int index) const;
    int gain_from_index(int index);
    int top_card_index(int index) const;
    int count_index(int index) const;
    bool is_pile_empty_index(int index) const;
    int pile_index_of(const std::string& pile_name) const;  // -1 if not found

    // --- Allocation-free iteration ---
    const std::vector<SupplyPile>& piles() const;

    // --- Legacy API (allocates, avoid on hot path) ---
    std::vector<std::string> available_piles() const;
    std::vector<std::string> all_pile_names() const;

    // Aggregate queries
    int empty_pile_count() const;
    bool is_game_over() const;

private:
    std::vector<SupplyPile> piles_;
    std::unordered_map<std::string, int> pile_index_;
    int province_pile_idx_ = -1;  // cached for fast game-over check
};
