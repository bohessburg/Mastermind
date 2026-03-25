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

    // Gain the top card from a pile, returns card_id (-1 if empty)
    int gain_from(const std::string& pile_name);

    // Look at the top card without removing it (-1 if empty)
    int top_card(const std::string& pile_name) const;

    // How many cards remain in a pile
    int count(const std::string& pile_name) const;

    // Is a specific pile empty?
    bool is_pile_empty(const std::string& pile_name) const;

    // All pile names that still have cards
    std::vector<std::string> available_piles() const;

    // All pile names (including empty ones)
    std::vector<std::string> all_pile_names() const;

    // How many piles are empty
    int empty_pile_count() const;

    // Standard Dominion end condition: Province pile empty OR 3+ piles empty
    bool is_game_over() const;

    // Direct pile access (for rotating split piles, inspecting contents, etc.)
    SupplyPile* get_pile(const std::string& pile_name);
    const SupplyPile* get_pile(const std::string& pile_name) const;

private:
    std::vector<SupplyPile> piles_;
    std::unordered_map<std::string, int> pile_index_;  // pile_name -> index into piles_
};
