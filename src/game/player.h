#pragma once

#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

class Player {
public:
    Player(int id);

    int get_id() const;
    int hand_size() const;
    int deck_size() const;
    int discard_size() const;

    const std::vector<int>& get_hand() const;
    const std::vector<int>& get_deck() const;
    const std::vector<int>& get_discard() const;
    const std::vector<int>& get_in_play() const;

    // Set-aside and mat access
    const std::unordered_map<std::string, std::vector<int>>& get_set_aside() const;
    const std::unordered_map<std::string, std::vector<int>>& get_mats() const;
    const std::vector<int>& get_set_aside_by(const std::string& source) const;
    const std::vector<int>& get_mat(const std::string& mat_name) const;

    // Card movement
    void draw_cards(int count);
    void discard_from_hand(int hand_index);
    void discard_hand();
    void trash_from_hand(int hand_index);
    int trash_from_hand_return(int hand_index);
    void add_to_hand(int card_id);
    void add_to_discard(int card_id);
    void add_to_deck_top(int card_id);
    void play_from_hand(int hand_index);
    void add_to_in_play(int card_id);
    void topdeck_from_hand(int hand_index);
    void cleanup();

    // Deck inspection
    std::vector<int> peek_deck(int count);
    int remove_deck_top();

    // Hand search
    int find_in_hand(int card_id) const;

    // Discard manipulation
    bool remove_from_discard(int card_id);
    int remove_from_discard_by_index(int idx);

    // All cards in all zones
    std::vector<int> all_cards() const;

    // Zero-allocation iteration over all cards in all zones
    template<typename Fn>
    void for_each_card(Fn fn) const {
        for (int cid : hand_) fn(cid);
        for (int cid : deck_) fn(cid);
        for (int cid : discard_) fn(cid);
        for (int cid : in_play_) fn(cid);
        for (const auto& [_, cards] : set_aside_) for (int cid : cards) fn(cid);
        for (const auto& [_, cards] : mats_) for (int cid : cards) fn(cid);
    }

    int total_card_count() const;

    // Set-aside and mat manipulation
    void set_aside(int card_id, const std::string& source);
    int remove_from_set_aside(const std::string& source, int index);
    void add_to_mat(int card_id, const std::string& mat_name);
    int remove_from_mat(const std::string& mat_name, int index);

    void shuffle_discard_into_deck();
    void set_rng(std::mt19937 rng);

private:
    int id_;
    std::vector<int> hand_;
    std::vector<int> deck_;        // draw from the back
    std::vector<int> discard_;
    std::vector<int> in_play_;
    std::unordered_map<std::string, std::vector<int>> set_aside_;  // source card name -> cards
    std::unordered_map<std::string, std::vector<int>> mats_;       // mat name -> cards
    std::mt19937 rng_;
};
