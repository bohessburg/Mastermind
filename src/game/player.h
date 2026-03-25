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
    void add_to_hand(int card_id);
    void add_to_discard(int card_id);
    void add_to_deck_top(int card_id);
    void play_from_hand(int hand_index);
    void cleanup();

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
