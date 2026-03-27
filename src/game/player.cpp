#include "player.h"

// Empty vector returned when querying a source/mat that doesn't exist
static const std::vector<int> EMPTY_VEC;

Player::Player(int id) : id_(id), rng_(std::random_device{}()) {}

int Player::get_id() const { return id_; }
int Player::hand_size() const { return static_cast<int>(hand_.size()); }
int Player::deck_size() const { return static_cast<int>(deck_.size()); }
int Player::discard_size() const { return static_cast<int>(discard_.size()); }

const std::vector<int>& Player::get_hand() const { return hand_; }
const std::vector<int>& Player::get_deck() const { return deck_; }
const std::vector<int>& Player::get_discard() const { return discard_; }
const std::vector<int>& Player::get_in_play() const { return in_play_; }

const std::unordered_map<std::string, std::vector<int>>& Player::get_set_aside() const {
    return set_aside_;
}

const std::unordered_map<std::string, std::vector<int>>& Player::get_mats() const {
    return mats_;
}

const std::vector<int>& Player::get_set_aside_by(const std::string& source) const {
    auto it = set_aside_.find(source);
    if (it == set_aside_.end()) return EMPTY_VEC;
    return it->second;
}

const std::vector<int>& Player::get_mat(const std::string& mat_name) const {
    auto it = mats_.find(mat_name);
    if (it == mats_.end()) return EMPTY_VEC;
    return it->second;
}

void Player::draw_cards(int count) {
    for (int i = 0; i < count; i++) {
        if (deck_.empty()) {
            if (discard_.empty()) return;  // nothing left to draw
            shuffle_discard_into_deck();
        }
        hand_.push_back(deck_.back());
        deck_.pop_back();
    }
}

void Player::discard_from_hand(int hand_index) {
    discard_.push_back(hand_[hand_index]);
    hand_.erase(hand_.begin() + hand_index);
}

void Player::discard_hand() {
    discard_.insert(discard_.end(), hand_.begin(), hand_.end());
    hand_.clear();
}

void Player::trash_from_hand(int hand_index) {
    // Removes from hand but does NOT add anywhere — caller
    // is responsible for adding to the game's trash pile
    hand_.erase(hand_.begin() + hand_index);
}

int Player::trash_from_hand_return(int hand_index) {
    int card_id = hand_[hand_index];
    hand_.erase(hand_.begin() + hand_index);
    return card_id;
}

void Player::add_to_hand(int card_id) {
    hand_.push_back(card_id);
}

void Player::add_to_discard(int card_id) {
    discard_.push_back(card_id);
}

void Player::add_to_deck_top(int card_id) {
    deck_.push_back(card_id);
}

void Player::play_from_hand(int hand_index) {
    in_play_.push_back(hand_[hand_index]);
    hand_.erase(hand_.begin() + hand_index);
}

void Player::add_to_in_play(int card_id) {
    in_play_.push_back(card_id);
}

void Player::topdeck_from_hand(int hand_index) {
    int card_id = hand_[hand_index];
    hand_.erase(hand_.begin() + hand_index);
    deck_.push_back(card_id);
}

void Player::cleanup() {
    // Move hand and in-play cards to discard, draw 5 new cards
    discard_hand();
    discard_.insert(discard_.end(), in_play_.begin(), in_play_.end());
    in_play_.clear();
    draw_cards(5);
}

std::vector<int> Player::peek_deck(int count) {
    if (static_cast<int>(deck_.size()) < count) {
        if (!discard_.empty()) {
            shuffle_discard_into_deck();
        }
    }
    int actual = std::min(count, static_cast<int>(deck_.size()));
    std::vector<int> result;
    result.reserve(actual);
    // deck_ draws from the back, so "top" = back
    for (int i = 0; i < actual; i++) {
        result.push_back(deck_[deck_.size() - 1 - i]);
    }
    return result;
}

int Player::remove_deck_top() {
    if (deck_.empty()) {
        if (discard_.empty()) return -1;
        shuffle_discard_into_deck();
    }
    if (deck_.empty()) return -1;
    int card_id = deck_.back();
    deck_.pop_back();
    return card_id;
}

int Player::find_in_hand(int card_id) const {
    for (int i = 0; i < static_cast<int>(hand_.size()); i++) {
        if (hand_[i] == card_id) return i;
    }
    return -1;
}

bool Player::remove_from_discard(int card_id) {
    auto it = std::find(discard_.begin(), discard_.end(), card_id);
    if (it == discard_.end()) return false;
    discard_.erase(it);
    return true;
}

int Player::remove_from_discard_by_index(int idx) {
    int card_id = discard_[idx];
    discard_.erase(discard_.begin() + idx);
    return card_id;
}

std::vector<int> Player::all_cards() const {
    std::vector<int> result;
    auto append = [&](const std::vector<int>& v) {
        result.insert(result.end(), v.begin(), v.end());
    };
    append(hand_);
    append(deck_);
    append(discard_);
    append(in_play_);
    for (const auto& [_, cards] : set_aside_) append(cards);
    for (const auto& [_, cards] : mats_) append(cards);
    return result;
}

// Set-aside
void Player::set_aside(int card_id, const std::string& source) {
    set_aside_[source].push_back(card_id);
}

int Player::remove_from_set_aside(const std::string& source, int index) {
    auto& cards = set_aside_[source];
    int card_id = cards[index];
    cards.erase(cards.begin() + index);
    return card_id;
}

// Mats
void Player::add_to_mat(int card_id, const std::string& mat_name) {
    mats_[mat_name].push_back(card_id);
}

int Player::remove_from_mat(const std::string& mat_name, int index) {
    auto& cards = mats_[mat_name];
    int card_id = cards[index];
    cards.erase(cards.begin() + index);
    return card_id;
}

void Player::shuffle_discard_into_deck() {
    deck_.insert(deck_.end(), discard_.begin(), discard_.end());
    discard_.clear();
    std::shuffle(deck_.begin(), deck_.end(), rng_);
}

void Player::set_rng(std::mt19937 rng) {
    rng_ = rng;
}
