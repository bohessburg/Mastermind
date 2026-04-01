#include "test_helpers.h"

#include <algorithm>

static bool g_cards_registered = false;

TestGame::TestGame(int num_players)
    : state_(num_players)
{
    ensure_cards_registered();
}

void TestGame::ensure_cards_registered() {
    if (!g_cards_registered) {
        BaseCards::register_all();
        Level1Cards::register_all();
        g_cards_registered = true;
    }
}

void TestGame::clear_hand(int pid) {
    Player& p = state_.get_player(pid);
    while (p.hand_size() > 0) {
        p.trash_from_hand(0);
    }
}

void TestGame::clear_deck(int pid) {
    Player& p = state_.get_player(pid);
    while (p.deck_size() > 0) {
        p.remove_deck_top();
    }
}

void TestGame::clear_discard(int pid) {
    Player& p = state_.get_player(pid);
    while (p.discard_size() > 0) {
        p.remove_from_discard_by_index(0);
    }
}

void TestGame::set_hand(int pid, const std::vector<std::string>& card_names) {
    clear_hand(pid);
    Player& p = state_.get_player(pid);
    for (const auto& name : card_names) {
        int id = state_.create_card(name);
        p.add_to_hand(id);
    }
}

void TestGame::set_deck(int pid, const std::vector<std::string>& card_names) {
    clear_deck(pid);
    Player& p = state_.get_player(pid);
    // Last element = top of deck = drawn first = pushed last (back of vector)
    for (const auto& name : card_names) {
        int id = state_.create_card(name);
        p.add_to_deck_top(id);
    }
}

void TestGame::set_discard(int pid, const std::vector<std::string>& card_names) {
    clear_discard(pid);
    Player& p = state_.get_player(pid);
    for (const auto& name : card_names) {
        int id = state_.create_card(name);
        p.add_to_discard(id);
    }
}

void TestGame::set_seed(uint64_t seed) {
    for (int i = 0; i < state_.num_players(); i++) {
        state_.get_player(i).set_rng(std::mt19937(seed + i));
    }
}

void TestGame::setup_supply(const std::vector<std::string>& kingdom_cards) {
    BaseCards::setup_supply(state_, kingdom_cards);
}

DecisionFn TestGame::scripted_decisions(std::vector<std::vector<int>> choices) {
    auto q = std::make_shared<std::queue<std::vector<int>>>();
    for (auto& c : choices) q->push(std::move(c));

    return [q](int /*player_id*/, ChoiceType /*type*/,
               const std::vector<int>& options,
               int min_choices, int /*max_choices*/) -> std::vector<int> {
        if (!q->empty()) {
            auto result = q->front();
            q->pop();
            return result;
        }
        std::vector<int> result;
        for (int i = 0; i < min_choices && i < static_cast<int>(options.size()); i++) {
            result.push_back(options[i]);
        }
        return result;
    };
}

DecisionFn TestGame::minimal_decisions() {
    return [](int /*player_id*/, ChoiceType /*type*/,
              const std::vector<int>& options,
              int min_choices, int /*max_choices*/) -> std::vector<int> {
        std::vector<int> result;
        for (int i = 0; i < min_choices && i < static_cast<int>(options.size()); i++) {
            result.push_back(options[i]);
        }
        return result;
    };
}

DecisionFn TestGame::greedy_decisions() {
    return [](int /*player_id*/, ChoiceType /*type*/,
              const std::vector<int>& options,
              int /*min_choices*/, int max_choices) -> std::vector<int> {
        std::vector<int> result;
        for (int i = 0; i < max_choices && i < static_cast<int>(options.size()); i++) {
            result.push_back(options[i]);
        }
        return result;
    };
}

void TestGame::play_card(int pid, const std::string& card_name, DecisionFn decide) {
    Player& player = state_.get_player(pid);
    const auto& hand = player.get_hand();
    for (int i = 0; i < static_cast<int>(hand.size()); i++) {
        if (state_.card_name(hand[i]) == card_name) {
            state_.play_card_from_hand(pid, i, decide);
            return;
        }
    }
}

GameState& TestGame::state() { return state_; }
const GameState& TestGame::state() const { return state_; }

std::vector<std::string> TestGame::hand_names(int pid) const {
    std::vector<std::string> result;
    for (int id : state_.get_player(pid).get_hand()) {
        result.push_back(state_.card_name(id));
    }
    return result;
}

std::vector<std::string> TestGame::deck_names(int pid) const {
    std::vector<std::string> result;
    for (int id : state_.get_player(pid).get_deck()) {
        result.push_back(state_.card_name(id));
    }
    return result;
}

std::vector<std::string> TestGame::discard_names(int pid) const {
    std::vector<std::string> result;
    for (int id : state_.get_player(pid).get_discard()) {
        result.push_back(state_.card_name(id));
    }
    return result;
}

std::vector<std::string> TestGame::in_play_names(int pid) const {
    std::vector<std::string> result;
    for (int id : state_.get_player(pid).get_in_play()) {
        result.push_back(state_.card_name(id));
    }
    return result;
}
