#pragma once

#include "../game/game_state.h"
#include "../game/card.h"

#include <iostream>
#include <string>
#include <vector>

namespace term {

inline void print_divider() {
    std::cout << "────────────────────────────────────────────────────\n";
}

inline void print_supply(const GameState& state) {
    std::cout << "  SUPPLY\n";
    std::cout << "  Treasures: ";
    for (auto& n : {"Copper", "Silver", "Gold"})
        std::cout << n << "(" << state.get_supply().count(n) << ") ";
    std::cout << "\n  Victory:   ";
    for (auto& n : {"Estate", "Duchy", "Province"})
        std::cout << n << "(" << state.get_supply().count(n) << ") ";
    std::cout << "\n  Curse:     Curse(" << state.get_supply().count("Curse") << ")\n";

    std::cout << "  Kingdom:   ";
    int col = 0;
    for (const auto& pile_name : state.get_supply().all_pile_names()) {
        if (pile_name == "Copper" || pile_name == "Silver" || pile_name == "Gold" ||
            pile_name == "Estate" || pile_name == "Duchy" || pile_name == "Province" ||
            pile_name == "Curse") continue;

        const Card* card = CardRegistry::get(pile_name);
        int cost = card ? card->cost : 0;
        int count = state.get_supply().count(pile_name);
        std::cout << pile_name << "(" << cost << "$,x" << count << ") ";
        col++;
        if (col % 5 == 0) std::cout << "\n             ";
    }
    std::cout << "\n";
}

inline void print_hand(const GameState& state, int pid, const std::string& label) {
    const Player& player = state.get_player(pid);
    const auto& hand = player.get_hand();
    std::cout << "  " << label << " hand [" << hand.size() << "]: ";
    for (int i = 0; i < static_cast<int>(hand.size()); i++) {
        const Card* card = state.card_def(hand[i]);
        std::string name = card ? card->name : "???";
        std::string types;
        if (card) {
            if (card->is_action()) types += "A";
            if (card->is_treasure()) types += "T";
            if (card->is_victory()) types += "V";
            if (card->is_curse()) types += "C";
            if (card->is_attack()) types += "!";
            if (card->is_reaction()) types += "R";
        }
        std::cout << name;
        if (!types.empty()) std::cout << "[" << types << "]";
        if (i < static_cast<int>(hand.size()) - 1) std::cout << ", ";
    }
    std::cout << "\n";
}

inline void print_in_play(const GameState& state, int pid, const std::string& label) {
    const auto& in_play = state.get_player(pid).get_in_play();
    if (!in_play.empty()) {
        std::cout << "  " << label << " in play: ";
        for (int i = 0; i < static_cast<int>(in_play.size()); i++) {
            std::cout << state.card_name(in_play[i]);
            if (i < static_cast<int>(in_play.size()) - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

inline void print_trash(const GameState& state) {
    const auto& trash = state.get_trash();
    if (!trash.empty()) {
        std::cout << "  Trash [" << trash.size() << "]: ";
        for (int i = 0; i < static_cast<int>(trash.size()); i++) {
            std::cout << state.card_name(trash[i]);
            if (i < static_cast<int>(trash.size()) - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

inline std::string discard_top_name(const GameState& state, int pid) {
    const auto& d = state.get_player(pid).get_discard();
    if (d.empty()) return "empty";
    return state.card_name(d.back());
}

inline void print_status(const GameState& state, int pid) {
    std::cout << "  Actions: " << state.actions()
              << "  Buys: " << state.buys()
              << "  Coins: " << state.coins()
              << "  |  Deck: " << state.get_player(pid).deck_size()
              << "  Discard: " << state.get_player(pid).discard_size() << "\n";
}

inline int prompt_choice(const std::string& prompt, int min_val, int max_val) {
    while (true) {
        std::cout << prompt;
        if (min_val == max_val) {
            std::cout << " (auto " << min_val << "): ";
            return min_val;
        }
        std::cout << " [" << min_val << "-" << max_val << "]: ";
        std::string line;
        if (!std::getline(std::cin, line)) return min_val;
        if (line.empty()) continue;
        try {
            int val = std::stoi(line);
            if (val >= min_val && val <= max_val) return val;
        } catch (...) {}
        std::cout << "  Invalid input.\n";
    }
}

} // namespace term
