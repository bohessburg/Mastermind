#include "level_2_cards.h"
#include "../card.h"
#include "../game_state.h"

#include <algorithm>
#include <set>

namespace Level2Cards {

void register_all() {
    CardRegistry::register_card({
        .name = "Worker's Village",
        .cost = 4,
        .types = CardType::Action,
        .text = "+1 Card, +2 Actions, +1 Buy",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(2);
            state.add_buys(1);
        },
    });

    CardRegistry::register_card({
        .name = "Courtyard",
        .cost = 2,
        .types = CardType::Action,
        .text = "+3 Cards. Put a card from your hand onto your deck.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(3);
            if (player.hand_size() > 0) {
                std::vector<int> topdeck_opts;
                for (int i = 0; i < player.hand_size(); i++) topdeck_opts.push_back(i);
                auto chosen = decide(pid, ChoiceType::TOPDECK, topdeck_opts, 1, 1);
                if (!chosen.empty()) {
                    player.topdeck_from_hand(chosen[0]);
                }
            }
        },
    });

    CardRegistry::register_card({
        .name = "Hamlet",
        .cost = 2,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. You may discard a card for +1 Action. You may discard a card for +1 Buy",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            // You may discard a card for +1 Action
            if (player.hand_size() > 0) {
                std::vector<int> discard_opts;
                for (int i = 0; i < player.hand_size(); i++) discard_opts.push_back(i);
                auto chosen = decide(pid, ChoiceType::DISCARD, discard_opts, 0, 1);
                if (!chosen.empty()) {
                    player.discard_from_hand(chosen[0]);
                    state.add_actions(1);
                }
            }
            // You may discard a card for +1 Buy
            if (player.hand_size() > 0) {
                std::vector<int> discard_opts2;
                for (int i = 0; i < player.hand_size(); i++) discard_opts2.push_back(i);
                auto chosen2 = decide(pid, ChoiceType::DISCARD, discard_opts2, 0, 1);
                if (!chosen2.empty()) {
                    player.discard_from_hand(chosen2[0]);
                    state.add_buys(1);
                }
            }
        },
    });

    CardRegistry::register_card({
        .name = "Lookout",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Action. Look at the top 3 cards of your deck. Trash one of them. Discard one of them. Put the other one on top of your deck.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_actions(1);
            auto top_cards = player.peek_deck(3);
            int num_peeked = static_cast<int>(top_cards.size());
            // Remove them from deck
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            if (num_peeked == 0) return;

            // Trash one (decide returns an index into top_cards)
            auto trash_chosen = decide(pid, ChoiceType::TRASH, top_cards, 1, 1);
            if (!trash_chosen.empty()) {
                int trashed_id = top_cards[trash_chosen[0]];
                state.trash_card(trashed_id);
                top_cards.erase(std::find(top_cards.begin(), top_cards.end(), trashed_id));
            }

            // Discard one (if 2+ remain)
            if (top_cards.size() >= 2) {
                auto disc_chosen = decide(pid, ChoiceType::DISCARD, top_cards, 1, 1);
                if (!disc_chosen.empty()) {
                    int discarded_id = top_cards[disc_chosen[0]];
                    player.add_to_discard(discarded_id);
                    top_cards.erase(std::find(top_cards.begin(), top_cards.end(), discarded_id));
                }
            }

            // Put the remaining card back on top
            if (!top_cards.empty()) {
                player.add_to_deck_top(top_cards[0]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Swindler",
        .cost = 3,
        .types = CardType::Action | CardType::Attack,
        .text = "+2 Coins. Each other player trashes the top card of their deck and gains a card with the same cost that you choose",   
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_coins(2);
            state.resolve_attack(pid, [pid](GameState& st, int target, DecisionFn dec) {
                Player& p = st.get_player(target);
                int card_id = p.remove_deck_top();
                if (card_id == -1) return;

                const Card* trashed = st.card_def(card_id);
                st.trash_card(card_id);
                if (!trashed) return;

                // Attacker chooses a card with the same cost for the target to gain
                auto all_piles = st.gainable_piles(trashed->cost);
                std::vector<std::string> piles;
                for (const auto& pile : all_piles) {
                    const Card* c = CardRegistry::get(pile);
                    if (c && c->cost == trashed->cost) piles.push_back(pile);
                }
                if (piles.empty()) return;

                std::vector<int> gain_opts;
                for (const auto& p : piles) gain_opts.push_back(st.get_supply().top_card(p));
                auto chosen = dec(pid, ChoiceType::GAIN, gain_opts, 1, 1);
                if (!chosen.empty()) {
                    st.gain_card(target, piles[chosen[0]]);
                }
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Scholar",
        .cost = 5,
        .types = CardType::Action,
        .text = "Discard your hand. +7 Cards.",   
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            player.discard_hand();
            player.draw_cards(7);
        },
    });

    CardRegistry::register_card({
        .name = "Storeroom",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Buy. Discard any number of cards, then draw that many. Discard any number of cards for +1 Coin each.",   
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_buys(1);
            if (player.hand_size() == 0) return;

            //First discard then redraw like cellar
            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

            auto chosen = decide(pid, ChoiceType::DISCARD, options, 0, player.hand_size());
            std::sort(chosen.begin(), chosen.end(), std::greater<int>());
            int count = static_cast<int>(chosen.size());
            for (int idx : chosen) player.discard_from_hand(idx);
            player.draw_cards(count);

            if (player.hand_size() == 0) return;
            //next discard again for Coins
            std::vector<int> options2;
            for (int i = 0; i < player.hand_size(); i++) options2.push_back(i);

            auto chosen2 = decide(pid, ChoiceType::DISCARD, options2, 0, player.hand_size());
            std::sort(chosen2.begin(), chosen2.end(), std::greater<int>());
            int count2 = static_cast<int>(chosen2.size());
            for (int idx : chosen2) player.discard_from_hand(idx);
            state.add_coins(count2);

        },
    });

    CardRegistry::register_card({
        .name = "Conspirator",
        .cost = 4,
        .types = CardType::Action,
        .text = "+2 Coins. If you've played 3 or more Actions this turn (counting this), +1 Card and +1 Action.",   
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.add_coins(2);
            if (state.actions_played() >= 3) {
                state.get_player(pid).draw_cards(1);
                state.add_actions(1);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Menagerie",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Action. Reveal your hand. If the revealed cards all have different names, +3 Cards. Otherwise, +1 Card.",   
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.add_actions(1);
            Player& player = state.get_player(pid);
            std::set<std::string> seen;
            bool all_different = true;
            for (int card_id : player.get_hand()) {
                if (!seen.insert(state.card_name(card_id)).second) {
                    all_different = false;
                    break;
                }
            }
            player.draw_cards(all_different ? 3 : 1);
        },
    });

    CardRegistry::register_card({
        .name = "Oasis",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action, +1 Coin. Discard a card.",   
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            state.add_coins(1);

            if (player.hand_size() == 0) return;
            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
            auto chosen = decide(pid, ChoiceType::DISCARD, options, 1, 1);
            if (!chosen.empty()) {
                player.discard_from_hand(chosen[0]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "King's Court",
        .cost = 7,
        .types = CardType::Action,
        .text = "You may play an Action card from your hand three times.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);

            std::vector<int> action_indices;
            for (int i = 0; i < player.hand_size(); i++) {
                const Card* card = state.card_def(player.get_hand()[i]);
                if (card && card->is_action()) action_indices.push_back(i);
            }

            if (action_indices.empty()) return;

            auto chosen = decide(pid, ChoiceType::PLAY_CARD, action_indices, 0, 1);
            if (chosen.empty()) return;

            // chosen[0] is an index into action_indices, not a hand index
            int hand_idx = action_indices[chosen[0]];
            int card_id = player.get_hand()[hand_idx];
            player.play_from_hand(hand_idx);

            state.play_card_effect(card_id, pid, decide);
            state.play_card_effect(card_id, pid, decide);
            state.play_card_effect(card_id, pid, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Courier",
        .cost = 4,
        .types = CardType::Action,
        .text = "+1 Coin. Discard the top card of your deck. Look through your discard pile; you may play an Action or Treasure from it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_coins(1);
            int card_id = player.remove_deck_top();
            if (card_id != -1) {
                player.add_to_discard(card_id);
            }
            std::vector<int> play_choices = player.get_discard();
            auto chosen = decide(pid, ChoiceType::PLAY_CARD, play_choices, 0, 1);
            if (!chosen.empty()) {
                int play_id = chosen[0];
                player.add_to_in_play(play_id);
                state.play_card_effect(play_id, pid, decide);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Sentinel",
        .cost = 3,
        .types = CardType::Action,
        .text = "Look at the top 5 cards of your deck. You may trash up to 2 of them. Put the rest back in any order.",   
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            auto top_cards = player.peek_deck(5);
            int num_peeked = static_cast<int>(top_cards.size());
            // Remove them from deck
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            if (num_peeked == 0) return;

            // Trash up to 2
            auto trash_chosen = decide(pid, ChoiceType::TRASH, top_cards, 0, 2);
            for (int card_id : trash_chosen) {
                state.trash_card(card_id);
                top_cards.erase(std::find(top_cards.begin(), top_cards.end(), card_id));
            }

            // Put the rest back in any order
            std::vector<int> ordered;
            while (top_cards.size() > 1) {
                auto chosen = decide(pid, ChoiceType::ORDER, top_cards, 1, 1);
                int pick = chosen.empty() ? top_cards[0] : chosen[0];
                ordered.push_back(pick);
                top_cards.erase(std::find(top_cards.begin(), top_cards.end(), pick));
            }
            if (!top_cards.empty()) ordered.push_back(top_cards[0]);

            // First picked = top of deck, so add in reverse
            for (int i = static_cast<int>(ordered.size()) - 1; i >= 0; i--) {
                player.add_to_deck_top(ordered[i]);
            }
        },
    });

}


} // namespace Level2Cards