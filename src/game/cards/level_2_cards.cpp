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
            // Filter discard to only Action or Treasure cards
            std::vector<int> play_choices;
            for (int cid : player.get_discard()) {
                const Card* c = state.card_def(cid);
                if (c && (c->is_action() || c->is_treasure())) {
                    play_choices.push_back(cid);
                }
            }
            if (play_choices.empty()) return;
            auto chosen = decide(pid, ChoiceType::PLAY_CARD, play_choices, 0, 1);
            if (!chosen.empty()) {
                int play_id = play_choices[chosen[0]];
                player.remove_from_discard(play_id);
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

            // First picked = top of deck, so add in reverse.
            for (int i = static_cast<int>(ordered.size()) - 1; i >= 0; i--) {
                player.add_to_deck_top(ordered[i]);
            }
        },
    });

    // =============================================
    // Batch A — Trivial cards
    // =============================================

    CardRegistry::register_card({
        .name = "Bazaar",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +2 Actions, +1 Coin.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(2);
            state.add_coins(1);
        },
    });

    CardRegistry::register_card({
        .name = "Junk Dealer",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action, +1 Coin. Trash a card from your hand.",
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
            auto chosen = decide(pid, ChoiceType::TRASH, options, 1, 1);
            if (!chosen.empty()) {
                int card_id = player.trash_from_hand_return(chosen[0]);
                state.trash_card(card_id);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Warehouse",
        .cost = 3,
        .types = CardType::Action,
        .text = "+3 Cards, +1 Action. Discard 3 cards.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(3);
            state.add_actions(1);
            int to_discard = std::min(3, player.hand_size());
            if (to_discard == 0) return;
            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
            auto chosen = decide(pid, ChoiceType::DISCARD, options, to_discard, to_discard);
            std::sort(chosen.begin(), chosen.end(), std::greater<int>());
            for (int idx : chosen) player.discard_from_hand(idx);
        },
    });

    CardRegistry::register_card({
        .name = "Pilgrim",
        .cost = 5,
        .types = CardType::Action,
        .text = "+4 Cards. Put a card from your hand onto your deck.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(4);
            if (player.hand_size() > 0) {
                std::vector<int> options;
                for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
                auto chosen = decide(pid, ChoiceType::TOPDECK, options, 1, 1);
                if (!chosen.empty()) {
                    player.topdeck_from_hand(chosen[0]);
                }
            }
        },
    });

    // =============================================
    // Batch B — Remodel variants
    // =============================================

    CardRegistry::register_card({
        .name = "Altar",
        .cost = 6,
        .types = CardType::Action,
        .text = "Trash a card from your hand. Gain a card costing up to 5.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            if (player.hand_size() > 0) {
                std::vector<int> trash_opts;
                for (int i = 0; i < player.hand_size(); i++) trash_opts.push_back(i);
                auto chosen = decide(pid, ChoiceType::TRASH, trash_opts, 1, 1);
                if (!chosen.empty()) {
                    int card_id = player.trash_from_hand_return(chosen[0]);
                    state.trash_card(card_id);
                }
            }
            auto piles = state.gainable_piles(5);
            if (piles.empty()) return;
            std::vector<int> gain_opts;
            for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
            auto chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
            if (!chosen.empty()) state.gain_card(pid, piles[chosen[0]]);
        },
    });

    CardRegistry::register_card({
        .name = "Armory",
        .cost = 4,
        .types = CardType::Action,
        .text = "Gain a card onto your deck costing up to 4.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            auto piles = state.gainable_piles(4);
            if (piles.empty()) return;
            std::vector<int> options;
            for (const auto& p : piles) options.push_back(state.get_supply().top_card(p));
            auto chosen = decide(pid, ChoiceType::GAIN, options, 1, 1);
            if (!chosen.empty()) state.gain_card_to_deck_top(pid, piles[chosen[0]]);
        },
    });

    CardRegistry::register_card({
        .name = "Expand",
        .cost = 7,
        .types = CardType::Action,
        .text = "Trash a card from your hand. Gain a card costing up to 3 more than it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            if (player.hand_size() == 0) return;
            std::vector<int> trash_opts;
            for (int i = 0; i < player.hand_size(); i++) trash_opts.push_back(i);
            auto chosen = decide(pid, ChoiceType::TRASH, trash_opts, 1, 1);
            if (chosen.empty()) return;
            int card_id = player.trash_from_hand_return(chosen[0]);
            const Card* trashed = state.card_def(card_id);
            state.trash_card(card_id);
            if (!trashed) return;
            int max_cost = trashed->cost + 3;
            auto piles = state.gainable_piles(max_cost);
            if (piles.empty()) return;
            std::vector<int> gain_opts;
            for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
            auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
            if (!gain_chosen.empty()) state.gain_card(pid, piles[gain_chosen[0]]);
        },
    });

    CardRegistry::register_card({
        .name = "Salvager",
        .cost = 4,
        .types = CardType::Action,
        .text = "+1 Buy. Trash a card from your hand. +1 Coin per 1 it costs.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_buys(1);
            if (player.hand_size() == 0) return;
            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
            auto chosen = decide(pid, ChoiceType::TRASH, options, 1, 1);
            if (!chosen.empty()) {
                int card_id = player.trash_from_hand_return(chosen[0]);
                const Card* trashed = state.card_def(card_id);
                state.trash_card(card_id);
                if (trashed) state.add_coins(trashed->cost);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Upgrade",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. Trash a card from your hand. Gain a card costing exactly 1 more than it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            if (player.hand_size() == 0) return;
            std::vector<int> trash_opts;
            for (int i = 0; i < player.hand_size(); i++) trash_opts.push_back(i);
            auto chosen = decide(pid, ChoiceType::TRASH, trash_opts, 1, 1);
            if (chosen.empty()) return;
            int card_id = player.trash_from_hand_return(chosen[0]);
            const Card* trashed = state.card_def(card_id);
            state.trash_card(card_id);
            if (!trashed) return;
            auto piles = state.gainable_piles_exact(trashed->cost + 1);
            if (piles.empty()) return;
            std::vector<int> gain_opts;
            for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
            auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
            if (!gain_chosen.empty()) state.gain_card(pid, piles[gain_chosen[0]]);
        },
    });

    // =============================================
    // Batch C — Reveal-hand condition cards
    // =============================================

    CardRegistry::register_card({
        .name = "Shanty Town",
        .cost = 3,
        .types = CardType::Action,
        .text = "+2 Actions. Reveal your hand. If you have no Action cards in hand, +2 Cards.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            state.add_actions(2);
            bool has_action = false;
            for (int card_id : player.get_hand()) {
                const Card* c = state.card_def(card_id);
                if (c && c->is_action()) { has_action = true; break; }
            }
            if (!has_action) player.draw_cards(2);
        },
    });

    CardRegistry::register_card({
        .name = "Magnate",
        .cost = 5,
        .types = CardType::Action,
        .text = "Reveal your hand. +1 Card per Treasure in it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            int treasure_count = 0;
            for (int card_id : player.get_hand()) {
                const Card* c = state.card_def(card_id);
                if (c && c->is_treasure()) treasure_count++;
            }
            if (treasure_count > 0) player.draw_cards(treasure_count);
        },
    });

    CardRegistry::register_card({
        .name = "Sea Chart",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. Reveal the top card of your deck. If you have a copy of it in play, put it into your hand.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            auto top = player.peek_deck(1);
            if (top.empty()) return;
            int top_id = top[0];
            const std::string& top_name = state.card_name(top_id);
            bool copy_in_play = false;
            for (int cid : player.get_in_play()) {
                if (state.card_name(cid) == top_name) { copy_in_play = true; break; }
            }
            if (copy_in_play) {
                player.remove_deck_top();
                player.add_to_hand(top_id);
            }
        },
    });

    // =============================================
    // Batch D — Attacks
    // =============================================

    CardRegistry::register_card({
        .name = "Cutpurse",
        .cost = 4,
        .types = CardType::Action | CardType::Attack,
        .text = "+2 Coins. Each other player discards a Copper (or reveals a hand with no Copper).",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_coins(2);
            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
                Player& p = st.get_player(target);
                for (int i = 0; i < p.hand_size(); i++) {
                    if (st.card_name(p.get_hand()[i]) == "Copper") {
                        p.discard_from_hand(i);
                        return;
                    }
                }
                // No Copper — hand is revealed (no mechanical effect)
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Old Witch",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "+3 Cards. Each other player gains a Curse and may trash a Curse from their hand.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.get_player(pid).draw_cards(3);
            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
                st.gain_card(target, "Curse");
                Player& p = st.get_player(target);
                int curse_idx = -1;
                for (int i = 0; i < p.hand_size(); i++) {
                    if (st.card_name(p.get_hand()[i]) == "Curse") {
                        curse_idx = i;
                        break;
                    }
                }
                if (curse_idx == -1) return;
                auto chosen = dec(target, ChoiceType::YES_NO, {0, 1}, 1, 1);
                if (!chosen.empty() && chosen[0] == 1) {
                    int card_id = p.trash_from_hand_return(curse_idx);
                    st.trash_card(card_id);
                }
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Soothsayer",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "Gain a Gold. Each other player gains a Curse, and if they did, draws a card.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.gain_card(pid, "Gold");
            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
                int curse_id = st.gain_card(target, "Curse");
                if (curse_id != -1) {
                    st.get_player(target).draw_cards(1);
                }
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Witch's Hut",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "+4 Cards. Discard 2 cards, revealed. If they're both Actions, each other player gains a Curse.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(4);
            int to_discard = std::min(2, player.hand_size());
            if (to_discard == 0) return;
            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
            auto chosen = decide(pid, ChoiceType::DISCARD, options, to_discard, to_discard);
            std::sort(chosen.begin(), chosen.end(), std::greater<int>());
            bool both_actions = true;
            for (int idx : chosen) {
                const Card* c = state.card_def(player.get_hand()[idx]);
                if (!c || !c->is_action()) both_actions = false;
            }
            for (int idx : chosen) player.discard_from_hand(idx);
            if (both_actions && static_cast<int>(chosen.size()) == 2) {
                state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
                    st.gain_card(target, "Curse");
                }, decide);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Barbarian",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "+2 Coins. Each other player trashes the top card of their deck. If it costs 3 or more they gain a cheaper card sharing a type with it; otherwise they gain a Curse.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_coins(2);
            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
                Player& p = st.get_player(target);
                int card_id = p.remove_deck_top();
                if (card_id == -1) return;
                const Card* trashed = st.card_def(card_id);
                st.trash_card(card_id);
                if (!trashed) return;
                if (trashed->cost >= 3) {
                    // Gain a cheaper card sharing a type
                    auto piles = st.gainable_piles(trashed->cost - 1);
                    std::vector<std::string> valid;
                    for (const auto& pile : piles) {
                        const Card* c = CardRegistry::get(pile);
                        if (c && (c->types & trashed->types) != CardType::None) {
                            valid.push_back(pile);
                        }
                    }
                    if (valid.empty()) return;
                    std::vector<int> gain_opts;
                    for (const auto& pile : valid) gain_opts.push_back(st.get_supply().top_card(pile));
                    auto chosen = dec(target, ChoiceType::GAIN, gain_opts, 1, 1);
                    if (!chosen.empty()) st.gain_card(target, valid[chosen[0]]);
                } else {
                    st.gain_card(target, "Curse");
                }
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Jester",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "+2 Coins. Each other player discards the top card of their deck. If it's a Victory card they gain a Curse; otherwise they gain a copy of the discarded card or you do, your choice.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_coins(2);
            state.resolve_attack(pid, [pid](GameState& st, int target, DecisionFn dec) {
                Player& p = st.get_player(target);
                int card_id = p.remove_deck_top();
                if (card_id == -1) return;
                const Card* discarded = st.card_def(card_id);
                p.add_to_discard(card_id);
                if (!discarded) return;
                if (discarded->is_victory()) {
                    st.gain_card(target, "Curse");
                } else {
                    // Attacker chooses: 0 = target gains copy, 1 = attacker gains copy
                    auto chosen = dec(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
                    if (!chosen.empty() && chosen[0] == 1) {
                        st.gain_card(pid, discarded->name);
                    } else {
                        st.gain_card(target, discarded->name);
                    }
                }
            }, decide);
        },
    });

    // =============================================
    // Batch E — Peek-deck manipulation
    // =============================================

    CardRegistry::register_card({
        .name = "Cartographer",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. Look at the top 4 cards of your deck. Discard any number of them, then put the rest back in any order.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            auto top_cards = player.peek_deck(4);
            int num_peeked = static_cast<int>(top_cards.size());
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            if (num_peeked == 0) return;
            // Discard any number
            auto disc_chosen = decide(pid, ChoiceType::DISCARD, top_cards, 0, num_peeked);
            for (int card_id : disc_chosen) {
                player.add_to_discard(card_id);
                top_cards.erase(std::find(top_cards.begin(), top_cards.end(), card_id));
            }
            // Put rest back in order
            std::vector<int> ordered;
            while (top_cards.size() > 1) {
                auto chosen = decide(pid, ChoiceType::ORDER, top_cards, 1, 1);
                int pick = chosen.empty() ? top_cards[0] : chosen[0];
                ordered.push_back(pick);
                top_cards.erase(std::find(top_cards.begin(), top_cards.end(), pick));
            }
            if (!top_cards.empty()) ordered.push_back(top_cards[0]);
            for (int i = static_cast<int>(ordered.size()) - 1; i >= 0; i--) {
                player.add_to_deck_top(ordered[i]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Wandering Minstrel",
        .cost = 4,
        .types = CardType::Action,
        .text = "+1 Card, +2 Actions. Reveal the top 3 cards of your deck. Put the Action cards back in any order and discard the rest.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(2);
            auto top_cards = player.peek_deck(3);
            int num_peeked = static_cast<int>(top_cards.size());
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            std::vector<int> actions;
            for (int card_id : top_cards) {
                const Card* c = state.card_def(card_id);
                if (c && c->is_action()) {
                    actions.push_back(card_id);
                } else {
                    player.add_to_discard(card_id);
                }
            }
            // Put action cards back in chosen order
            std::vector<int> ordered;
            while (actions.size() > 1) {
                auto chosen = decide(pid, ChoiceType::ORDER, actions, 1, 1);
                int pick = chosen.empty() ? actions[0] : chosen[0];
                ordered.push_back(pick);
                actions.erase(std::find(actions.begin(), actions.end(), pick));
            }
            if (!actions.empty()) ordered.push_back(actions[0]);
            for (int i = static_cast<int>(ordered.size()) - 1; i >= 0; i--) {
                player.add_to_deck_top(ordered[i]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Seer",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. Reveal the top 3 cards of your deck. Put the ones costing from 2 to 4 into your hand. Put the rest back in any order.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            auto top_cards = player.peek_deck(3);
            int num_peeked = static_cast<int>(top_cards.size());
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            std::vector<int> rest;
            for (int card_id : top_cards) {
                const Card* c = state.card_def(card_id);
                if (c && c->cost >= 2 && c->cost <= 4) {
                    player.add_to_hand(card_id);
                } else {
                    rest.push_back(card_id);
                }
            }
            std::vector<int> ordered;
            while (rest.size() > 1) {
                auto chosen = decide(pid, ChoiceType::ORDER, rest, 1, 1);
                int pick = chosen.empty() ? rest[0] : chosen[0];
                ordered.push_back(pick);
                rest.erase(std::find(rest.begin(), rest.end(), pick));
            }
            if (!rest.empty()) ordered.push_back(rest[0]);
            for (int i = static_cast<int>(ordered.size()) - 1; i >= 0; i--) {
                player.add_to_deck_top(ordered[i]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Patrol",
        .cost = 5,
        .types = CardType::Action,
        .text = "+3 Cards. Reveal the top 4 cards of your deck. Put the Victory cards and Curses into your hand. Put the rest back in any order.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(3);
            auto top_cards = player.peek_deck(4);
            int num_peeked = static_cast<int>(top_cards.size());
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            std::vector<int> rest;
            for (int card_id : top_cards) {
                const Card* c = state.card_def(card_id);
                if (c && (c->is_victory() || c->is_curse())) {
                    player.add_to_hand(card_id);
                } else {
                    rest.push_back(card_id);
                }
            }
            std::vector<int> ordered;
            while (rest.size() > 1) {
                auto chosen = decide(pid, ChoiceType::ORDER, rest, 1, 1);
                int pick = chosen.empty() ? rest[0] : chosen[0];
                ordered.push_back(pick);
                rest.erase(std::find(rest.begin(), rest.end(), pick));
            }
            if (!rest.empty()) ordered.push_back(rest[0]);
            for (int i = static_cast<int>(ordered.size()) - 1; i >= 0; i--) {
                player.add_to_deck_top(ordered[i]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Fortune Hunter",
        .cost = 4,
        .types = CardType::Action,
        .text = "+2 Coins. Look at the top 3 cards of your deck. You may play a Treasure from them. Put the rest back in any order.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_coins(2);
            auto top_cards = player.peek_deck(3);
            int num_peeked = static_cast<int>(top_cards.size());
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            if (num_peeked == 0) return;
            // Filter to treasures
            std::vector<int> treasures;
            for (int card_id : top_cards) {
                const Card* c = state.card_def(card_id);
                if (c && c->is_treasure()) treasures.push_back(card_id);
            }
            if (!treasures.empty()) {
                auto chosen = decide(pid, ChoiceType::PLAY_CARD, treasures, 0, 1);
                if (!chosen.empty()) {
                    int play_id = treasures[chosen[0]];
                    top_cards.erase(std::find(top_cards.begin(), top_cards.end(), play_id));
                    player.add_to_in_play(play_id);
                    const Card* card = state.card_def(play_id);
                    if (card && card->on_play) card->on_play(state, pid, decide);
                }
            }
            // Put rest back in order
            std::vector<int> ordered;
            while (top_cards.size() > 1) {
                auto chosen = decide(pid, ChoiceType::ORDER, top_cards, 1, 1);
                int pick = chosen.empty() ? top_cards[0] : chosen[0];
                ordered.push_back(pick);
                top_cards.erase(std::find(top_cards.begin(), top_cards.end(), pick));
            }
            if (!top_cards.empty()) ordered.push_back(top_cards[0]);
            for (int i = static_cast<int>(ordered.size()) - 1; i >= 0; i--) {
                player.add_to_deck_top(ordered[i]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Carnival",
        .cost = 5,
        .types = CardType::Action,
        .text = "Reveal the top 4 cards of your deck. Put one of each differently named card into your hand and discard the rest.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            auto top_cards = player.peek_deck(4);
            int num_peeked = static_cast<int>(top_cards.size());
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            std::set<std::string> seen;
            for (int card_id : top_cards) {
                const std::string& name = state.card_name(card_id);
                if (seen.insert(name).second) {
                    player.add_to_hand(card_id);
                } else {
                    player.add_to_discard(card_id);
                }
            }
        },
    });

    // =============================================
    // Batch F — Reveal-from-deck loops
    // =============================================

    CardRegistry::register_card({
        .name = "Sage",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Action. Reveal cards from the top of your deck until you reveal one costing 3 or more. Put that card into your hand and discard the rest.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            state.add_actions(1);
            while (true) {
                int card_id = player.remove_deck_top();
                if (card_id == -1) break;
                const Card* c = state.card_def(card_id);
                if (c && c->cost >= 3) {
                    player.add_to_hand(card_id);
                    break;
                }
                player.add_to_discard(card_id);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Hunting Party",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. Reveal your hand. Reveal cards from your deck until you reveal a card that isn't a copy of one in your hand. Put it into your hand and discard the rest.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            // Collect names in hand
            std::set<std::string> hand_names;
            for (int card_id : player.get_hand()) {
                hand_names.insert(state.card_name(card_id));
            }
            while (true) {
                int card_id = player.remove_deck_top();
                if (card_id == -1) break;
                if (hand_names.find(state.card_name(card_id)) == hand_names.end()) {
                    player.add_to_hand(card_id);
                    break;
                }
                player.add_to_discard(card_id);
            }
        },
    });

    // =============================================
    // Batch G — Discard-for-benefit & discard-pile manipulation
    // =============================================

    CardRegistry::register_card({
        .name = "Stables",
        .cost = 5,
        .types = CardType::Action,
        .text = "You may discard a Treasure, for +3 Cards and +1 Action.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            std::vector<int> treasure_indices;
            for (int i = 0; i < player.hand_size(); i++) {
                const Card* c = state.card_def(player.get_hand()[i]);
                if (c && c->is_treasure()) treasure_indices.push_back(i);
            }
            if (treasure_indices.empty()) return;
            auto chosen = decide(pid, ChoiceType::DISCARD, treasure_indices, 0, 1);
            if (!chosen.empty()) {
                player.discard_from_hand(treasure_indices[chosen[0]]);
                player.draw_cards(3);
                state.add_actions(1);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Wheelwright",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. You may discard a card to gain an Action card costing as much as it or less.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(1);
            state.add_actions(1);
            if (player.hand_size() == 0) return;
            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
            auto chosen = decide(pid, ChoiceType::DISCARD, options, 0, 1);
            if (chosen.empty()) return;
            int disc_idx = chosen[0];
            const Card* discarded = state.card_def(player.get_hand()[disc_idx]);
            player.discard_from_hand(disc_idx);
            if (!discarded) return;
            auto piles = state.gainable_piles(discarded->cost, CardType::Action);
            if (piles.empty()) return;
            std::vector<int> gain_opts;
            for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
            auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
            if (!gain_chosen.empty()) state.gain_card(pid, piles[gain_chosen[0]]);
        },
    });

    CardRegistry::register_card({
        .name = "Mountain Village",
        .cost = 4,
        .types = CardType::Action,
        .text = "+2 Actions. Look through your discard pile and put a card from it into your hand; if you can't, +1 Card.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_actions(2);
            if (player.discard_size() == 0) {
                player.draw_cards(1);
                return;
            }
            const auto& discard = player.get_discard();
            std::vector<int> options;
            for (int i = 0; i < static_cast<int>(discard.size()); i++) options.push_back(i);
            auto chosen = decide(pid, ChoiceType::SELECT_FROM_DISCARD, options, 1, 1);
            if (!chosen.empty()) {
                int card_id = discard[chosen[0]];
                player.remove_from_discard(card_id);
                player.add_to_hand(card_id);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Scavenger",
        .cost = 4,
        .types = CardType::Action,
        .text = "+2 Coins. You may put your deck into your discard pile. Put a card from your discard pile onto your deck.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_coins(2);
            // May put deck into discard
            if (player.deck_size() > 0) {
                auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
                if (!chosen.empty() && chosen[0] == 1) {
                    player.dump_deck_to_discard();
                }
            }
            // Put a card from discard onto deck
            if (player.discard_size() > 0) {
                const auto& discard = player.get_discard();
                std::vector<int> options;
                for (int i = 0; i < static_cast<int>(discard.size()); i++) options.push_back(i);
                auto chosen = decide(pid, ChoiceType::SELECT_FROM_DISCARD, options, 1, 1);
                if (!chosen.empty()) {
                    int card_id = discard[chosen[0]];
                    player.remove_from_discard(card_id);
                    player.add_to_deck_top(card_id);
                }
            }
        },
    });

    // =============================================
    // Batch H — Multi-step / conditional cards
    // =============================================

    CardRegistry::register_card({
        .name = "Remake",
        .cost = 4,
        .types = CardType::Action,
        .text = "Do this twice: Trash a card from your hand, then gain a card costing exactly 1 more than it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            for (int round = 0; round < 2; round++) {
                if (player.hand_size() == 0) break;
                std::vector<int> trash_opts;
                for (int i = 0; i < player.hand_size(); i++) trash_opts.push_back(i);
                auto chosen = decide(pid, ChoiceType::TRASH, trash_opts, 1, 1);
                if (chosen.empty()) continue;
                int card_id = player.trash_from_hand_return(chosen[0]);
                const Card* trashed = state.card_def(card_id);
                state.trash_card(card_id);
                if (!trashed) continue;
                auto piles = state.gainable_piles_exact(trashed->cost + 1);
                if (piles.empty()) continue;
                std::vector<int> gain_opts;
                for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
                auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
                if (!gain_chosen.empty()) state.gain_card(pid, piles[gain_chosen[0]]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Dismantle",
        .cost = 4,
        .types = CardType::Action,
        .text = "Trash a card from your hand. If it costs 1 or more, gain a cheaper card and a Gold.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            if (player.hand_size() == 0) return;
            std::vector<int> trash_opts;
            for (int i = 0; i < player.hand_size(); i++) trash_opts.push_back(i);
            auto chosen = decide(pid, ChoiceType::TRASH, trash_opts, 1, 1);
            if (chosen.empty()) return;
            int card_id = player.trash_from_hand_return(chosen[0]);
            const Card* trashed = state.card_def(card_id);
            state.trash_card(card_id);
            if (!trashed || trashed->cost < 1) return;
            // Gain a cheaper card
            auto piles = state.gainable_piles(trashed->cost - 1);
            if (!piles.empty()) {
                std::vector<int> gain_opts;
                for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
                auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
                if (!gain_chosen.empty()) state.gain_card(pid, piles[gain_chosen[0]]);
            }
            // And a Gold
            state.gain_card(pid, "Gold");
        },
    });

    CardRegistry::register_card({
        .name = "Trading Post",
        .cost = 5,
        .types = CardType::Action,
        .text = "Trash 2 cards from your hand. If you did, gain a Silver to your hand.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            int to_trash = std::min(2, player.hand_size());
            if (to_trash == 0) return;
            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
            auto chosen = decide(pid, ChoiceType::TRASH, options, to_trash, to_trash);
            std::sort(chosen.begin(), chosen.end(), std::greater<int>());
            for (int idx : chosen) {
                int card_id = player.trash_from_hand_return(idx);
                state.trash_card(card_id);
            }
            if (static_cast<int>(chosen.size()) == 2) {
                state.gain_card_to_hand(pid, "Silver");
            }
        },
    });

    CardRegistry::register_card({
        .name = "Marquis",
        .cost = 6,
        .types = CardType::Action,
        .text = "+1 Buy. +1 Card per card in your hand. Discard down to 10 cards in hand.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_buys(1);
            int draw_count = player.hand_size();
            player.draw_cards(draw_count);
            if (player.hand_size() > 10) {
                int must_discard = player.hand_size() - 10;
                std::vector<int> options;
                for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
                auto chosen = decide(pid, ChoiceType::DISCARD, options, must_discard, must_discard);
                std::sort(chosen.begin(), chosen.end(), std::greater<int>());
                for (int idx : chosen) player.discard_from_hand(idx);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Hunter",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Action. Reveal the top 3 cards of your deck. From those cards, put an Action, a Treasure, and a Victory card into your hand. Discard the rest.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            state.add_actions(1);
            auto top_cards = player.peek_deck(3);
            int num_peeked = static_cast<int>(top_cards.size());
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();
            if (num_peeked == 0) return;

            // For each type (Action, Treasure, Victory), pick one if available
            auto pick_type = [&](auto type_check) {
                std::vector<int> matches;
                for (int card_id : top_cards) {
                    const Card* c = state.card_def(card_id);
                    if (c && type_check(c)) matches.push_back(card_id);
                }
                if (matches.empty()) return;
                int pick_id;
                if (matches.size() == 1) {
                    pick_id = matches[0];
                } else {
                    auto chosen = decide(pid, ChoiceType::PLAY_CARD, matches, 1, 1);
                    pick_id = chosen.empty() ? matches[0] : matches[chosen[0]];
                }
                player.add_to_hand(pick_id);
                top_cards.erase(std::find(top_cards.begin(), top_cards.end(), pick_id));
            };

            pick_type([](const Card* c) { return c->is_action(); });
            pick_type([](const Card* c) { return c->is_treasure(); });
            pick_type([](const Card* c) { return c->is_victory(); });

            // Discard the rest
            for (int card_id : top_cards) player.add_to_discard(card_id);
        },
    });

    // =============================================
    // Batch I — Self-trashing from play
    // =============================================

    CardRegistry::register_card({
        .name = "Treasure Map",
        .cost = 4,
        .types = CardType::Action,
        .text = "Trash this and a Treasure Map from your hand. If you trashed two Treasure Maps, gain 4 Golds onto your deck.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            Player& player = state.get_player(pid);
            int trashed_count = 0;
            // Trash this from in_play
            for (int cid : player.get_in_play()) {
                if (state.card_name(cid) == "Treasure Map") {
                    if (player.remove_from_in_play(cid)) {
                        state.trash_card(cid);
                        trashed_count++;
                    }
                    break;
                }
            }
            // Trash a Treasure Map from hand
            for (int i = 0; i < player.hand_size(); i++) {
                if (state.card_name(player.get_hand()[i]) == "Treasure Map") {
                    int card_id = player.trash_from_hand_return(i);
                    state.trash_card(card_id);
                    trashed_count++;
                    break;
                }
            }
            if (trashed_count == 2) {
                for (int g = 0; g < 4; g++) {
                    state.gain_card_to_deck_top(pid, "Gold");
                }
            }
        },
    });

    CardRegistry::register_card({
        .name = "Tragic Hero",
        .cost = 5,
        .types = CardType::Action,
        .text = "+3 Cards, +1 Buy. If you have 8 or more cards in hand (after drawing), trash this and gain a Treasure.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            player.draw_cards(3);
            state.add_buys(1);
            if (player.hand_size() >= 8) {
                // Trash this from in_play (may fail if already trashed by Throne Room replay)
                for (int cid : player.get_in_play()) {
                    if (state.card_name(cid) == "Tragic Hero") {
                        player.remove_from_in_play(cid);
                        state.trash_card(cid);
                        break;
                    }
                }
                // Gain a Treasure — "and" not "to", so this happens regardless
                auto piles = state.gainable_piles(999, CardType::Treasure);
                if (!piles.empty()) {
                    std::vector<int> gain_opts;
                    for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
                    auto chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
                    if (!chosen.empty()) state.gain_card(pid, piles[chosen[0]]);
                }
            }
        },
    });

    // =============================================
    // Batch J — Complex interactions
    // =============================================

    CardRegistry::register_card({
        .name = "Minion",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "+1 Action. Choose one: +2 Coins; or discard your hand, +4 Cards, and each other player with at least 5 cards in hand discards their hand and draws 4 cards.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_actions(1);
            // Choose: 0 = +2 Coins, 1 = discard/draw/attack
            auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
            int choice = chosen.empty() ? 0 : chosen[0];
            if (choice == 0) {
                state.add_coins(2);
            } else {
                Player& player = state.get_player(pid);
                player.discard_hand();
                player.draw_cards(4);
                state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
                    Player& p = st.get_player(target);
                    if (p.hand_size() >= 5) {
                        p.discard_hand();
                        p.draw_cards(4);
                    }
                }, decide);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Replace",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "Trash a card from your hand. Gain a card costing up to 2 more than it. If the gained card is an Action or Treasure, put it onto your deck; if it's a Victory card, each other player gains a Curse.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            if (player.hand_size() == 0) return;
            std::vector<int> trash_opts;
            for (int i = 0; i < player.hand_size(); i++) trash_opts.push_back(i);
            auto chosen = decide(pid, ChoiceType::TRASH, trash_opts, 1, 1);
            if (chosen.empty()) return;
            int card_id = player.trash_from_hand_return(chosen[0]);
            const Card* trashed = state.card_def(card_id);
            state.trash_card(card_id);
            if (!trashed) return;
            int max_cost = trashed->cost + 2;
            auto piles = state.gainable_piles(max_cost);
            if (piles.empty()) return;
            std::vector<int> gain_opts;
            for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
            auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
            if (gain_chosen.empty()) return;
            const std::string& pile_name = piles[gain_chosen[0]];
            const Card* gained = CardRegistry::get(pile_name);
            if (!gained) return;
            if (gained->is_action() || gained->is_treasure()) {
                state.gain_card_to_deck_top(pid, pile_name);
            } else {
                state.gain_card(pid, pile_name);
            }
            if (gained->is_victory()) {
                state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
                    st.gain_card(target, "Curse");
                }, decide);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Masquerade",
        .cost = 3,
        .types = CardType::Action,
        .text = "+2 Cards. Each player with any cards in hand passes one to the next such player to their left, at once. Then you may trash a card from your hand.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.get_player(pid).draw_cards(2);
            int n = state.num_players();
            // Collect which players have cards, and what they pass
            std::vector<int> pass_card_ids(n, -1);
            // Each player with cards in hand chooses a card to pass
            for (int offset = 0; offset < n; offset++) {
                int p = (pid + offset) % n;
                Player& player = state.get_player(p);
                if (player.hand_size() == 0) continue;
                std::vector<int> options;
                for (int i = 0; i < player.hand_size(); i++) options.push_back(i);
                auto chosen = decide(p, ChoiceType::DISCARD, options, 1, 1);
                if (!chosen.empty()) {
                    pass_card_ids[p] = player.trash_from_hand_return(chosen[0]);
                }
            }
            // Pass cards to the left (next player with cards)
            for (int offset = 0; offset < n; offset++) {
                int p = (pid + offset) % n;
                if (pass_card_ids[p] == -1) continue;
                // Find next player to the left who also had cards
                for (int j = 1; j < n; j++) {
                    int target = (p + j) % n;
                    if (pass_card_ids[target] != -1 || state.get_player(target).hand_size() > 0) {
                        // Actually, pass to next player who participated
                        // Per rules: pass to next such player to the left
                        state.get_player(target).add_to_hand(pass_card_ids[p]);
                        break;
                    }
                }
            }
            // Then may trash from hand
            Player& current = state.get_player(pid);
            if (current.hand_size() > 0) {
                std::vector<int> trash_opts;
                for (int i = 0; i < current.hand_size(); i++) trash_opts.push_back(i);
                auto chosen = decide(pid, ChoiceType::TRASH, trash_opts, 0, 1);
                if (!chosen.empty()) {
                    int card_id = current.trash_from_hand_return(chosen[0]);
                    state.trash_card(card_id);
                }
            }
        },
    });
}


} // namespace Level2Cards