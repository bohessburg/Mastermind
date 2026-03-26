#include "base_kingdom.h"
#include "../card.h"
#include "../game_state.h"

#include <algorithm>

namespace BaseKingdom {

void register_all() {
    // Wave 1: Simple effects (no decisions)

    CardRegistry::register_card({
        .name = "Smithy",
        .cost = 4,
        .types = CardType::Action,
        .text = "+3 Cards",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(3);
        },
    });

    CardRegistry::register_card({
        .name = "Village",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Card, +2 Actions",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(2);
        },
    });

    CardRegistry::register_card({
        .name = "Laboratory",
        .cost = 5,
        .types = CardType::Action,
        .text = "+2 Cards, +1 Action",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(2);
            state.add_actions(1);
        },
    });

    CardRegistry::register_card({
        .name = "Festival",
        .cost = 5,
        .types = CardType::Action,
        .text = "+2 Actions, +1 Buy, +2 Coins",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int /*pid*/, DecisionFn) {
            state.add_actions(2);
            state.add_buys(1);
            state.add_coins(2);
        },
    });

    CardRegistry::register_card({
        .name = "Market",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action, +1 Buy, +1 Coin",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(1);
            state.add_buys(1);
            state.add_coins(1);
        },
    });

    // Wave 2: Simple choices

    CardRegistry::register_card({
        .name = "Cellar",
        .cost = 2,
        .types = CardType::Action,
        .text = "+1 Action. Discard any number of cards, then draw that many.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_actions(1);
            Player& player = state.get_player(pid);
            if (player.hand_size() == 0) return;

            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

            auto chosen = decide(pid, ChoiceType::DISCARD, options, 0, player.hand_size());
            std::sort(chosen.begin(), chosen.end(), std::greater<int>());
            int count = static_cast<int>(chosen.size());
            for (int idx : chosen) player.discard_from_hand(idx);
            player.draw_cards(count);
        },
    });

    CardRegistry::register_card({
        .name = "Chapel",
        .cost = 2,
        .types = CardType::Action,
        .text = "Trash up to 4 cards from your hand.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            if (player.hand_size() == 0) return;

            std::vector<int> options;
            for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

            int max_trash = std::min(4, player.hand_size());
            auto chosen = decide(pid, ChoiceType::TRASH, options, 0, max_trash);
            std::sort(chosen.begin(), chosen.end(), std::greater<int>());
            for (int idx : chosen) {
                int cid = player.trash_from_hand_return(idx);
                state.trash_card(cid);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Workshop",
        .cost = 3,
        .types = CardType::Action,
        .text = "Gain a card costing up to 4.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            auto piles = state.gainable_piles(4);
            if (piles.empty()) return;

            std::vector<int> options;
            for (const auto& p : piles) options.push_back(state.get_supply().top_card(p));

            auto chosen = decide(pid, ChoiceType::GAIN, options, 1, 1);
            if (!chosen.empty()) {
                state.gain_card(pid, piles[chosen[0]]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Moneylender",
        .cost = 4,
        .types = CardType::Action,
        .text = "You may trash a Copper from your hand for +3 Coins.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            int copper_idx = -1;
            for (int i = 0; i < player.hand_size(); i++) {
                if (state.card_name(player.get_hand()[i]) == "Copper") {
                    copper_idx = i;
                    break;
                }
            }
            if (copper_idx == -1) return;

            auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
            if (!chosen.empty() && chosen[0] == 1) {
                int cid = player.trash_from_hand_return(copper_idx);
                state.trash_card(cid);
                state.add_coins(3);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Poacher",
        .cost = 4,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action, +1 Coin. Discard a card per empty Supply pile.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(1);
            state.add_coins(1);

            int empty_piles = state.get_supply().empty_pile_count();
            Player& player = state.get_player(pid);
            int to_discard = std::min(empty_piles, player.hand_size());

            if (to_discard > 0) {
                std::vector<int> options;
                for (int i = 0; i < player.hand_size(); i++) options.push_back(i);

                auto chosen = decide(pid, ChoiceType::DISCARD, options, to_discard, to_discard);
                std::sort(chosen.begin(), chosen.end(), std::greater<int>());
                for (int idx : chosen) player.discard_from_hand(idx);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Vassal",
        .cost = 3,
        .types = CardType::Action,
        .text = "+2 Coins. Discard the top card of your deck. If it's an Action, you may play it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_coins(2);
            Player& player = state.get_player(pid);

            int card_id = player.remove_deck_top();
            if (card_id == -1) return;

            player.add_to_discard(card_id);

            const Card* card = state.card_def(card_id);
            if (card && card->is_action()) {
                auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
                if (!chosen.empty() && chosen[0] == 1) {
                    // Move from discard to in_play, then execute
                    player.remove_from_discard(card_id);
                    player.add_to_hand(card_id);
                    int hidx = player.find_in_hand(card_id);
                    player.play_from_hand(hidx);
                    if (card->on_play) {
                        card->on_play(state, pid, decide);
                    }
                }
            }
        },
    });

    CardRegistry::register_card({
        .name = "Harbinger",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. Look through your discard pile. You may put a card from it onto your deck.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(1);

            Player& player = state.get_player(pid);
            const auto& discard = player.get_discard();
            if (discard.empty()) return;

            std::vector<int> options;
            for (int i = 0; i < static_cast<int>(discard.size()); i++) options.push_back(i);

            auto chosen = decide(pid, ChoiceType::SELECT_FROM_DISCARD, options, 0, 1);
            if (!chosen.empty()) {
                int discard_idx = chosen[0];
                int card_id = discard[discard_idx];
                player.remove_from_discard_by_index(discard_idx);
                player.add_to_deck_top(card_id);
            }
        },
    });

    // Wave 3: Complex multi-step choices

    CardRegistry::register_card({
        .name = "Remodel",
        .cost = 4,
        .types = CardType::Action,
        .text = "Trash a card from your hand. Gain a card costing up to 2 more than it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            if (player.hand_size() == 0) return;

            std::vector<int> trash_options;
            for (int i = 0; i < player.hand_size(); i++) trash_options.push_back(i);

            auto trash_chosen = decide(pid, ChoiceType::TRASH, trash_options, 1, 1);
            if (trash_chosen.empty()) return;

            int hand_idx = trash_chosen[0];
            int trashed_id = player.trash_from_hand_return(hand_idx);
            const Card* trashed = state.card_def(trashed_id);
            state.trash_card(trashed_id);

            if (!trashed) return;
            int max_cost = trashed->cost + 2;

            auto piles = state.gainable_piles(max_cost);
            if (piles.empty()) return;

            std::vector<int> gain_options;
            for (const auto& p : piles) gain_options.push_back(state.get_supply().top_card(p));

            auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_options, 1, 1);
            if (!gain_chosen.empty()) {
                state.gain_card(pid, piles[gain_chosen[0]]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Mine",
        .cost = 5,
        .types = CardType::Action,
        .text = "You may trash a Treasure from your hand. Gain a Treasure to your hand costing up to 3 more than it.",
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

            auto trash_chosen = decide(pid, ChoiceType::TRASH, treasure_indices, 0, 1);
            if (trash_chosen.empty()) return;

            int hand_idx = trash_chosen[0];
            int trashed_id = player.trash_from_hand_return(hand_idx);
            const Card* trashed = state.card_def(trashed_id);
            state.trash_card(trashed_id);

            if (!trashed) return;
            int max_cost = trashed->cost + 3;

            auto piles = state.gainable_piles(max_cost, CardType::Treasure);
            if (piles.empty()) return;

            std::vector<int> gain_options;
            for (const auto& p : piles) gain_options.push_back(state.get_supply().top_card(p));

            auto gain_chosen = decide(pid, ChoiceType::GAIN, gain_options, 1, 1);
            if (!gain_chosen.empty()) {
                state.gain_card_to_hand(pid, piles[gain_chosen[0]]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Artisan",
        .cost = 6,
        .types = CardType::Action,
        .text = "Gain a card to your hand costing up to 5. Put a card from your hand onto your deck.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            auto piles = state.gainable_piles(5);
            if (!piles.empty()) {
                std::vector<int> gain_opts;
                for (const auto& p : piles) gain_opts.push_back(state.get_supply().top_card(p));
                auto chosen = decide(pid, ChoiceType::GAIN, gain_opts, 1, 1);
                if (!chosen.empty()) state.gain_card_to_hand(pid, piles[chosen[0]]);
            }

            Player& player = state.get_player(pid);
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
        .name = "Sentry",
        .cost = 5,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. Look at the top 2 cards of your deck. Trash and/or discard any number of them. Put the rest back on top in any order.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(1);

            Player& player = state.get_player(pid);
            auto top_cards = player.peek_deck(2);
            int num_peeked = static_cast<int>(top_cards.size());
            // Remove them from deck
            for (int i = 0; i < num_peeked; i++) player.remove_deck_top();

            // For each card: 0=put back, 1=discard, 2=trash
            std::vector<int> keep_cards;
            for (int card_id : top_cards) {
                // Stash which card we're deciding on so the UI can display it
                state.set_turn_flag("sentry_card_id", card_id);
                auto chosen = decide(pid, ChoiceType::MULTI_FATE, {0, 1, 2}, 1, 1);
                int fate = chosen.empty() ? 0 : chosen[0];
                if (fate == 1) {
                    player.add_to_discard(card_id);
                } else if (fate == 2) {
                    state.trash_card(card_id);
                } else {
                    keep_cards.push_back(card_id);
                }
            }

            // Put back kept cards. If >1, ask for order.
            if (keep_cards.size() == 2) {
                // Stash card IDs so the UI can display names
                state.set_turn_flag("sentry_order_card_0", keep_cards[0]);
                state.set_turn_flag("sentry_order_card_1", keep_cards[1]);
                // options: 0 = first on top, 1 = second on top
                auto chosen = decide(pid, ChoiceType::ORDER, {0, 1}, 1, 1);
                int top_pick = chosen.empty() ? 0 : chosen[0];
                // Put bottom card first, then top card
                int bottom = (top_pick == 0) ? 1 : 0;
                player.add_to_deck_top(keep_cards[bottom]);
                player.add_to_deck_top(keep_cards[top_pick]);
            } else if (keep_cards.size() == 1) {
                player.add_to_deck_top(keep_cards[0]);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Library",
        .cost = 5,
        .types = CardType::Action,
        .text = "Draw until you have 7 cards in hand, skipping any Action cards you choose to; set those aside, discarding them afterwards.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            Player& player = state.get_player(pid);
            std::vector<int> set_aside_cards;

            while (player.hand_size() < 7) {
                int card_id = player.remove_deck_top();
                if (card_id == -1) break;

                const Card* card = state.card_def(card_id);
                if (card && card->is_action()) {
                    auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
                    if (!chosen.empty() && chosen[0] == 1) {
                        set_aside_cards.push_back(card_id);
                        continue;
                    }
                }
                player.add_to_hand(card_id);
            }

            for (int cid : set_aside_cards) {
                player.add_to_discard(cid);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Bureaucrat",
        .cost = 4,
        .types = CardType::Action | CardType::Attack,
        .text = "Gain a Silver onto your deck. Each other player reveals a Victory card from their hand and puts it onto their deck (or reveals a hand with no Victory cards).",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.gain_card_to_deck_top(pid, "Silver");

            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
                Player& p = st.get_player(target);

                std::vector<int> victory_indices;
                for (int i = 0; i < p.hand_size(); i++) {
                    const Card* c = st.card_def(p.get_hand()[i]);
                    if (c && c->is_victory()) victory_indices.push_back(i);
                }

                if (victory_indices.empty()) return;

                if (victory_indices.size() == 1) {
                    p.topdeck_from_hand(victory_indices[0]);
                } else {
                    auto chosen = dec(target, ChoiceType::REVEAL, victory_indices, 1, 1);
                    if (!chosen.empty()) {
                        p.topdeck_from_hand(chosen[0]);
                    }
                }
            }, decide);
        },
    });

    // Wave 4: Attacks, reactions, recursion

    CardRegistry::register_card({
        .name = "Moat",
        .cost = 2,
        .types = CardType::Action | CardType::Reaction,
        .text = "+2 Cards. When another player plays an Attack card, you may first reveal this from your hand, to be unaffected by it.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(2);
        },
        .on_react = [](GameState& /*state*/, int pid, int /*attacker_id*/, DecisionFn decide) -> bool {
            auto chosen = decide(pid, ChoiceType::YES_NO, {0, 1}, 1, 1);
            return !chosen.empty() && chosen[0] == 1;
        },
    });

    CardRegistry::register_card({
        .name = "Militia",
        .cost = 4,
        .types = CardType::Action | CardType::Attack,
        .text = "+2 Coins. Each other player discards down to 3 cards in hand.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.add_coins(2);

            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
                Player& p = st.get_player(target);
                if (p.hand_size() <= 3) return;

                int must_discard = p.hand_size() - 3;
                std::vector<int> options;
                for (int i = 0; i < p.hand_size(); i++) options.push_back(i);

                auto chosen = dec(target, ChoiceType::DISCARD, options, must_discard, must_discard);
                std::sort(chosen.begin(), chosen.end(), std::greater<int>());
                for (int idx : chosen) p.discard_from_hand(idx);
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Bandit",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "Gain a Gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper, and discards the rest.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.gain_card(pid, "Gold");

            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn dec) {
                Player& p = st.get_player(target);
                auto revealed = p.peek_deck(2);
                int num_revealed = static_cast<int>(revealed.size());
                for (int i = 0; i < num_revealed; i++) p.remove_deck_top();

                // Find trashable treasures (non-Copper Treasures)
                std::vector<int> trashable;
                for (int i = 0; i < num_revealed; i++) {
                    const Card* c = st.card_def(revealed[i]);
                    if (c && c->is_treasure() && c->name != "Copper") {
                        trashable.push_back(i);
                    }
                }

                int trashed_idx = -1;
                if (trashable.size() == 1) {
                    trashed_idx = trashable[0];
                } else if (trashable.size() > 1) {
                    auto chosen = dec(target, ChoiceType::TRASH, trashable, 1, 1);
                    if (!chosen.empty()) trashed_idx = chosen[0];
                }

                for (int i = 0; i < num_revealed; i++) {
                    if (i == trashed_idx) {
                        st.trash_card(revealed[i]);
                    } else {
                        p.add_to_discard(revealed[i]);
                    }
                }
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Witch",
        .cost = 5,
        .types = CardType::Action | CardType::Attack,
        .text = "+2 Cards. Each other player gains a Curse.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn decide) {
            state.get_player(pid).draw_cards(2);

            state.resolve_attack(pid, [](GameState& st, int target, DecisionFn) {
                st.gain_card(target, "Curse");
            }, decide);
        },
    });

    CardRegistry::register_card({
        .name = "Council Room",
        .cost = 5,
        .types = CardType::Action,
        .text = "+4 Cards, +1 Buy. Each other player draws a card.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(4);
            state.add_buys(1);

            for (int i = 0; i < state.num_players(); i++) {
                if (i != pid) {
                    state.get_player(i).draw_cards(1);
                }
            }
        },
    });

    CardRegistry::register_card({
        .name = "Throne Room",
        .cost = 4,
        .types = CardType::Action,
        .text = "You may play an Action card from your hand twice.",
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

            const Card* target = state.card_def(card_id);
            if (target && target->on_play) {
                target->on_play(state, pid, decide);
                target->on_play(state, pid, decide);
            }
        },
    });

    CardRegistry::register_card({
        .name = "Gardens",
        .cost = 4,
        .types = CardType::Victory,
        .text = "Worth 1 VP per 10 cards you have (round down).",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = nullptr,
        .on_react = nullptr,
        .vp_fn = [](const GameState& state, int pid) -> int {
            return state.total_cards_owned(pid) / 10;
        },
    });

    // Wave 5: Triggered ability

    CardRegistry::register_card({
        .name = "Merchant",
        .cost = 3,
        .types = CardType::Action,
        .text = "+1 Card, +1 Action. The first time you play a Silver this turn, +1 Coin.",
        .victory_points = 0,
        .coin_value = 0,
        .tags = {},
        .on_play = [](GameState& state, int pid, DecisionFn) {
            state.get_player(pid).draw_cards(1);
            state.add_actions(1);
            // Track that a Merchant is active. The bonus is applied when Silver
            // is played, checked by the GameRunner or treasure-playing logic.
            state.set_turn_flag("merchant_count",
                state.get_turn_flag("merchant_count") + 1);
        },
    });
}

} // namespace BaseKingdom
