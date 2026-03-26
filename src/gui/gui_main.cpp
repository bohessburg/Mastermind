#include "gui_agent.h"
#include "../engine/game_runner.h"
#include "../game/card.h"

#include "raylib.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

// ─── Layout constants ─────────────────────────────────────────────

static constexpr int SCREEN_W = 1280;
static constexpr int SCREEN_H = 800;

static constexpr int CARD_W = 90;
static constexpr int CARD_H = 120;
static constexpr int CARD_PAD = 6;

// Vertical regions
static constexpr int SUPPLY_Y = 10;
static constexpr int SUPPLY_ROW_H = CARD_H + CARD_PAD;
static constexpr int INFO_Y = SUPPLY_Y + SUPPLY_ROW_H * 2 + 10;
static constexpr int INFO_H = 30;
static constexpr int PLAY_AREA_Y = INFO_Y + INFO_H + 10;
static constexpr int PLAY_AREA_H = 140;
static constexpr int DECISION_Y = PLAY_AREA_Y + PLAY_AREA_H + 10;
static constexpr int DECISION_H = 120;
static constexpr int HAND_Y = DECISION_Y + DECISION_H + 10;

static constexpr int LOG_X = 950;
static constexpr int LOG_W = SCREEN_W - LOG_X - 10;

// ─── Colors ───────────────────────────────────────────────────────

static Color card_color(const Card* card) {
    if (!card) return LIGHTGRAY;
    if (card->is_curse()) return {200, 160, 220, 255};
    if (card->is_victory() && card->is_treasure()) return {180, 210, 140, 255};
    if (card->is_victory()) return {140, 200, 140, 255};
    if (card->is_treasure()) return {240, 220, 130, 255};
    if (card->is_attack()) return {240, 160, 160, 255};
    if (card->is_reaction()) return {160, 190, 240, 255};
    if (card->is_action()) return {230, 230, 230, 255};
    return LIGHTGRAY;
}

static Color card_color_by_name(const std::string& name) {
    const Card* card = CardRegistry::get(name);
    return card_color(card);
}

// ─── Drawing helpers ──────────────────────────────────────────────

static Rectangle draw_card(int x, int y, const std::string& name, int count,
                            const Card* card, bool highlight, bool selected) {
    Rectangle rect = {(float)x, (float)y, (float)CARD_W, (float)CARD_H};
    Color bg = card_color(card);

    if (selected) {
        DrawRectangleRec(rect, GOLD);
        Rectangle inner = {(float)x + 3, (float)y + 3, (float)CARD_W - 6, (float)CARD_H - 6};
        DrawRectangleRec(inner, bg);
    } else {
        DrawRectangleRec(rect, bg);
    }

    if (highlight) {
        DrawRectangleLinesEx(rect, 3, BLUE);
    } else {
        DrawRectangleLinesEx(rect, 1, DARKGRAY);
    }

    // Card name (wrap if long)
    int font_size = (name.size() > 10) ? 10 : 12;
    DrawText(name.c_str(), x + 4, y + 4, font_size, BLACK);

    // Cost in bottom-left
    if (card) {
        std::string cost_str = "$" + std::to_string(card->cost);
        DrawText(cost_str.c_str(), x + 4, y + CARD_H - 18, 14, DARKBROWN);
    }

    // Count in bottom-right
    if (count >= 0) {
        std::string count_str = "x" + std::to_string(count);
        int tw = MeasureText(count_str.c_str(), 12);
        DrawText(count_str.c_str(), x + CARD_W - tw - 4, y + CARD_H - 16, 12, DARKGRAY);
    }

    return rect;
}

static void draw_card_back(int x, int y) {
    Rectangle rect = {(float)x, (float)y, (float)CARD_W, (float)CARD_H};
    DrawRectangleRec(rect, {70, 70, 130, 255});
    DrawRectangleLinesEx(rect, 1, DARKGRAY);
    DrawText("?", x + CARD_W / 2 - 5, y + CARD_H / 2 - 10, 20, WHITE);
}

// ─── Supply rendering ─────────────────────────────────────────────

struct SupplySlot {
    std::string pile_name;
    Rectangle rect;
};

static std::vector<SupplySlot> draw_supply(const GameState& state) {
    std::vector<SupplySlot> slots;

    // Row 1: Base treasures + victory + curse
    const char* row1[] = {"Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"};
    int x = 10;
    for (auto& name : row1) {
        const Card* card = CardRegistry::get(name);
        int cnt = state.get_supply().count(name);
        auto rect = draw_card(x, SUPPLY_Y, name, cnt, card, false, false);
        slots.push_back({name, rect});
        x += CARD_W + CARD_PAD;
    }

    // Row 2: Kingdom piles
    x = 10;
    for (const auto& pile_name : state.get_supply().all_pile_names()) {
        if (pile_name == "Copper" || pile_name == "Silver" || pile_name == "Gold" ||
            pile_name == "Estate" || pile_name == "Duchy" || pile_name == "Province" ||
            pile_name == "Curse") continue;

        const Card* card = CardRegistry::get(pile_name);
        int cnt = state.get_supply().count(pile_name);
        auto rect = draw_card(x, SUPPLY_Y + SUPPLY_ROW_H, pile_name, cnt, card, false, false);
        slots.push_back({pile_name, rect});
        x += CARD_W + CARD_PAD;
    }

    return slots;
}

// ─── Hand rendering ───────────────────────────────────────────────

struct HandSlot {
    int hand_index;
    int option_index;  // index into DecisionPoint::options, or -1
    Rectangle rect;
};

static std::vector<HandSlot> draw_hand(const GameState& state, int pid,
                                        const std::string& label, int y,
                                        const std::vector<int>& selected_indices,
                                        bool clickable) {
    std::vector<HandSlot> slots;
    const auto& hand = state.get_player(pid).get_hand();

    DrawText(label.c_str(), 10, y, 16, WHITE);
    int x = 10;
    int card_y = y + 20;

    for (int i = 0; i < static_cast<int>(hand.size()); i++) {
        const Card* card = state.card_def(hand[i]);
        std::string name = card ? card->name : "???";
        bool is_selected = std::find(selected_indices.begin(), selected_indices.end(), i)
                           != selected_indices.end();
        bool hover = false;
        if (clickable) {
            Rectangle r = {(float)x, (float)card_y, (float)CARD_W, (float)CARD_H};
            hover = CheckCollisionPointRec(GetMousePosition(), r);
        }
        auto rect = draw_card(x, card_y, name, -1, card, hover, is_selected);
        slots.push_back({i, -1, rect});
        x += CARD_W + CARD_PAD;
    }

    return slots;
}

// ─── In-play rendering ───────────────────────────────────────────

static void draw_in_play(const GameState& state, int pid,
                          const std::string& label, int x_start, int y) {
    const auto& in_play = state.get_player(pid).get_in_play();
    if (in_play.empty()) return;

    DrawText(label.c_str(), x_start, y, 14, WHITE);
    int x = x_start;
    int card_y = y + 16;
    for (int cid : in_play) {
        const Card* card = state.card_def(cid);
        std::string name = card ? card->name : "???";
        draw_card(x, card_y, name, -1, card, false, false);
        x += CARD_W + CARD_PAD;
    }
}

// ─── Decision panel ───────────────────────────────────────────────

struct ButtonSlot {
    int option_index;
    Rectangle rect;
};

static std::vector<ButtonSlot> draw_decision_panel(const DecisionPoint& dp,
                                                     const GameState& state,
                                                     const std::vector<int>& selected) {
    std::vector<ButtonSlot> buttons;

    int y = DECISION_Y;
    int x = 10;

    // Phase label
    std::string phase_label;
    switch (dp.type) {
        case DecisionType::PLAY_ACTION:   phase_label = "ACTION PHASE"; break;
        case DecisionType::PLAY_TREASURE: phase_label = "TREASURE PHASE"; break;
        case DecisionType::BUY_CARD:
            phase_label = "BUY PHASE ($" + std::to_string(state.coins()) +
                          ", " + std::to_string(state.buys()) + " buys)";
            break;
        case DecisionType::CHOOSE_CARDS_IN_HAND:
        case DecisionType::CHOOSE_OPTION:
        case DecisionType::REACT:
            phase_label = dp.source_card.empty() ? "CHOOSE" : dp.source_card;
            break;
        default: phase_label = "DECISION"; break;
    }

    DrawText(phase_label.c_str(), x, y, 18, YELLOW);
    y += 24;

    // Sub-decision prompt
    if (dp.type == DecisionType::CHOOSE_CARDS_IN_HAND ||
        dp.type == DecisionType::CHOOSE_OPTION ||
        dp.type == DecisionType::REACT) {
        std::string prompt;
        switch (dp.sub_choice_type) {
            case ChoiceType::DISCARD:  prompt = "Choose card(s) to DISCARD"; break;
            case ChoiceType::TRASH:    prompt = "Choose card(s) to TRASH"; break;
            case ChoiceType::GAIN:     prompt = "Choose a card to GAIN"; break;
            case ChoiceType::TOPDECK:  prompt = "Put on top of deck"; break;
            case ChoiceType::YES_NO:   prompt = "Choose yes or no"; break;
            case ChoiceType::PLAY_CARD: prompt = "Choose a card to PLAY"; break;
            default: prompt = "Make a choice"; break;
        }
        if (dp.min_choices > 1 || dp.max_choices > 1) {
            prompt += " (pick " + std::to_string(dp.min_choices);
            if (dp.min_choices != dp.max_choices)
                prompt += "-" + std::to_string(dp.max_choices);
            prompt += ")";
        }
        DrawText(prompt.c_str(), x, y, 14, RAYWHITE);
        y += 20;
    }

    // Option buttons
    x = 10;
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        const auto& opt = dp.options[i];

        std::string label = opt.label.empty() ? opt.card_name : opt.label;
        if (label.empty()) label = "Option " + std::to_string(i);

        int btn_w = MeasureText(label.c_str(), 14) + 20;
        if (btn_w < 80) btn_w = 80;
        int btn_h = 28;

        // Wrap to next line if needed
        if (x + btn_w > LOG_X - 20) {
            x = 10;
            y += btn_h + 4;
        }

        Rectangle rect = {(float)x, (float)y, (float)btn_w, (float)btn_h};
        bool hover = CheckCollisionPointRec(GetMousePosition(), rect);
        bool is_selected = std::find(selected.begin(), selected.end(), i)
                           != selected.end();

        Color bg = opt.is_pass ? Color{100, 100, 100, 255} : Color{60, 80, 120, 255};
        if (is_selected) bg = GOLD;
        else if (hover) bg = Color{80, 110, 160, 255};

        DrawRectangleRec(rect, bg);
        DrawRectangleLinesEx(rect, 1, DARKGRAY);
        Color text_col = is_selected ? BLACK : WHITE;
        DrawText(label.c_str(), x + 10, y + 7, 14, text_col);

        buttons.push_back({i, rect});
        x += btn_w + 6;
    }

    return buttons;
}

// ─── Log panel ────────────────────────────────────────────────────

static void draw_log(const std::vector<std::string>& messages) {
    DrawRectangle(LOG_X, PLAY_AREA_Y, LOG_W, SCREEN_H - PLAY_AREA_Y - 10, {30, 30, 30, 255});
    DrawRectangleLines(LOG_X, PLAY_AREA_Y, LOG_W, SCREEN_H - PLAY_AREA_Y - 10, DARKGRAY);
    DrawText("Event Log", LOG_X + 8, PLAY_AREA_Y + 4, 14, GRAY);

    int y = PLAY_AREA_Y + 22;
    int max_lines = (SCREEN_H - PLAY_AREA_Y - 40) / 14;
    int start = std::max(0, static_cast<int>(messages.size()) - max_lines);

    for (int i = start; i < static_cast<int>(messages.size()); i++) {
        const auto& msg = messages[i];
        // Truncate long lines
        std::string display = msg.substr(0, LOG_W / 7);
        DrawText(display.c_str(), LOG_X + 8, y, 12, RAYWHITE);
        y += 14;
        if (y > SCREEN_H - 20) break;
    }
}

// ─── Info bar ─────────────────────────────────────────────────────

static void draw_info_bar(const GameState& state) {
    DrawRectangle(0, INFO_Y, LOG_X - 10, INFO_H, {40, 40, 40, 255});
    std::string info = "Turn " + std::to_string(state.turn_number()) +
                       "  |  Player " + std::to_string(state.current_player_id()) + "'s turn" +
                       "  |  Actions: " + std::to_string(state.actions()) +
                       "  Buys: " + std::to_string(state.buys()) +
                       "  Coins: $" + std::to_string(state.coins());
    DrawText(info.c_str(), 10, INFO_Y + 8, 16, RAYWHITE);
}

// ─── Game over screen ─────────────────────────────────────────────

static void draw_game_over(const GameResult& result) {
    DrawRectangle(SCREEN_W / 2 - 200, SCREEN_H / 2 - 100, 400, 200, {20, 20, 20, 240});
    DrawRectangleLines(SCREEN_W / 2 - 200, SCREEN_H / 2 - 100, 400, 200, GOLD);

    DrawText("GAME OVER", SCREEN_W / 2 - 60, SCREEN_H / 2 - 80, 24, GOLD);

    std::string s1 = "Player 1: " + std::to_string(result.scores[0]) + " VP";
    std::string s2 = "Player 2: " + std::to_string(result.scores[1]) + " VP";
    DrawText(s1.c_str(), SCREEN_W / 2 - 80, SCREEN_H / 2 - 30, 18, WHITE);
    DrawText(s2.c_str(), SCREEN_W / 2 - 80, SCREEN_H / 2 - 5, 18, WHITE);

    std::string winner_text;
    if (result.winner == 0) winner_text = "Player 1 wins!";
    else if (result.winner == 1) winner_text = "Player 2 wins!";
    else winner_text = "It's a tie!";
    int tw = MeasureText(winner_text.c_str(), 22);
    DrawText(winner_text.c_str(), SCREEN_W / 2 - tw / 2, SCREEN_H / 2 + 35, 22, YELLOW);

    DrawText("Press ESC to quit", SCREEN_W / 2 - 70, SCREEN_H / 2 + 75, 14, GRAY);
}

// ─── Main ─────────────────────────────────────────────────────────

int main() {
    // --- Setup ---
    BaseCards::register_all();
    Level1Cards::register_all();

    std::vector<std::string> kingdom = {
        "Smithy", "Village", "Market", "Laboratory", "Festival",
        "Cellar", "Chapel", "Workshop", "Moat", "Militia"
    };

    GuiAgent player1(0);
    GuiAgent player2(1);
    GameLog game_log;
    GameResult final_result = {};
    std::atomic<bool> game_over{false};

    GameRunner runner(2, kingdom);
    runner.set_observer([&game_log](const std::string& msg) {
        game_log.push(msg);
    });

    // Run game on background thread
    std::thread game_thread([&]() {
        auto result = runner.run({&player1, &player2});
        final_result = result;
        game_over.store(true);
    });

    // --- Raylib window ---
    InitWindow(SCREEN_W, SCREEN_H, "Dominion - Local 2P");
    SetTargetFPS(60);

    std::vector<int> selected_options;  // for multi-select decisions

    while (!WindowShouldClose()) {
        // --- Input handling ---
        bool clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        Vector2 mouse = GetMousePosition();

        // Find which agent has a pending decision
        GuiAgent* active_agent = nullptr;
        if (player1.has_pending()) active_agent = &player1;
        else if (player2.has_pending()) active_agent = &player2;

        // --- Drawing ---
        BeginDrawing();
        ClearBackground({25, 25, 35, 255});

        if (game_over.load()) {
            // Still draw final state
            const auto& state = final_result;
            draw_game_over(final_result);

            // Draw log
            auto log_msgs = game_log.snapshot();
            draw_log(log_msgs);
        } else if (active_agent) {
            const auto& pending = active_agent->get_pending();
            const auto& state = *pending.state_snapshot;
            const auto& dp = pending.dp;

            // Supply (clickable during BUY/GAIN)
            auto supply_slots = draw_supply(state);

            // Info bar
            draw_info_bar(state);

            // In-play area
            draw_in_play(state, 0, "Player 1 in play:", 10, PLAY_AREA_Y);
            draw_in_play(state, 1, "Player 2 in play:", 10, PLAY_AREA_Y + 70);

            // Decision panel
            auto buttons = draw_decision_panel(dp, state, selected_options);

            // Hands
            bool hand_clickable = (dp.type == DecisionType::PLAY_ACTION ||
                                    dp.type == DecisionType::PLAY_TREASURE ||
                                    dp.type == DecisionType::CHOOSE_CARDS_IN_HAND);

            std::string p1_label = (dp.player_id == 0) ? ">> Player 1" : "Player 1";
            std::string p2_label = (dp.player_id == 1) ? ">> Player 2" : "Player 2";

            auto p1_hand = draw_hand(state, 0, p1_label, HAND_Y, selected_options,
                                      hand_clickable && dp.player_id == 0);
            auto p2_hand = draw_hand(state, 1, p2_label, HAND_Y + CARD_H + 30, selected_options,
                                      hand_clickable && dp.player_id == 1);

            // Log
            auto log_msgs = game_log.snapshot();
            draw_log(log_msgs);

            // Multi-select: show confirm button when enough selected
            bool is_multi = (dp.max_choices > 1);
            int num_selected = static_cast<int>(selected_options.size());

            if (is_multi && num_selected >= dp.min_choices) {
                Rectangle confirm_rect = {(float)(LOG_X - 120), (float)(DECISION_Y + DECISION_H - 35), 100, 30};
                bool confirm_hover = CheckCollisionPointRec(mouse, confirm_rect);
                DrawRectangleRec(confirm_rect, confirm_hover ? GREEN : DARKGREEN);
                DrawRectangleLinesEx(confirm_rect, 1, WHITE);
                std::string confirm_label = "Confirm (" + std::to_string(num_selected) + ")";
                DrawText(confirm_label.c_str(), (int)confirm_rect.x + 8, (int)confirm_rect.y + 8, 14, WHITE);

                if (clicked && confirm_hover) {
                    active_agent->submit_answer(selected_options);
                    selected_options.clear();
                    clicked = false;
                }
            }

            // --- Click handling ---
            if (clicked) {
                bool handled = false;

                // Check button clicks
                if (!handled) {
                    for (const auto& btn : buttons) {
                        if (CheckCollisionPointRec(mouse, btn.rect)) {
                            if (is_multi) {
                                auto it = std::find(selected_options.begin(),
                                                   selected_options.end(), btn.option_index);
                                if (it != selected_options.end()) {
                                    selected_options.erase(it);
                                } else if (num_selected < dp.max_choices) {
                                    selected_options.push_back(btn.option_index);
                                }
                            } else {
                                active_agent->submit_answer({btn.option_index});
                                selected_options.clear();
                            }
                            handled = true;
                            break;
                        }
                    }
                }

                // Check supply clicks during BUY_CARD or GAIN
                if (!handled && (dp.type == DecisionType::BUY_CARD ||
                                  dp.sub_choice_type == ChoiceType::GAIN)) {
                    for (const auto& slot : supply_slots) {
                        if (CheckCollisionPointRec(mouse, slot.rect)) {
                            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                                if (dp.options[i].card_name == slot.pile_name) {
                                    active_agent->submit_answer({i});
                                    selected_options.clear();
                                    handled = true;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }

                // Check hand card clicks
                if (!handled && hand_clickable) {
                    auto& hand_slots = (dp.player_id == 0) ? p1_hand : p2_hand;
                    for (const auto& slot : hand_slots) {
                        if (CheckCollisionPointRec(mouse, slot.rect)) {
                            const auto& hand = state.get_player(dp.player_id).get_hand();
                            for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
                                if (slot.hand_index < static_cast<int>(hand.size()) &&
                                    dp.options[i].card_id == hand[slot.hand_index]) {
                                    if (is_multi) {
                                        auto it = std::find(selected_options.begin(),
                                                           selected_options.end(), i);
                                        if (it != selected_options.end()) {
                                            selected_options.erase(it);
                                        } else if (num_selected < dp.max_choices) {
                                            selected_options.push_back(i);
                                        }
                                    } else {
                                        active_agent->submit_answer({i});
                                        selected_options.clear();
                                    }
                                    handled = true;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            // Waiting for game thread (between decisions)
            DrawText("Waiting...", SCREEN_W / 2 - 40, SCREEN_H / 2, 20, GRAY);
            auto log_msgs = game_log.snapshot();
            draw_log(log_msgs);
        }

        EndDrawing();
    }

    // Clean shutdown
    player1.cancel();
    player2.cancel();
    if (game_thread.joinable()) game_thread.join();
    CloseWindow();

    return 0;
}
