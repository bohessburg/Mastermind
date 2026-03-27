#include "gui_agent.h"
#include "../engine/game_runner.h"
#include "../game/card.h"

#include "raylib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
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
static constexpr int INFO_H = 50;
static constexpr int PLAY_AREA_Y = INFO_Y + INFO_H + 10;
static constexpr int PLAY_AREA_H = 140;
static constexpr int DECISION_Y = PLAY_AREA_Y + PLAY_AREA_H + 10;
static constexpr int DECISION_H = 120;
static constexpr int HAND_Y = DECISION_Y + DECISION_H + 10;

static constexpr int LOG_X = 950;
static constexpr int LOG_W = SCREEN_W - LOG_X - 10;

// ─── All available kingdom cards ─────────────────────────────────

static const std::vector<std::string> ALL_KINGDOM = {
    // Level 1 (Base Set)
    "Cellar", "Chapel", "Moat", "Harbinger", "Merchant",
    "Vassal", "Village", "Workshop", "Bureaucrat", "Gardens",
    "Laboratory", "Market", "Militia", "Moneylender", "Poacher",
    "Remodel", "Smithy", "Throne Room", "Bandit", "Council Room",
    "Festival", "Library", "Mine", "Sentry", "Witch", "Artisan",
    // Level 2
    "Worker's Village", "Courtyard", "Hamlet", "Lookout", "Swindler", "Scholar"
};

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

    int font_size = (name.size() > 10) ? 10 : 12;
    DrawText(name.c_str(), x + 4, y + 4, font_size, BLACK);

    if (card) {
        std::string cost_str = "$" + std::to_string(card->cost);
        DrawText(cost_str.c_str(), x + 4, y + CARD_H - 18, 14, DARKBROWN);
    }

    if (count >= 0) {
        std::string count_str = "x" + std::to_string(count);
        int tw = MeasureText(count_str.c_str(), 12);
        DrawText(count_str.c_str(), x + CARD_W - tw - 4, y + CARD_H - 16, 12, DARKGRAY);
    }

    return rect;
}

// ─── Supply rendering ─────────────────────────────────────────────

struct SupplySlot {
    std::string pile_name;
    Rectangle rect;
};

static std::vector<SupplySlot> draw_supply(const GameState& state) {
    std::vector<SupplySlot> slots;

    const char* row1[] = {"Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"};
    int x = 10;
    for (auto& name : row1) {
        const Card* card = CardRegistry::get(name);
        int cnt = state.get_supply().count(name);
        auto rect = draw_card(x, SUPPLY_Y, name, cnt, card, false, false);
        slots.push_back({name, rect});
        x += CARD_W + CARD_PAD;
    }

    // Collect kingdom piles and sort by cost then name
    struct KingdomPile { std::string name; int cost; int count; };
    std::vector<KingdomPile> kingdom_piles;
    for (const auto& pile : state.get_supply().piles()) {
        const auto& pile_name = pile.pile_name;
        if (pile_name == "Copper" || pile_name == "Silver" || pile_name == "Gold" ||
            pile_name == "Estate" || pile_name == "Duchy" || pile_name == "Province" ||
            pile_name == "Curse") continue;
        const Card* card = CardRegistry::get(pile_name);
        int cost = card ? card->cost : 0;
        kingdom_piles.push_back({pile_name, cost, static_cast<int>(pile.card_ids.size())});
    }
    std::sort(kingdom_piles.begin(), kingdom_piles.end(), [](const KingdomPile& a, const KingdomPile& b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.name < b.name;
    });

    x = 10;
    for (const auto& kp : kingdom_piles) {
        const Card* card = CardRegistry::get(kp.name);
        auto rect = draw_card(x, SUPPLY_Y + SUPPLY_ROW_H, kp.name, kp.count, card, false, false);
        slots.push_back({kp.name, rect});
        x += CARD_W + CARD_PAD;
    }

    return slots;
}

// ─── Hand rendering ───────────────────────────────────────────────

struct HandSlot {
    int hand_index;
    int option_index;
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
            case ChoiceType::MULTI_FATE:
                prompt = "Choose fate";
                if (!dp.source_card.empty()) prompt += " for " + dp.source_card;
                break;
            case ChoiceType::ORDER:    prompt = "Choose which goes on top"; break;
            case ChoiceType::SELECT_FROM_DISCARD: prompt = "Choose from discard"; break;
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

    x = 10;
    for (int i = 0; i < static_cast<int>(dp.options.size()); i++) {
        const auto& opt = dp.options[i];

        std::string label = opt.label.empty() ? opt.card_name : opt.label;
        if (label.empty() && opt.card_id >= 0) {
            const Card* c = state.card_def(opt.card_id);
            if (c) label = c->name + " ($" + std::to_string(c->cost) + ")";
            else label = state.card_name(opt.card_id);
        }
        if (label.empty()) label = "Option " + std::to_string(i);

        int btn_w = MeasureText(label.c_str(), 14) + 20;
        if (btn_w < 80) btn_w = 80;
        int btn_h = 28;

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
        std::string display = msg.substr(0, LOG_W / 7);
        DrawText(display.c_str(), LOG_X + 8, y, 12, RAYWHITE);
        y += 14;
        if (y > SCREEN_H - 20) break;
    }
}

// ─── Info bar (with discard tops + trash) ────────────────────────

static void draw_info_bar(const GameState& state) {
    DrawRectangle(0, INFO_Y, LOG_X - 10, INFO_H, {40, 40, 40, 255});

    // Turn info line
    std::string info = "Turn " + std::to_string(state.turn_number()) +
                       "  |  Player " + std::to_string(state.current_player_id() + 1) + "'s turn" +
                       "  |  Actions: " + std::to_string(state.actions()) +
                       "  Buys: " + std::to_string(state.buys()) +
                       "  Coins: $" + std::to_string(state.coins());
    DrawText(info.c_str(), 10, INFO_Y + 4, 14, RAYWHITE);

    // Player deck/discard info line
    auto discard_top = [&](int pid) -> std::string {
        const auto& d = state.get_player(pid).get_discard();
        if (d.empty()) return "empty";
        return state.card_name(d.back());
    };

    std::string p1_info = "P1: Deck(" + std::to_string(state.get_player(0).deck_size()) +
                          ") Discard(" + std::to_string(state.get_player(0).discard_size()) +
                          ", top: " + discard_top(0) + ")";
    std::string p2_info = "P2: Deck(" + std::to_string(state.get_player(1).deck_size()) +
                          ") Discard(" + std::to_string(state.get_player(1).discard_size()) +
                          ", top: " + discard_top(1) + ")";

    std::string trash_info = "Trash(" + std::to_string(state.get_trash().size()) + ")";
    if (!state.get_trash().empty()) {
        trash_info += ":";
        int shown = 0;
        for (int cid : state.get_trash()) {
            if (shown >= 8) { trash_info += " ..."; break; }
            trash_info += " " + state.card_name(cid);
            shown++;
        }
    }

    DrawText(p1_info.c_str(), 10, INFO_Y + 22, 12, {180, 200, 255, 255});
    DrawText(p2_info.c_str(), 350, INFO_Y + 22, 12, {255, 200, 180, 255});
    DrawText(trash_info.c_str(), 10, INFO_Y + 36, 12, {200, 200, 200, 255});
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

// ─── Kingdom selection screen ────────────────────────────────────

enum class SetupMode { CHOOSING, PICKING, MODE_SELECT, READY };
enum class GameMode { LOCAL_2P, VS_ENGINE };

struct SetupResult {
    std::vector<std::string> kingdom;
    GameMode game_mode;
};

static SetupResult run_setup() {
    SetupMode mode = SetupMode::CHOOSING;
    GameMode game_mode = GameMode::LOCAL_2P;
    std::vector<bool> selected(ALL_KINGDOM.size(), false);
    std::vector<std::string> kingdom;

    while (!WindowShouldClose()) {
        bool clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        Vector2 mouse = GetMousePosition();

        BeginDrawing();
        ClearBackground({25, 25, 35, 255});

        if (mode == SetupMode::CHOOSING) {
            DrawText("DOMINION", SCREEN_W / 2 - 80, 40, 32, GOLD);
            DrawText("Kingdom Setup", SCREEN_W / 2 - 70, 80, 20, WHITE);

            // Random button
            Rectangle rand_btn = {(float)(SCREEN_W / 2 - 150), 140, 140, 40};
            bool rand_hover = CheckCollisionPointRec(mouse, rand_btn);
            DrawRectangleRec(rand_btn, rand_hover ? Color{80, 110, 160, 255} : Color{60, 80, 120, 255});
            DrawRectangleLinesEx(rand_btn, 1, WHITE);
            DrawText("Random 10", (int)rand_btn.x + 20, (int)rand_btn.y + 12, 16, WHITE);

            // Pick button
            Rectangle pick_btn = {(float)(SCREEN_W / 2 + 10), 140, 140, 40};
            bool pick_hover = CheckCollisionPointRec(mouse, pick_btn);
            DrawRectangleRec(pick_btn, pick_hover ? Color{80, 110, 160, 255} : Color{60, 80, 120, 255});
            DrawRectangleLinesEx(pick_btn, 1, WHITE);
            DrawText("Pick Cards", (int)pick_btn.x + 20, (int)pick_btn.y + 12, 16, WHITE);

            if (clicked && rand_hover) {
                auto seed = static_cast<unsigned>(
                    std::chrono::steady_clock::now().time_since_epoch().count());
                std::mt19937 rng(seed);
                auto shuffled = ALL_KINGDOM;
                std::shuffle(shuffled.begin(), shuffled.end(), rng);
                kingdom.assign(shuffled.begin(), shuffled.begin() + 10);
                std::sort(kingdom.begin(), kingdom.end());
                mode = SetupMode::MODE_SELECT;
            }
            if (clicked && pick_hover) {
                mode = SetupMode::PICKING;
            }

        } else if (mode == SetupMode::PICKING) {
            int num_selected = 0;
            for (bool s : selected) if (s) num_selected++;

            DrawText("Pick 10 Kingdom Cards", SCREEN_W / 2 - 100, 20, 20, WHITE);
            std::string count_str = std::to_string(num_selected) + " / 10 selected";
            DrawText(count_str.c_str(), SCREEN_W / 2 - 50, 48, 16,
                     num_selected == 10 ? GREEN : YELLOW);

            // Draw card grid
            int cols = 8;
            int start_x = 30;
            int start_y = 80;
            int cell_w = CARD_W + CARD_PAD;
            int cell_h = CARD_H + CARD_PAD + 4;

            for (int i = 0; i < static_cast<int>(ALL_KINGDOM.size()); i++) {
                int col = i % cols;
                int row = i / cols;
                int x = start_x + col * (cell_w + 4);
                int y = start_y + row * cell_h;

                const Card* card = CardRegistry::get(ALL_KINGDOM[i]);
                bool hover_card = false;
                Rectangle r = {(float)x, (float)y, (float)CARD_W, (float)CARD_H};
                hover_card = CheckCollisionPointRec(mouse, r);

                draw_card(x, y, ALL_KINGDOM[i], -1, card, hover_card, selected[i]);

                if (clicked && hover_card) {
                    if (selected[i]) {
                        selected[i] = false;
                    } else if (num_selected < 10) {
                        selected[i] = true;
                    }
                }
            }

            // Start button (only when 10 selected)
            if (num_selected >= 1) {
                Rectangle start_btn = {(float)(SCREEN_W / 2 - 60), (float)(SCREEN_H - 60), 120, 40};
                bool start_hover = CheckCollisionPointRec(mouse, start_btn);
                Color btn_col = num_selected >= 10 ? (start_hover ? GREEN : DARKGREEN) :
                                                      (start_hover ? YELLOW : Color{150, 150, 0, 255});
                DrawRectangleRec(start_btn, btn_col);
                DrawRectangleLinesEx(start_btn, 1, WHITE);
                std::string btn_text = num_selected >= 10 ? "Start Game" : "Start (" + std::to_string(num_selected) + ")";
                DrawText(btn_text.c_str(), (int)start_btn.x + 10, (int)start_btn.y + 12, 16, WHITE);

                if (clicked && start_hover) {
                    for (int i = 0; i < static_cast<int>(ALL_KINGDOM.size()); i++) {
                        if (selected[i]) kingdom.push_back(ALL_KINGDOM[i]);
                    }
                    std::sort(kingdom.begin(), kingdom.end());
                    mode = SetupMode::MODE_SELECT;
                }
            }

        } else if (mode == SetupMode::MODE_SELECT) {
            DrawText("DOMINION", SCREEN_W / 2 - 80, 40, 32, GOLD);
            DrawText("Game Mode", SCREEN_W / 2 - 55, 80, 20, WHITE);

            // Show selected kingdom
            std::string k_str = "Kingdom: ";
            for (const auto& k : kingdom) k_str += k + "  ";
            DrawText(k_str.c_str(), 30, 120, 12, GRAY);

            // 2-Player Local button
            Rectangle local_btn = {(float)(SCREEN_W / 2 - 150), 160, 300, 50};
            bool local_hover = CheckCollisionPointRec(mouse, local_btn);
            DrawRectangleRec(local_btn, local_hover ? Color{80, 110, 160, 255} : Color{60, 80, 120, 255});
            DrawRectangleLinesEx(local_btn, 1, WHITE);
            DrawText("2-Player Local (pass & play)", (int)local_btn.x + 30, (int)local_btn.y + 16, 18, WHITE);

            // vs Engine Bot button
            Rectangle engine_btn = {(float)(SCREEN_W / 2 - 150), 230, 300, 50};
            bool engine_hover = CheckCollisionPointRec(mouse, engine_btn);
            DrawRectangleRec(engine_btn, engine_hover ? Color{110, 80, 60, 255} : Color{80, 60, 40, 255});
            DrawRectangleLinesEx(engine_btn, 1, WHITE);
            DrawText("You vs Engine Bot", (int)engine_btn.x + 60, (int)engine_btn.y + 16, 18, WHITE);

            if (clicked && local_hover) {
                game_mode = GameMode::LOCAL_2P;
                mode = SetupMode::READY;
            }
            if (clicked && engine_hover) {
                game_mode = GameMode::VS_ENGINE;
                mode = SetupMode::READY;
            }
        }

        EndDrawing();

        if (mode == SetupMode::READY) break;
    }

    return {kingdom, game_mode};
}

// ─── Main ─────────────────────────────────────────────────────────

int main() {
    // --- Register cards ---
    BaseCards::register_all();
    Level1Cards::register_all();
    Level2Cards::register_all();

    // --- Raylib window ---
    InitWindow(SCREEN_W, SCREEN_H, "Dominion");
    SetTargetFPS(60);

    // --- Setup ---
    auto setup = run_setup();
    if (setup.kingdom.empty() || WindowShouldClose()) {
        CloseWindow();
        return 0;
    }

    // --- Game setup ---
    GuiAgent gui_player1(0);
    GuiAgent gui_player2(1);
    EngineBot engine_bot;
    GameLog game_log;
    GameResult final_result = {};
    std::atomic<bool> game_over{false};

    // Wire agents based on game mode
    std::vector<Agent*> agents;
    if (setup.game_mode == GameMode::VS_ENGINE) {
        agents = {&gui_player1, &engine_bot};
    } else {
        agents = {&gui_player1, &gui_player2};
    }

    GameRunner runner(2, setup.kingdom);
    runner.set_observer([&game_log](const std::string& msg) {
        game_log.push(msg);
    });

    std::thread game_thread([&]() {
        auto result = runner.run(agents);
        final_result = result;
        game_over.store(true);
    });

    std::vector<int> selected_options;

    while (!WindowShouldClose()) {
        bool clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        Vector2 mouse = GetMousePosition();

        GuiAgent* active_agent = nullptr;
        if (gui_player1.has_pending()) active_agent = &gui_player1;
        else if (setup.game_mode == GameMode::LOCAL_2P && gui_player2.has_pending())
            active_agent = &gui_player2;

        BeginDrawing();
        ClearBackground({25, 25, 35, 255});

        if (game_over.load()) {
            draw_game_over(final_result);
            auto log_msgs = game_log.snapshot();
            draw_log(log_msgs);
        } else if (active_agent) {
            const auto& pending = active_agent->get_pending();
            const auto& state = *pending.state_snapshot;
            const auto& dp = pending.dp;

            auto supply_slots = draw_supply(state);
            draw_info_bar(state);

            draw_in_play(state, 0, "Player 1 in play:", 10, PLAY_AREA_Y);
            draw_in_play(state, 1, "Player 2 in play:", 10, PLAY_AREA_Y + 70);

            auto buttons = draw_decision_panel(dp, state, selected_options);

            bool hand_clickable = (dp.type == DecisionType::PLAY_ACTION ||
                                    dp.type == DecisionType::PLAY_TREASURE ||
                                    dp.type == DecisionType::CHOOSE_CARDS_IN_HAND);

            std::string p1_label = (dp.player_id == 0) ? ">> Player 1" : "Player 1";
            std::string p2_label = (dp.player_id == 1) ? ">> Player 2" : "Player 2";

            auto p1_hand = draw_hand(state, 0, p1_label, HAND_Y, selected_options,
                                      hand_clickable && dp.player_id == 0);
            auto p2_hand = draw_hand(state, 1, p2_label, HAND_Y + CARD_H + 30, selected_options,
                                      hand_clickable && dp.player_id == 1);

            auto log_msgs = game_log.snapshot();
            draw_log(log_msgs);

            // Multi-select confirm button
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

            // Click handling
            if (clicked) {
                bool handled = false;

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
            DrawText("Waiting...", SCREEN_W / 2 - 40, SCREEN_H / 2, 20, GRAY);
            auto log_msgs = game_log.snapshot();
            draw_log(log_msgs);
        }

        EndDrawing();
    }

    gui_player1.cancel();
    gui_player2.cancel();
    if (game_thread.joinable()) game_thread.join();
    CloseWindow();

    return 0;
}
