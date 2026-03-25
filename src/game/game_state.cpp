#include "game_state.h"

static const std::string UNKNOWN_CARD = "???";

GameState::GameState(int num_players)
    : current_player_(0)
    , phase_(Phase::ACTION)
    , actions_(1)
    , buys_(1)
    , coins_(0)
    , turn_number_(1)
    , next_card_id_(0)
{
    for (int i = 0; i < num_players; i++) {
        players_.emplace_back(i);
    }
}

// --- Card ID system ---

int GameState::create_card(const std::string& card_name) {
    int id = next_card_id_++;
    card_names_[id] = card_name;
    return id;
}

const std::string& GameState::card_name(int card_id) const {
    auto it = card_names_.find(card_id);
    if (it == card_names_.end()) return UNKNOWN_CARD;
    return it->second;
}

const Card* GameState::card_def(int card_id) const {
    auto it = card_names_.find(card_id);
    if (it == card_names_.end()) return nullptr;
    return CardRegistry::get(it->second);
}

// --- Players ---

int GameState::num_players() const {
    return static_cast<int>(players_.size());
}

Player& GameState::get_player(int player_id) {
    return players_[player_id];
}

const Player& GameState::get_player(int player_id) const {
    return players_[player_id];
}

int GameState::current_player_id() const {
    return current_player_;
}

// --- Supply ---

Supply& GameState::get_supply() { return supply_; }
const Supply& GameState::get_supply() const { return supply_; }

// --- Turn state ---

Phase GameState::current_phase() const { return phase_; }
int GameState::actions() const { return actions_; }
int GameState::buys() const { return buys_; }
int GameState::coins() const { return coins_; }
int GameState::turn_number() const { return turn_number_; }

void GameState::add_actions(int n) { actions_ += n; }
void GameState::add_buys(int n) { buys_ += n; }
void GameState::add_coins(int n) { coins_ += n; }

// --- Game actions ---

int GameState::gain_card(int player_id, const std::string& pile_name) {
    int card_id = supply_.gain_from(pile_name);
    if (card_id == -1) return -1;
    players_[player_id].add_to_discard(card_id);
    return card_id;
}

int GameState::gain_card_to_hand(int player_id, const std::string& pile_name) {
    int card_id = supply_.gain_from(pile_name);
    if (card_id == -1) return -1;
    players_[player_id].add_to_hand(card_id);
    return card_id;
}

int GameState::gain_card_to_deck_top(int player_id, const std::string& pile_name) {
    int card_id = supply_.gain_from(pile_name);
    if (card_id == -1) return -1;
    players_[player_id].add_to_deck_top(card_id);
    return card_id;
}

void GameState::trash_card(int card_id) {
    trash_.push_back(card_id);
}

const std::vector<int>& GameState::get_trash() const {
    return trash_;
}

// --- Turn lifecycle ---

void GameState::start_game() {
    turn_number_ = 1;
    current_player_ = 0;
    phase_ = Phase::ACTION;
    actions_ = 1;
    buys_ = 1;
    coins_ = 0;
}

void GameState::start_turn() {
    phase_ = Phase::ACTION;
    actions_ = 1;
    buys_ = 1;
    coins_ = 0;
}

void GameState::advance_phase() {
    switch (phase_) {
        case Phase::ACTION:
            phase_ = Phase::BUY;
            break;
        case Phase::BUY:
            phase_ = Phase::CLEANUP;
            break;
        case Phase::CLEANUP:
            players_[current_player_].cleanup();
            // Move to next player
            current_player_ = (current_player_ + 1) % num_players();
            turn_number_++;
            if (supply_.is_game_over()) {
                phase_ = Phase::GAME_OVER;
            } else {
                start_turn();
            }
            break;
        case Phase::GAME_OVER:
            break;
    }
}

// --- Game end ---

bool GameState::is_game_over() const {
    return phase_ == Phase::GAME_OVER || supply_.is_game_over();
}

std::vector<int> GameState::calculate_scores() const {
    std::vector<int> scores(num_players(), 0);
    for (int p = 0; p < num_players(); p++) {
        const Player& player = players_[p];

        // Collect all cards the player owns (hand, deck, discard, in-play, set-aside, mats)
        std::vector<int> all_cards;
        auto append = [&all_cards](const std::vector<int>& cards) {
            all_cards.insert(all_cards.end(), cards.begin(), cards.end());
        };

        append(player.get_hand());
        append(player.get_deck());
        append(player.get_discard());
        append(player.get_in_play());

        for (const auto& [source, cards] : player.get_set_aside()) {
            append(cards);
        }
        for (const auto& [mat_name, cards] : player.get_mats()) {
            append(cards);
        }

        // Sum up victory points from all owned cards
        for (int card_id : all_cards) {
            const Card* card = card_def(card_id);
            if (card) {
                scores[p] += card->victory_points;
            }
        }
    }
    return scores;
}

int GameState::winner() const {
    std::vector<int> scores = calculate_scores();
    int best_score = scores[0];
    int best_player = 0;
    bool tie = false;

    for (int i = 1; i < num_players(); i++) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_player = i;
            tie = false;
        } else if (scores[i] == best_score) {
            tie = true;
        }
    }

    return tie ? -1 : best_player;
}
