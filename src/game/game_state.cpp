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
    turns_taken_.resize(num_players, 0);
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
void GameState::set_phase(Phase phase) { phase_ = phase; }
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

int GameState::play_card_from_hand(int player_id, int hand_index, DecisionFn decide) {
    Player& player = get_player(player_id);
    int card_id = player.get_hand()[hand_index];
    player.play_from_hand(hand_index);

    const Card* card = card_def(card_id);
    if (card && card->on_play) {
        card->on_play(*this, player_id, decide);
    }
    return card_id;
}

void GameState::play_card_effect(int card_id, int player_id, DecisionFn decide) {
    const Card* card = card_def(card_id);
    if (card && card->on_play) {
        card->on_play(*this, player_id, decide);
    }
}

void GameState::resolve_attack(
    int attacker_id,
    std::function<void(GameState&, int target_id, DecisionFn)> attack_effect,
    DecisionFn decide)
{
    for (int i = 1; i < num_players(); i++) {
        int target_id = (attacker_id + i) % num_players();
        bool blocked = false;

        // Check for Reaction cards in target's hand.
        // Snapshot the hand size since reactions could theoretically change it.
        const auto& hand = get_player(target_id).get_hand();
        for (int h = 0; h < static_cast<int>(hand.size()); h++) {
            const Card* card = card_def(hand[h]);
            if (card && card->is_reaction() && card->on_react) {
                if (card->on_react(*this, target_id, attacker_id, decide)) {
                    blocked = true;
                    break;
                }
            }
        }

        if (!blocked) {
            attack_effect(*this, target_id, decide);
        }
    }
}

std::vector<std::string> GameState::gainable_piles(int max_cost) const {
    std::vector<std::string> result;
    for (const auto& pile_name : supply_.all_pile_names()) {
        if (supply_.is_pile_empty(pile_name)) continue;
        int top_id = supply_.top_card(pile_name);
        if (top_id == -1) continue;
        const Card* card = card_def(top_id);
        if (card && card->cost <= max_cost) {
            result.push_back(pile_name);
        }
    }
    return result;
}

std::vector<std::string> GameState::gainable_piles(int max_cost, CardType required_type) const {
    std::vector<std::string> result;
    for (const auto& pile_name : supply_.all_pile_names()) {
        if (supply_.is_pile_empty(pile_name)) continue;
        int top_id = supply_.top_card(pile_name);
        if (top_id == -1) continue;
        const Card* card = card_def(top_id);
        if (card && card->cost <= max_cost && has_type(card->types, required_type)) {
            result.push_back(pile_name);
        }
    }
    return result;
}

int GameState::effective_cost(const std::string& card_name_str) const {
    const Card* card = CardRegistry::get(card_name_str);
    if (!card) return 0;
    return card->cost;  // Future: apply cost reductions
}

int GameState::total_cards_owned(int player_id) const {
    return static_cast<int>(players_[player_id].all_cards().size());
}

// --- Turn lifecycle ---

void GameState::start_game() {
    turn_number_ = 1;
    current_player_ = 0;
    phase_ = Phase::ACTION;
    actions_ = 1;
    buys_ = 1;
    coins_ = 0;
    turn_flags_.clear();
    std::fill(turns_taken_.begin(), turns_taken_.end(), 0);
}

void GameState::start_turn() {
    phase_ = Phase::ACTION;
    actions_ = 1;
    buys_ = 1;
    coins_ = 0;
    turn_flags_.clear();
}

void GameState::advance_phase() {
    switch (phase_) {
        case Phase::ACTION:
            phase_ = Phase::TREASURE;
            break;
        case Phase::TREASURE:
            phase_ = Phase::BUY;
            break;
        case Phase::BUY:
            phase_ = Phase::CLEANUP;
            break;
        case Phase::NIGHT:
            phase_ = Phase::CLEANUP;
            break;
        case Phase::CLEANUP:
            players_[current_player_].cleanup();
            turns_taken_[current_player_]++;
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

        std::vector<int> owned = player.all_cards();

        for (int card_id : owned) {
            const Card* card = card_def(card_id);
            if (card) {
                if (card->vp_fn) {
                    scores[p] += card->vp_fn(*this, p);
                } else {
                    scores[p] += card->victory_points;
                }
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

    if (!tie) return best_player;

    // Tiebreaker: fewer turns wins
    int best_turns = turns_taken_[best_player];
    best_player = -1;
    tie = false;

    for (int i = 0; i < num_players(); i++) {
        if (scores[i] != best_score) continue;
        if (best_player == -1 || turns_taken_[i] < best_turns) {
            best_turns = turns_taken_[i];
            best_player = i;
            tie = false;
        } else if (turns_taken_[i] == best_turns) {
            tie = true;
        }
    }

    return tie ? -1 : best_player;
}

// --- Turn flags ---

int GameState::get_turn_flag(const std::string& key) const {
    auto it = turn_flags_.find(key);
    if (it == turn_flags_.end()) return 0;
    return it->second;
}

void GameState::set_turn_flag(const std::string& key, int value) {
    turn_flags_[key] = value;
}
