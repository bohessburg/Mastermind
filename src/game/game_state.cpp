#include "game_state.h"

static const std::string UNKNOWN_CARD = "???";

// Static well-known def IDs
int GameState::def_copper_   = -1;
int GameState::def_silver_   = -1;
int GameState::def_gold_     = -1;
int GameState::def_estate_   = -1;
int GameState::def_duchy_    = -1;
int GameState::def_province_ = -1;
int GameState::def_curse_    = -1;

GameState::GameState(int num_players)
    : current_player_(0)
    , phase_(Phase::ACTION)
    , actions_(1)
    , buys_(1)
    , coins_(0)
    , turn_number_(1)
    , actions_played_(0)
    , next_card_id_(0)
{
    for (int i = 0; i < num_players; i++) {
        players_.emplace_back(i);
    }
    turns_taken_.resize(num_players, 0);
    card_names_.reserve(256);  // typical game has ~200 cards
    card_defs_.reserve(256);
    card_def_ids_.reserve(256);
    turn_flags_.fill(0);
}

// --- Card ID system ---

int GameState::create_card(const std::string& card_name) {
    int id = next_card_id_++;
    const Card* def = CardRegistry::get(card_name);
    card_names_.push_back(card_name);
    card_defs_.push_back(def);
    card_def_ids_.push_back(def ? def->def_id : -1);
    return id;
}

const std::string& GameState::card_name(int card_id) const {
    if (card_id < 0 || card_id >= next_card_id_) return UNKNOWN_CARD;
    return card_names_[card_id];
}

const Card* GameState::card_def(int card_id) const {
    if (card_id < 0 || card_id >= next_card_id_) return nullptr;
    return card_defs_[card_id];
}

int GameState::card_def_id(int card_id) const {
    if (card_id < 0 || card_id >= next_card_id_) return -1;
    return card_def_ids_[card_id];
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
int GameState::actions_played() const { return actions_played_; }

void GameState::add_actions(int n) { actions_ += n; }
void GameState::add_buys(int n) { buys_ += n; }
void GameState::add_coins(int n) { coins_ += n; }

// --- Game actions (pile-index-based) ---

int GameState::gain_card(int player_id, int pile_index) {
    int card_id = supply_.gain_from_index(pile_index);
    if (card_id == -1) return -1;
    players_[player_id].add_to_discard(card_id);
    if (log_fn_) log("    P" + std::to_string(player_id + 1) + " gains " +
        supply_.pile_at(pile_index).pile_name + " to discard");
    return card_id;
}

int GameState::gain_card_to_hand(int player_id, int pile_index) {
    int card_id = supply_.gain_from_index(pile_index);
    if (card_id == -1) return -1;
    players_[player_id].add_to_hand(card_id);
    if (log_fn_) log("    P" + std::to_string(player_id + 1) + " gains " +
        supply_.pile_at(pile_index).pile_name + " to hand");
    return card_id;
}

int GameState::gain_card_to_deck_top(int player_id, int pile_index) {
    int card_id = supply_.gain_from_index(pile_index);
    if (card_id == -1) return -1;
    players_[player_id].add_to_deck_top(card_id);
    if (log_fn_) log("    P" + std::to_string(player_id + 1) + " gains " +
        supply_.pile_at(pile_index).pile_name + " to deck top");
    return card_id;
}

void GameState::trash_card(int card_id) {
    trash_.push_back(card_id);
    if (log_fn_) log("    Trashed " + card_name(card_id));
}

const std::vector<int>& GameState::get_trash() const {
    return trash_;
}

int GameState::play_card_from_hand(int player_id, int hand_index, DecisionFn decide) {
    Player& player = get_player(player_id);
    int card_id = player.get_hand()[hand_index];
    player.play_from_hand(hand_index);

    const Card* card = card_def(card_id);
    if (card && card->is_action()) {
        actions_played_++;
    }
    if (log_fn_) log("  P" + std::to_string(player_id + 1) + " plays " + card_name(card_id));
    if (card && card->on_play) {
        card->on_play(*this, player_id, decide);
    }
    return card_id;
}

void GameState::play_card_effect(int card_id, int player_id, DecisionFn decide) {
    const Card* card = card_def(card_id);
    if (card && card->is_action()) {
        actions_played_++;
    }
    if (log_fn_) log("    P" + std::to_string(player_id + 1) + " plays " + card_name(card_id) + " again");
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

        const auto& hand = get_player(target_id).get_hand();
        for (int h = 0; h < static_cast<int>(hand.size()); h++) {
            const Card* card = card_def(hand[h]);
            if (card && card->is_reaction() && card->on_react) {
                if (card->on_react(*this, target_id, attacker_id, decide)) {
                    if (log_fn_) log("    P" + std::to_string(target_id + 1) + " reveals " +
                        card->name + " — blocks the attack!");
                    blocked = true;
                    break;
                }
            }
        }

        if (!blocked) {
            if (log_fn_) log("    Attack hits P" + std::to_string(target_id + 1));
            attack_effect(*this, target_id, decide);
        }
    }
}

std::vector<int> GameState::gainable_piles(int max_cost) const {
    std::vector<int> result;
    const auto& piles = supply_.piles();
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i].card_ids.empty()) continue;
        const Card* card = card_def(piles[i].card_ids.back());
        if (card && card->cost <= max_cost) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<int> GameState::gainable_piles(int max_cost, CardType required_type) const {
    std::vector<int> result;
    const auto& piles = supply_.piles();
    for (int i = 0; i < static_cast<int>(piles.size()); i++) {
        if (piles[i].card_ids.empty()) continue;
        const Card* card = card_def(piles[i].card_ids.back());
        if (card && card->cost <= max_cost && has_type(card->types, required_type)) {
            result.push_back(i);
        }
    }
    return result;
}

int GameState::effective_cost(int pile_index) const {
    int top = supply_.top_card_index(pile_index);
    if (top == -1) return 0;
    const Card* card = card_def(top);
    return card ? card->cost : 0;
}

int GameState::total_cards_owned(int player_id) const {
    return players_[player_id].total_card_count();
}

// --- Well-known pile/def caching ---

void GameState::cache_well_known_piles() {
    pile_copper_   = supply_.pile_index_of("Copper");
    pile_silver_   = supply_.pile_index_of("Silver");
    pile_gold_     = supply_.pile_index_of("Gold");
    pile_estate_   = supply_.pile_index_of("Estate");
    pile_duchy_    = supply_.pile_index_of("Duchy");
    pile_province_ = supply_.pile_index_of("Province");
    pile_curse_    = supply_.pile_index_of("Curse");
}

void GameState::cache_well_known_def_ids() {
    def_copper_   = CardRegistry::def_id_of("Copper");
    def_silver_   = CardRegistry::def_id_of("Silver");
    def_gold_     = CardRegistry::def_id_of("Gold");
    def_estate_   = CardRegistry::def_id_of("Estate");
    def_duchy_    = CardRegistry::def_id_of("Duchy");
    def_province_ = CardRegistry::def_id_of("Province");
    def_curse_    = CardRegistry::def_id_of("Curse");
}

// --- Turn lifecycle ---

void GameState::start_game() {
    turn_number_ = 1;
    current_player_ = 0;
    phase_ = Phase::ACTION;
    actions_ = 1;
    buys_ = 1;
    coins_ = 0;
    turn_flags_.fill(0);
    std::fill(turns_taken_.begin(), turns_taken_.end(), 0);
}

void GameState::start_turn() {
    phase_ = Phase::ACTION;
    actions_ = 1;
    buys_ = 1;
    coins_ = 0;
    actions_played_ = 0;
    turn_flags_.fill(0);
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
        player.for_each_card([&](int card_id) {
            const Card* card = card_def(card_id);
            if (card) {
                if (card->vp_fn) {
                    scores[p] += card->vp_fn(*this, p);
                } else {
                    scores[p] += card->victory_points;
                }
            }
        });
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

int GameState::get_turn_flag(TurnFlag flag) const {
    return turn_flags_[static_cast<int>(flag)];
}

void GameState::set_turn_flag(TurnFlag flag, int value) {
    turn_flags_[static_cast<int>(flag)] = value;
}

void GameState::set_log(LogFn fn) {
    log_fn_ = std::move(fn);
}

void GameState::log(const std::string& msg) const {
    if (log_fn_) log_fn_(msg);
}
