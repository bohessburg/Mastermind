#include "diff_bridge.h"

#include "v2/core/game.h"
#include "v2/core/score.h"
#include "v2/core/turns.h"

#include <sstream>
#include <stdexcept>

namespace dz_diff {
namespace {

[[nodiscard]] DefId def_named(const std::string& name) {
    for (DefId def = 0; def < card_def_count(); ++def) {
        if (name == card_def(def).name) {
            return def;
        }
    }
    throw std::runtime_error("v2 unknown card: " + name);
}

[[nodiscard]] Setup setup_for(const Scenario& scenario) {
    Setup setup{};
    setup.num_players = 2;
    if (scenario.kingdom.size() > static_cast<std::size_t>(MAX_KINGDOM_DEFS)) {
        throw std::runtime_error("v2 differential kingdom too large");
    }
    setup.kingdom_count = static_cast<std::uint8_t>(scenario.kingdom.size());
    for (std::size_t i = 0; i < scenario.kingdom.size(); ++i) {
        setup.kingdom[i] = def_named(scenario.kingdom[i]);
    }
    return setup;
}

void clear_player(PlayerState& player) {
    for (std::uint8_t slot = 0; slot < MAX_SLOTS; ++slot) {
        player.hand[slot] = 0;
        player.exile[slot] = 0;
        player.tavern[slot] = 0;
        player.island_mat[slot] = 0;
    }
    player.deck.size = 0;
    player.discard.size = 0;
    player.set_aside.size = 0;
    player.in_play_size = 0;
    player.pending_size = 0;
}

[[nodiscard]] Slot slot_named(const GameState& state, const std::string& name) {
    const DefId def = def_named(name);
    const Slot slot = slot_of(state, def);
    if (slot == NONE) {
        throw std::runtime_error("v2 missing slot: " + name);
    }
    return slot;
}

void add_cards_to_hand(GameState& state, PlayerId player_id, const std::vector<std::string>& names) {
    PlayerState& player = state.players[player_id];
    for (const std::string& name : names) {
        ++player.hand[slot_named(state, name)];
    }
}

void add_cards_to_deck(GameState& state, PlayerId player_id, const std::vector<std::string>& names) {
    PlayerState& player = state.players[player_id];
    for (const std::string& name : names) {
        if (player.deck.size >= MAX_DECK_CARDS) {
            throw std::runtime_error("v2 deck setup overflow");
        }
        player.deck.cards[player.deck.size] = slot_named(state, name);
        ++player.deck.size;
    }
}

void add_cards_to_discard(GameState& state, PlayerId player_id, const std::vector<std::string>& names) {
    PlayerState& player = state.players[player_id];
    for (const std::string& name : names) {
        if (player.discard.size >= MAX_DECK_CARDS) {
            throw std::runtime_error("v2 discard setup overflow");
        }
        player.discard.cards[player.discard.size] = slot_named(state, name);
        ++player.discard.size;
    }
}

void reset_for_scenario(GameState& state, const Scenario& scenario) {
    for (PlayerId player = 0; player < state.num_players; ++player) {
        clear_player(state.players[player]);
    }
    for (std::size_t player = 0; player < scenario.players.size(); ++player) {
        const PlayerSetup& setup = scenario.players[player];
        const PlayerId player_id = static_cast<PlayerId>(player);
        add_cards_to_hand(state, player_id, setup.hand);
        add_cards_to_deck(state, player_id, setup.deck);
        add_cards_to_discard(state, player_id, setup.discard);
    }
    state.phase = static_cast<std::uint8_t>(Phase::Action);
    state.actions = 1;
    state.buys = 1;
    state.coins = 0;
    state.potion_coins = 0;
    state.effect_depth = 0;
    state.turn_counter = 0;
    state.truncated = 0;
    state.decision = PendingDecision{0U, static_cast<std::uint8_t>(DecisionKind::PhaseAction), 0, 0, 0};
    state.trigger_table.dirty = 1U;
    turn_queue_clear(state.turn_queue);
    const bool pushed = turn_queue_push(state.turn_queue, 0U, TurnKind::Normal);
    if (!pushed) {
        throw std::runtime_error("v2 failed to seed turn queue");
    }
}

void step_checked(GameState& state, Action action) {
    ActionMask legal{};
    const int count = Game::legal_actions(state, legal);
    if (count <= 0 || action >= ACTION_SPACE_SIZE || !legal.test(action)) {
        throw std::runtime_error("v2 illegal differential action");
    }
    (void)Game::step(state, action);
}

void apply_choice(GameState& state, const Choice& choice) {
    switch (choice.kind) {
    case ChoiceKind::Pass:
        step_checked(state, A_PASS);
        break;
    case ChoiceKind::Option:
        step_checked(state, option_action(static_cast<std::uint8_t>(choice.option)));
        break;
    case ChoiceKind::Select:
        step_checked(state, select_action(def_named(choice.name)));
        break;
    }
}

void start_buy(GameState& state) {
    if (state.phase == static_cast<std::uint8_t>(Phase::Action)) {
        step_checked(state, A_PASS);
    }
}

void play_named(GameState& state, const Step& step) {
    step_checked(state, play_action(def_named(step.name)));
    for (const Choice& choice : step.choices) {
        apply_choice(state, choice);
    }
}

void buy_named(GameState& state, const std::string& name) {
    start_buy(state);
    step_checked(state, buy_action(def_named(name)));
}

void end_turn(GameState& state) {
    const std::uint16_t before = state.turn_counter;
    for (int i = 0; i < 8 && state.turn_counter == before; ++i) {
        step_checked(state, A_PASS);
    }
    if (state.turn_counter == before) {
        throw std::runtime_error("v2 failed to end turn");
    }
}

void add_slot_name(std::map<std::string, int>& out, const GameState& state, Slot slot) {
    ++out[card_def(state.slot_to_def[slot]).name];
}

void add_count_zone(std::map<std::string, int>& out, const GameState& state, const std::uint8_t (&zone)[MAX_SLOTS]) {
    for (std::uint8_t slot = 0; slot < state.num_slots; ++slot) {
        if (zone[slot] != 0U) {
            out[card_def(state.slot_to_def[slot]).name] += zone[slot];
        }
    }
}

void add_ordered_zone(std::map<std::string, int>& out, const GameState& state, const OrderedZone& zone) {
    for (std::uint8_t i = 0; i < zone.size; ++i) {
        add_slot_name(out, state, zone.cards[i]);
    }
}

[[nodiscard]] std::map<std::string, int> owned_counts(const GameState& state, PlayerId player_id) {
    std::map<std::string, int> result;
    const PlayerState& player = state.players[player_id];
    add_count_zone(result, state, player.hand);
    add_count_zone(result, state, player.exile);
    add_count_zone(result, state, player.tavern);
    add_count_zone(result, state, player.island_mat);
    add_ordered_zone(result, state, player.deck);
    add_ordered_zone(result, state, player.discard);
    add_ordered_zone(result, state, player.set_aside);
    for (std::uint8_t i = 0; i < player.in_play_size; ++i) {
        add_slot_name(result, state, player.in_play[i].slot);
    }
    return result;
}

[[nodiscard]] std::map<std::string, int> trash_counts(const GameState& state) {
    std::map<std::string, int> result;
    add_count_zone(result, state, state.trash);
    return result;
}

[[nodiscard]] std::map<std::string, int> supply_counts(const GameState& state) {
    std::map<std::string, int> result;
    for (std::uint8_t i = 0; i < state.num_piles; ++i) {
        const Pile& pile = state.piles[i];
        if (pile.mixed_len == 0U) {
            result[card_def(state.slot_to_def[pile.base]).name] = pile.count;
        }
    }
    return result;
}

[[nodiscard]] std::string dump_zones(const GameState& state) {
    std::ostringstream out;
    for (PlayerId player_id = 0; player_id < state.num_players; ++player_id) {
        const PlayerState& player = state.players[player_id];
        out << "zones p" << static_cast<int>(player_id)
            << " deck=" << static_cast<int>(player.deck.size)
            << " discard=" << static_cast<int>(player.discard.size)
            << " set_aside=" << static_cast<int>(player.set_aside.size)
            << " in_play=" << static_cast<int>(player.in_play_size)
            << "\n";
    }
    return out.str();
}

} // namespace

Snapshot run_v2(const Scenario& scenario) {
    GameState state = Game::new_game(setup_for(scenario), 0xD1FF'2026ULL);
    reset_for_scenario(state, scenario);

    for (const Step& step : scenario.steps) {
        switch (step.kind) {
        case StepKind::StartBuy:
            start_buy(state);
            break;
        case StepKind::Play:
            play_named(state, step);
            break;
        case StepKind::Buy:
            buy_named(state, step.name);
            break;
        case StepKind::EndTurn:
            end_turn(state);
            break;
        }
    }

    Snapshot snapshot{};
    snapshot.owned.resize(state.num_players);
    for (PlayerId player = 0; player < state.num_players; ++player) {
        snapshot.owned[player] = owned_counts(state, player);
        snapshot.scores.push_back(score(state, player));
    }
    snapshot.supply = supply_counts(state);
    snapshot.trash = trash_counts(state);
    snapshot.completed_turns = state.turn_counter;
    snapshot.dump = dump_zones(state);
    return snapshot;
}

} // namespace dz_diff
