#pragma once

#include "../engine/game_runner.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Holds a decision waiting for the GUI player to respond
struct PendingDecision {
    DecisionPoint dp;
    std::optional<GameState> state_snapshot;
    bool ready = false;
};

// Thread-safe game event log
class GameLog {
public:
    void push(const std::string& msg);
    std::vector<std::string> snapshot() const;

private:
    mutable std::mutex mtx_;
    std::vector<std::string> messages_;
};

// Agent that bridges the game thread and the GUI thread.
// - Game thread calls decide() which blocks until the GUI submits an answer.
// - GUI thread checks has_pending(), renders, and calls submit_answer().
class GuiAgent : public Agent {
public:
    explicit GuiAgent(int player_id);

    // Called from game thread — blocks until GUI answers
    std::vector<int> decide(const DecisionPoint& dp, const GameState& state) override;

    // Called from GUI thread
    bool has_pending() const;
    const PendingDecision& get_pending() const;
    void submit_answer(std::vector<int> choices);

    // Unblock decide() so the game thread can exit cleanly
    void cancel();
    bool is_cancelled() const;

    int player_id() const { return pid_; }

private:
    int pid_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    PendingDecision pending_;
    std::vector<int> answer_;
    bool answer_ready_ = false;
    std::atomic<bool> cancelled_{false};
};
