#include "gui_agent.h"

// --- GameLog ---

void GameLog::push(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    messages_.push_back(msg);
}

std::vector<std::string> GameLog::snapshot() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return messages_;
}

// --- GuiAgent ---

GuiAgent::GuiAgent(int player_id) : pid_(player_id) {}

std::vector<int> GuiAgent::decide(const DecisionPoint& dp, const GameState& state) {
    // Post the decision for the GUI thread
    {
        std::lock_guard<std::mutex> lock(mtx_);
        pending_.dp = dp;
        pending_.state_snapshot = state;
        pending_.ready = true;
        answer_ready_ = false;
    }

    // Wait for the GUI thread to submit an answer (or cancel)
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return answer_ready_ || cancelled_.load(); });

    if (cancelled_.load()) {
        pending_.ready = false;
        return {};
    }

    auto result = std::move(answer_);
    answer_ready_ = false;
    pending_.ready = false;
    return result;
}

bool GuiAgent::has_pending() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pending_.ready;
}

const PendingDecision& GuiAgent::get_pending() const {
    // Caller should only read this when has_pending() is true.
    // The data is stable while pending_.ready is true because
    // the game thread is blocked.
    return pending_;
}

void GuiAgent::submit_answer(std::vector<int> choices) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        answer_ = std::move(choices);
        answer_ready_ = true;
    }
    cv_.notify_one();
}

void GuiAgent::cancel() {
    cancelled_.store(true);
    cv_.notify_one();
}

bool GuiAgent::is_cancelled() const {
    return cancelled_.load();
}
