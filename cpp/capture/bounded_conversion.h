#ifndef ROI_H265_CAPTURE_BOUNDED_CONVERSION_H_
#define ROI_H265_CAPTURE_BOUNDED_CONVERSION_H_

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace roi_h265 {

// One owned operation, no FIFO. A timed-out operation keeps its buffers until
// the driver returns; callers must never reuse its output as a later frame.
class BoundedConversion {
public:
    enum Outcome { COMPLETE, BUSY, DEADLINE };
    struct Result {
        Outcome outcome;
        int status;
    };

    BoundedConversion() : state_(new State), thread_(&BoundedConversion::work, state_) {}

    ~BoundedConversion() {
        bool active;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            active = static_cast<bool>(state_->task);
            state_->stopping = true;
            state_->wake.notify_all();
        }
        // A stuck driver must not stall shutdown. The thread references only
        // shared State/Task objects, never this owner or camera storage.
        if (active) thread_.detach(); else thread_.join();
    }

    bool busy() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return static_cast<bool>(state_->task);
    }

    Result run(const std::function<int()> &operation, std::chrono::milliseconds budget) {
        std::unique_lock<std::mutex> lock(state_->mutex);
        if (state_->task) return {BUSY, -1};
        std::shared_ptr<Task> task(new Task(operation));
        state_->task = task;
        state_->wake.notify_one();
        if (!state_->finished.wait_for(lock, budget, [&]() { return task->done; })) {
            return {DEADLINE, -1};
        }
        return {COMPLETE, task->status};
    }

    BoundedConversion(const BoundedConversion &) = delete;
    BoundedConversion &operator=(const BoundedConversion &) = delete;

private:
    struct Task {
        std::function<int()> operation;
        bool done;
        int status;
        explicit Task(const std::function<int()> &op) : operation(op), done(false), status(-1) {}
    };
    struct State {
        std::mutex mutex;
        std::condition_variable wake;
        std::condition_variable finished;
        std::shared_ptr<Task> task;
        bool stopping;
        State() : stopping(false) {}
    };

    static void work(const std::shared_ptr<State> &state) {
        std::unique_lock<std::mutex> lock(state->mutex);
        for (;;) {
            state->wake.wait(lock, [&]() { return state->stopping || state->task; });
            if (state->stopping) return;
            std::shared_ptr<Task> task = state->task;
            lock.unlock();
            int status = -1;
            try { status = task->operation(); } catch (...) { status = -1; }
            lock.lock();
            task->status = status;
            task->done = true;
            state->task.reset();
            state->finished.notify_all();
        }
    }

    std::shared_ptr<State> state_;
    std::thread thread_;
};

}  // namespace roi_h265
#endif
