#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <thread>

#include "capture/bounded_conversion.h"
#include "capture/planar_to_nv12.h"

using namespace roi_h265;

int main() {
    const uint8_t planar[] = {0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 21, 22};
    const std::vector<uint8_t> expected = {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 12, 22};
    std::vector<uint8_t> nv12;
    assert(planarI420ToNv12(planar, sizeof(planar), 4, 2, &nv12) && nv12 == expected);
    assert(!planarI420ToNv12(planar, 10, 4, 2, &nv12));
    assert(!planarI420ToNv12(planar, sizeof(planar), 3, 2, &nv12));
    BoundedConversion worker;
    int output = 0;
    BoundedConversion::Result ok = worker.run([&]() { output = 7; return 0; },
                                             std::chrono::milliseconds(1000));
    assert(ok.outcome == BoundedConversion::COMPLETE && ok.status == 0 && output == 7);

    std::promise<void> release;
    std::shared_future<void> barrier = release.get_future().share();
    std::atomic<bool> second(false);
    BoundedConversion::Result late = worker.run([barrier]() { barrier.wait(); return -9; },
                                               std::chrono::milliseconds(1));
    assert(late.outcome == BoundedConversion::DEADLINE);
    BoundedConversion::Result busy = worker.run([&]() { second = true; return 0; },
                                               std::chrono::milliseconds(1000));
    assert(busy.outcome == BoundedConversion::BUSY && !second);
    release.set_value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.busy() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    assert(!worker.busy());
    ok = worker.run([]() { return 0; }, std::chrono::milliseconds(1000));
    assert(ok.outcome == BoundedConversion::COMPLETE && ok.status == 0);

    // Destruction while blocked neither waits for the accelerator nor frees
    // its storage. Release it only after the owner is gone.
    std::promise<void> unblock;
    std::shared_future<void> blocked = unblock.get_future().share();
    std::promise<void> started;
    std::future<void> running = started.get_future();
    std::weak_ptr<int> lifetime;
    {
        BoundedConversion temporary;
        std::shared_ptr<int> buffer(new int(42));
        lifetime = buffer;
        late = temporary.run([buffer, blocked, &started]() { started.set_value(); blocked.wait(); return *buffer; },
                             std::chrono::milliseconds(1));
        assert(late.outcome == BoundedConversion::DEADLINE);
        assert(running.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    }
    assert(!lifetime.expired());
    unblock.set_value();
    const auto released = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!lifetime.expired() && std::chrono::steady_clock::now() < released) std::this_thread::yield();
    assert(lifetime.expired());
    std::cout << "bounded conversion ownership/deadline tests passed" << std::endl;
}
