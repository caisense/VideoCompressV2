#include <cassert>
#include <iostream>

#include "capture/capture_policy.h"

using namespace roi_h265;

int main() {
    unsigned int calls = 0;
    CaptureDrainResult result = drainReadyCaptureFrames(7, [&]() { ++calls; return 1; });
    assert(result.ok && result.discarded == 7 && calls == 7);
    calls = 0;
    result = drainReadyCaptureFrames(7, [&]() { return ++calls < 3 ? 1 : 0; });
    assert(result.ok && result.discarded == 2 && calls == 3);
    calls = 0;
    result = drainReadyCaptureFrames(0, [&]() { ++calls; return 1; });
    assert(result.ok && result.discarded == 0 && calls == 0);
    result = drainReadyCaptureFrames(7, [&]() { return -1; });
    assert(!result.ok && result.discarded == 0);
    assert(monotonicCaptureTimestamp(100, 123, true, 100000200) == 100000123);
    assert(monotonicCaptureTimestamp(100, 123, false, 100000200) == 0);
    assert(monotonicCaptureTimestamp(101, 0, true, 100000200) == 0);
    assert(monotonicCaptureTimestamp(-1, 0, true, 100000200) == 0);
    assert(monotonicCaptureTimestamp(100, -1, true, 100000200) == 0);
    assert(monotonicCaptureTimestamp(100, 1000000, true, 100000200) == 0);
    std::cout << "capture latest-buffer / timestamp policy tests passed" << std::endl;
    return 0;
}
