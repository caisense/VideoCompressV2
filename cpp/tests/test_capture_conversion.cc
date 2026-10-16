#include <cassert>
#include <iostream>

#include "capture/conversion_retry.h"

using namespace roi_h265;

int main() {
    int calls = 0;
    CaptureConversionResult result = convertCaptureFrame([&]() { ++calls; return 0; }, true);
    assert(result.status == 0 && result.attempts == 1 && calls == 1);

    calls = 0;
    result = convertCaptureFrame([&]() { return ++calls == 1 ? -1 : 0; }, true);
    assert(result.status == 0 && result.attempts == 2 && calls == 2);

    calls = 0;
    result = convertCaptureFrame([&]() { ++calls; return -7; }, true);
    assert(result.status == -7 && result.attempts == 2 && calls == 2);

    calls = 0;
    result = convertCaptureFrame([&]() { ++calls; return -7; }, false);
    assert(result.status == -7 && result.attempts == 1 && calls == 1);
    std::cout << "capture conversion retry tests passed" << std::endl;
    return 0;
}
