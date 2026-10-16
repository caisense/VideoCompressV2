#ifndef ROI_H265_CAPTURE_CAPTURE_POLICY_H_
#define ROI_H265_CAPTURE_CAPTURE_POLICY_H_

#include <stdint.h>
#include <limits>

namespace roi_h265 {

struct CaptureDrainResult {
    bool ok;
    unsigned int discarded;
};

// next() replaces the held frame and returns 1, reports no ready frame with 0,
// or reports an I/O error with -1. Never chase an indefinitely busy producer.
template <typename Next>
CaptureDrainResult drainReadyCaptureFrames(unsigned int max_extra, Next next) {
    CaptureDrainResult result = {true, 0};
    for (unsigned int i = 0; i < max_extra; ++i) {
        const int status = next();
        if (status < 0) { result.ok = false; break; }
        if (status == 0) break;
        ++result.discarded;
    }
    return result;
}

inline uint64_t monotonicCaptureTimestamp(int64_t seconds, int64_t micros,
                                        bool monotonic, uint64_t read_time_us) {
    if (!monotonic || seconds < 0 || micros < 0 || micros >= 1000000 ||
        static_cast<uint64_t>(seconds) > (std::numeric_limits<uint64_t>::max() -
            static_cast<uint64_t>(micros)) / 1000000U) return 0;
    const uint64_t value = static_cast<uint64_t>(seconds) * 1000000U +
        static_cast<uint64_t>(micros);
    return value <= read_time_us ? value : 0;
}

}  // namespace roi_h265

#endif  // ROI_H265_CAPTURE_CAPTURE_POLICY_H_
