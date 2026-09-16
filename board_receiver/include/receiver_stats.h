#ifndef BOARD_RECEIVER_RECEIVER_STATS_H_
#define BOARD_RECEIVER_RECEIVER_STATS_H_

#include <stdint.h>

#include <atomic>
#include <chrono>
#include <string>

#include "stream_profile.h"

namespace board_receiver {

class ReceiverStats {
public:
    ReceiverStats();
    void setProfile(const StreamProfile& profile);
    std::string report(uint64_t mpp_errors, int display_width, int display_height);

    std::atomic<uint64_t> packets;
    std::atomic<uint64_t> decoded_frames;
    std::atomic<uint64_t> displayed_frames;
    std::atomic<uint64_t> lost;
    std::atomic<uint64_t> duplicates;
    std::atomic<uint64_t> reordered;
    std::atomic<uint64_t> rejected_profiles;
    std::atomic<uint64_t> decoder_restarts;
    std::atomic<int64_t> last_idr_ms;

private:
    std::chrono::steady_clock::time_point started_;
    std::atomic<uint64_t> profile_bits_;
};

}  // namespace board_receiver
#endif

