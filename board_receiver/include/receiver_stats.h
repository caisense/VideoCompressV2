#ifndef BOARD_RECEIVER_RECEIVER_STATS_H_
#define BOARD_RECEIVER_RECEIVER_STATS_H_

#include <stdint.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "stream_profile.h"

namespace board_receiver {

struct ReceiverStatsSnapshot {
    StreamProfile profile;
    double receive_fps;
    double decode_fps;
    double rtp_kbps;
    double wire_kbps;
    double tx_wire_kbps;
    double p_fps;
    double i_fps;
    double packets_per_second;
    uint64_t packets;
    uint64_t p_frames;
    uint64_t i_frames;
    uint64_t lost;
    uint64_t reordered;
    uint64_t duplicates;
    uint64_t rejected_profiles;
    uint64_t decoder_restarts;
    uint64_t decode_errors;
    size_t packet_last_bytes;
    double packet_average_bytes;
    size_t packet_max_bytes;
    int64_t source_age_ms;
    int64_t idr_age_ms;

    ReceiverStatsSnapshot();
};

class ReceiverStats {
public:
    ReceiverStats();
    void setProfile(const StreamProfile& profile);
    void setLocalTxWireBps(uint32_t wire_bps);
    void recordPacket(size_t bytes);
    void recordAccessUnit(bool is_irap);
    void recordDecodedFrame();
    ReceiverStatsSnapshot snapshot(uint64_t mpp_errors);
    std::vector<std::string> hudLines(uint64_t mpp_errors);
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
    struct PacketSample {
        std::chrono::steady_clock::time_point at;
        size_t bytes;
        PacketSample(const std::chrono::steady_clock::time_point& value_at,
                     size_t value_bytes) : at(value_at), bytes(value_bytes) {}
    };
    void trimLocked(const std::chrono::steady_clock::time_point& now);

    std::chrono::steady_clock::time_point started_;
    std::atomic<uint64_t> profile_bits_;
    std::atomic<uint32_t> local_tx_wire_bps_;
    std::mutex samples_mutex_;
    std::deque<PacketSample> packet_samples_;
    std::deque<std::chrono::steady_clock::time_point> decode_samples_;
    std::deque<std::chrono::steady_clock::time_point> p_samples_;
    std::deque<std::chrono::steady_clock::time_point> i_samples_;
    uint64_t p_frames_;
    uint64_t i_frames_;
    int64_t last_source_ms_;
};

}  // namespace board_receiver
#endif
