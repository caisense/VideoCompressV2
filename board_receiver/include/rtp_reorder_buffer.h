#ifndef BOARD_RECEIVER_RTP_REORDER_BUFFER_H_
#define BOARD_RECEIVER_RTP_REORDER_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <vector>

#include "rtp_packet.h"

namespace board_receiver {

struct ReorderResult {
    std::vector<RtpPacket> ready;
    uint32_t lost;
    uint32_t duplicates;
    uint32_t reordered;
    bool discontinuity;

    ReorderResult();
};

class RtpReorderBuffer {
public:
    explicit RtpReorderBuffer(size_t window_packets);

    ReorderResult push(const RtpPacket& packet);
    ReorderResult flushGap();
    void reset();
    bool empty() const { return pending_.empty(); }

private:
    static uint16_t forwardDistance(uint16_t from, uint16_t to);
    void drainReady(ReorderResult* result);
    void releaseNearestAfterGap(ReorderResult* result);

    size_t window_packets_;
    bool initialized_;
    uint16_t expected_;
    std::map<uint16_t, RtpPacket> pending_;
};

}  // namespace board_receiver

#endif  // BOARD_RECEIVER_RTP_REORDER_BUFFER_H_
