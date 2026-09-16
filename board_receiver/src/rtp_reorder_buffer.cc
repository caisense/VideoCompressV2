#include "rtp_reorder_buffer.h"

#include <limits>

namespace board_receiver {

ReorderResult::ReorderResult()
    : lost(0), duplicates(0), reordered(0), discontinuity(false) {}

RtpReorderBuffer::RtpReorderBuffer(size_t window_packets)
    : window_packets_(window_packets == 0 ? 1 : window_packets),
      initialized_(false), expected_(0) {}

uint16_t RtpReorderBuffer::forwardDistance(uint16_t from, uint16_t to) {
    return static_cast<uint16_t>(to - from);
}

void RtpReorderBuffer::drainReady(ReorderResult* result) {
    while (true) {
        std::map<uint16_t, RtpPacket>::iterator found = pending_.find(expected_);
        if (found == pending_.end()) break;
        result->ready.push_back(found->second);
        pending_.erase(found);
        expected_ = static_cast<uint16_t>(expected_ + 1);
    }
}

void RtpReorderBuffer::releaseNearestAfterGap(ReorderResult* result) {
    if (pending_.empty()) return;
    uint16_t nearest = 0;
    uint16_t nearest_distance = std::numeric_limits<uint16_t>::max();
    for (std::map<uint16_t, RtpPacket>::const_iterator it = pending_.begin();
         it != pending_.end(); ++it) {
        const uint16_t distance = forwardDistance(expected_, it->first);
        if (distance < 0x8000 && distance < nearest_distance) {
            nearest = it->first;
            nearest_distance = distance;
        }
    }
    if (nearest_distance == std::numeric_limits<uint16_t>::max()) {
        pending_.clear();
        return;
    }
    if (nearest_distance > 0) {
        result->lost += nearest_distance;
        result->discontinuity = true;
        expected_ = nearest;
    }
    drainReady(result);
}

ReorderResult RtpReorderBuffer::push(const RtpPacket& packet) {
    ReorderResult result;
    if (!initialized_) {
        initialized_ = true;
        expected_ = packet.sequence;
    }

    const uint16_t distance = forwardDistance(expected_, packet.sequence);
    if (distance >= 0x8000) {
        ++result.duplicates;
        return result;
    }
    if (pending_.find(packet.sequence) != pending_.end()) {
        ++result.duplicates;
        return result;
    }

    if (distance > 0) ++result.reordered;
    pending_.insert(std::make_pair(packet.sequence, packet));
    if (distance >= window_packets_ || pending_.size() >= window_packets_) {
        releaseNearestAfterGap(&result);
    } else {
        drainReady(&result);
    }
    return result;
}

ReorderResult RtpReorderBuffer::flushGap() {
    ReorderResult result;
    releaseNearestAfterGap(&result);
    return result;
}

void RtpReorderBuffer::reset() {
    initialized_ = false;
    expected_ = 0;
    pending_.clear();
}

}  // namespace board_receiver
