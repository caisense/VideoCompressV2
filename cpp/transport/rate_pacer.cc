#include "transport/rate_pacer.h"

#include <algorithm>
#include <thread>
#include <stdexcept>

namespace roi_h265 {

namespace {

const size_t kIpv4UdpHeaderBytes = 28;
const size_t kEthernetWireOverheadBytes = 38;
const size_t kMinimumEthernetPayloadBytes = 46;

}  // namespace

RatePacer::RatePacer(int bitrate_bps, size_t max_burst_bytes)
    : bitrate_bps_(0), tokens_(0.0), max_burst_bytes_(0.0),
      last_refill_(std::chrono::steady_clock::now()) {
    reset(bitrate_bps, max_burst_bytes);
}

size_t RatePacer::wireBytesForUdpPayload(size_t udp_payload_bytes) {
    const size_t ip_packet_bytes = udp_payload_bytes + kIpv4UdpHeaderBytes;
    return kEthernetWireOverheadBytes + std::max(kMinimumEthernetPayloadBytes, ip_packet_bytes);
}

void RatePacer::reset(int bitrate_bps, size_t max_burst_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    bitrate_bps_ = std::max(1, bitrate_bps);
    max_burst_bytes_ = std::max(1.0, static_cast<double>(max_burst_bytes));
    // Start empty: startup is not allowed to spend a one-second burst budget.
    tokens_ = 0.0;
    last_refill_ = std::chrono::steady_clock::now();
}

int RatePacer::bitrateBps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bitrate_bps_;
}

void RatePacer::refillLocked(const std::chrono::steady_clock::time_point &now) {
    const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(
        now - last_refill_).count();
    if (elapsed_seconds > 0.0) {
        tokens_ = std::min(max_burst_bytes_, tokens_ + elapsed_seconds * bitrate_bps_ / 8.0);
        last_refill_ = now;
    }
}

void RatePacer::waitForTokens(size_t bytes) {
    size_t remaining = bytes;
    while (remaining > 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        refillLocked(now);
        const double allowed = std::min<double>(remaining, max_burst_bytes_);
        if (tokens_ >= allowed) {
            tokens_ -= allowed;
            remaining -= static_cast<size_t>(allowed);
            continue;
        }
        const double seconds = (allowed - tokens_) * 8.0 / bitrate_bps_;
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    }
    // Charge the whole datagram at its final admission time, even if its
    // token cost was acquired in several pieces. Keep history across reset.
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        while (!window_.empty() && now - window_.front().first >= std::chrono::seconds(1)) {
            window_bytes_ -= window_.front().second;
            window_.pop_front();
        }
        const size_t limit = std::max<size_t>(1, static_cast<size_t>(bitrate_bps_ / 8));
        if (bytes > limit) {
            throw std::invalid_argument("datagram exceeds one-second physical wire budget");
        }
        if (window_bytes_ + bytes <= limit) {
            window_.push_back(std::make_pair(now, bytes));
            window_bytes_ += bytes;
            return;
        }
        const auto wake = window_.front().first + std::chrono::seconds(1);
        lock.unlock();
        std::this_thread::sleep_until(wake);
    }
}

}  // namespace roi_h265
