#include "transport/async_rtp_sender.h"

#include <algorithm>
#include <utility>

namespace roi_h265 {
namespace {

uint64_t steadyNowMicros() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace

AsyncRtpSender::AsyncRtpSender(const std::string &host, int port, int pacing_bitrate_bps, int mtu,
                               size_t max_queue_frames, int max_queue_latency_ms,
                               const std::shared_ptr<RatePacer> &pacer,
                               const std::string &tx_rate_socket_path)
    : udp_sender_(host, port, pacing_bitrate_bps, mtu, pacer),
      packetizer_(0, 0x524f4931U, mtu - 28),
      tx_rate_publisher_(tx_rate_socket_path),
      max_queue_frames_(std::max<size_t>(1, max_queue_frames)),
      max_queue_latency_(std::max(1, max_queue_latency_ms)),
      started_(false), stopping_(false), failed_(false), waiting_for_key_frame_(false),
      sending_(false), sent_frames_(0), dropped_p_frames_(0), dropped_key_frames_(0),
      last_sent_frame_id_(0), last_capture_to_send_us_(0), last_queue_delay_us_(0) {}

AsyncRtpSender::~AsyncRtpSender() { stop(); }

bool AsyncRtpSender::start(std::string *error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) return !failed_;
    if (!udp_sender_.open(error)) return false;
    stopping_ = false;
    failed_ = false;
    started_ = true;
    worker_ = std::thread(&AsyncRtpSender::workerLoop, this);
    return true;
}

void AsyncRtpSender::enterRecoveryLocked() {
    waiting_for_key_frame_ = true;
    dropAllPFramesLocked();
}

void AsyncRtpSender::dropAllPFramesLocked() {
    for (std::deque<QueueItem>::iterator it = queue_.begin(); it != queue_.end();) {
        if (!it->access_unit.key_frame) {
            ++dropped_p_frames_;
            it = queue_.erase(it);
        } else {
            ++it;
        }
    }
}

void AsyncRtpSender::dropExpiredPFramesLocked(const std::chrono::steady_clock::time_point &now) {
    bool found_expired_p = false;
    for (std::deque<QueueItem>::const_iterator it = queue_.begin(); it != queue_.end(); ++it) {
        if (!it->access_unit.key_frame && now - it->enqueued_at > max_queue_latency_) {
            found_expired_p = true;
            break;
        }
    }
    if (found_expired_p) enterRecoveryLocked();
}

bool AsyncRtpSender::enqueue(RtpAccessUnit access_unit, std::string *error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_ || stopping_ || failed_) {
        if (error) *error = failed_ ? failure_message_ : "asynchronous RTP sender is not running";
        return false;
    }
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    dropExpiredPFramesLocked(now);

    if (waiting_for_key_frame_ && !access_unit.key_frame) {
        ++dropped_p_frames_;
        return true;
    }

    if (access_unit.key_frame) {
        // A new IDR supersedes any queued recovery point and all P frames that
        // depended on an older chain. Keep the currently transmitted AU whole.
        for (std::deque<QueueItem>::iterator it = queue_.begin(); it != queue_.end();) {
            if (it->access_unit.key_frame) ++dropped_key_frames_;
            else ++dropped_p_frames_;
            it = queue_.erase(it);
        }
        waiting_for_key_frame_ = false;
    } else if (queue_.size() >= max_queue_frames_) {
        // Dropping an encoded reference P frame invalidates following P frames.
        // Enter recovery and reject this P frame until MPP produces an IDR.
        enterRecoveryLocked();
        ++dropped_p_frames_;
        return true;
    }

    QueueItem item;
    item.access_unit = std::move(access_unit);
    item.enqueued_at = now;
    queue_.push_back(std::move(item));
    condition_.notify_one();
    return true;
}

void AsyncRtpSender::setFailureLocked(const std::string &error) {
    failed_ = true;
    stopping_ = true;
    failure_message_ = error;
    queue_.clear();
    condition_.notify_all();
}

void AsyncRtpSender::workerLoop() {
    for (;;) {
        QueueItem item;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_) break;
            dropExpiredPFramesLocked(std::chrono::steady_clock::now());
            if (queue_.empty()) continue;
            item = std::move(queue_.front());
            queue_.pop_front();
            sending_ = true;
        }

        const std::vector<std::vector<uint8_t> > packets = packetizer_.packetize(
            item.access_unit.bytes.data(), item.access_unit.bytes.size(),
            rtpTimestamp(item.access_unit.frame.pts_us), &item.access_unit.stream_profile);
        size_t sent_wire_bytes = 0;
        for (size_t index = 0; index < packets.size(); ++index) {
            sent_wire_bytes += RatePacer::wireBytesForUdpPayload(packets[index].size());
        }
        std::string send_error;
        const bool sent = udp_sender_.sendPackets(packets, &send_error);
        const uint64_t completed_us = steadyNowMicros();

        std::lock_guard<std::mutex> lock(mutex_);
        sending_ = false;
        condition_.notify_all();
        if (!sent) {
            setFailureLocked(send_error);
            break;
        }
        ++sent_frames_;
        if (item.access_unit.key_frame) waiting_for_key_frame_ = false;
        last_sent_frame_id_ = item.access_unit.frame.frame_id;
        last_capture_to_send_us_ = item.access_unit.frame.capture_time_us > 0 &&
            completed_us >= item.access_unit.frame.capture_time_us
            ? completed_us - item.access_unit.frame.capture_time_us : 0;
        last_queue_delay_us_ = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - item.enqueued_at).count());
        const std::chrono::steady_clock::time_point completed_at =
            std::chrono::steady_clock::now();
        wire_samples_.push_back(WireSample(completed_at, sent_wire_bytes));
        const std::chrono::steady_clock::time_point cutoff =
            completed_at - std::chrono::seconds(1);
        while (!wire_samples_.empty() && wire_samples_.front().at < cutoff) {
            wire_samples_.pop_front();
        }
        uint64_t wire_bytes = 0;
        for (std::deque<WireSample>::const_iterator it = wire_samples_.begin();
             it != wire_samples_.end(); ++it) {
            wire_bytes += it->bytes;
        }
        const uint64_t wire_bps = wire_bytes * 8U;
        tx_rate_publisher_.publish(static_cast<uint32_t>(
            std::min<uint64_t>(wire_bps, static_cast<uint64_t>(UINT32_MAX))));
    }
}

bool AsyncRtpSender::switchProfile(int pacing_bitrate_bps, std::string *error) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!started_ || stopping_ || failed_) {
        if (error) *error = failed_ ? failure_message_ : "asynchronous RTP sender is not running";
        return false;
    }
    for (std::deque<QueueItem>::const_iterator it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->access_unit.key_frame) ++dropped_key_frames_;
        else ++dropped_p_frames_;
    }
    queue_.clear();
    waiting_for_key_frame_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this] { return !sending_ || failed_; });
    if (failed_) {
        if (error) *error = failure_message_;
        return false;
    }
    udp_sender_.setPacingBitrate(pacing_bitrate_bps);
    return true;
}

void AsyncRtpSender::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        stopping_ = true;
        queue_.clear();
        condition_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
}

bool AsyncRtpSender::needsKeyFrame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waiting_for_key_frame_;
}

bool AsyncRtpSender::failed(std::string *error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_ && error) *error = failure_message_;
    return failed_;
}

AsyncRtpSenderSnapshot AsyncRtpSender::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    AsyncRtpSenderSnapshot value;
    value.queued_frames = queue_.size();
    for (std::deque<QueueItem>::const_iterator it = queue_.begin(); it != queue_.end(); ++it)
        value.queued_bytes += it->access_unit.bytes.size();
    value.sent_frames = sent_frames_;
    value.dropped_p_frames = dropped_p_frames_;
    value.dropped_key_frames = dropped_key_frames_;
    value.last_sent_frame_id = last_sent_frame_id_;
    value.last_capture_to_send_us = last_capture_to_send_us_;
    value.last_queue_delay_us = last_queue_delay_us_;
    uint64_t wire_bytes = 0;
    const std::chrono::steady_clock::time_point cutoff =
        std::chrono::steady_clock::now() - std::chrono::seconds(1);
    for (std::deque<WireSample>::const_iterator it = wire_samples_.begin();
         it != wire_samples_.end(); ++it) {
        if (it->at >= cutoff) wire_bytes += it->bytes;
    }
    const uint64_t wire_bps = wire_bytes * 8U;
    value.tx_wire_bps = static_cast<uint32_t>(
        std::min<uint64_t>(wire_bps, static_cast<uint64_t>(UINT32_MAX)));
    value.waiting_for_key_frame = waiting_for_key_frame_;
    value.sending = sending_;
    return value;
}

uint32_t AsyncRtpSender::rtpTimestamp(uint64_t pts_us) {
    return static_cast<uint32_t>((pts_us * 90ULL) / 1000ULL);
}

}  // namespace roi_h265
