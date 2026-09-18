#include "receiver_stats.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace board_receiver {
namespace {

int64_t steadyMilliseconds(const std::chrono::steady_clock::time_point& value) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        value.time_since_epoch()).count();
}

size_t ethernetWireBytes(size_t udp_payload_bytes) {
    return 38U + std::max<size_t>(46U, udp_payload_bytes + 28U);
}

StreamProfile unpackProfile(uint64_t bits) {
    StreamProfile profile;
    profile.id = bits & 0xff;
    profile.generation = (bits >> 8) & 0xff;
    profile.fps = (bits >> 16) & 0xff;
    profile.width = (bits >> 24) & 0xffff;
    profile.height = (bits >> 40) & 0xffff;
    return profile;
}

std::string ageText(int64_t milliseconds) {
    if (milliseconds < 0) return "none";
    std::ostringstream out;
    out << std::fixed << std::setprecision(1)
        << static_cast<double>(milliseconds) / 1000.0 << 's';
    return out.str();
}

}  // namespace

ReceiverStatsSnapshot::ReceiverStatsSnapshot()
    : receive_fps(0.0), decode_fps(0.0), rtp_kbps(0.0), wire_kbps(0.0),
      tx_wire_kbps(0.0), p_fps(0.0), i_fps(0.0), packets_per_second(0.0), packets(0),
      p_frames(0), i_frames(0), lost(0), reordered(0), duplicates(0),
      rejected_profiles(0), decoder_restarts(0), decode_errors(0),
      packet_last_bytes(0), packet_average_bytes(0.0), packet_max_bytes(0),
      source_age_ms(-1), idr_age_ms(-1) {}

ReceiverStats::ReceiverStats()
    : packets(0), decoded_frames(0), displayed_frames(0), lost(0), duplicates(0),
      reordered(0), rejected_profiles(0), decoder_restarts(0), last_idr_ms(0),
      started_(std::chrono::steady_clock::now()), profile_bits_(0), local_tx_wire_bps_(0), p_frames_(0),
      i_frames_(0), last_source_ms_(0) {}

void ReceiverStats::setProfile(const StreamProfile& value) {
    uint64_t bits = value.id;
    bits |= static_cast<uint64_t>(value.generation) << 8;
    bits |= static_cast<uint64_t>(value.fps) << 16;
    bits |= static_cast<uint64_t>(value.width) << 24;
    bits |= static_cast<uint64_t>(value.height) << 40;
    profile_bits_.store(bits);
}

void ReceiverStats::setLocalTxWireBps(uint32_t wire_bps) {
    local_tx_wire_bps_.store(wire_bps);
}

void ReceiverStats::trimLocked(const std::chrono::steady_clock::time_point& now) {
    const std::chrono::steady_clock::time_point cutoff = now - std::chrono::seconds(1);
    while (!packet_samples_.empty() && packet_samples_.front().at < cutoff)
        packet_samples_.pop_front();
    while (!decode_samples_.empty() && decode_samples_.front() < cutoff)
        decode_samples_.pop_front();
    while (!p_samples_.empty() && p_samples_.front() < cutoff)
        p_samples_.pop_front();
    while (!i_samples_.empty() && i_samples_.front() < cutoff)
        i_samples_.pop_front();
}

void ReceiverStats::recordPacket(size_t bytes) {
    ++packets;
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(samples_mutex_);
    packet_samples_.push_back(PacketSample(now, bytes));
    trimLocked(now);
}

void ReceiverStats::recordAccessUnit(bool is_irap) {
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(samples_mutex_);
    if (is_irap) {
        ++i_frames_;
        i_samples_.push_back(now);
    } else {
        ++p_frames_;
        p_samples_.push_back(now);
    }
    trimLocked(now);
}

void ReceiverStats::recordDecodedFrame() {
    ++decoded_frames;
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(samples_mutex_);
    decode_samples_.push_back(now);
    last_source_ms_ = steadyMilliseconds(now);
    trimLocked(now);
}

ReceiverStatsSnapshot ReceiverStats::snapshot(uint64_t mpp_errors) {
    ReceiverStatsSnapshot value;
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    value.profile = unpackProfile(profile_bits_.load());
    const uint32_t local_tx_wire_bps = local_tx_wire_bps_.load();
    value.tx_wire_kbps = static_cast<double>(local_tx_wire_bps) / 1000.0;
    value.packets = packets.load();
    value.lost = lost.load();
    value.reordered = reordered.load();
    value.duplicates = duplicates.load();
    value.rejected_profiles = rejected_profiles.load();
    value.decoder_restarts = decoder_restarts.load();
    value.decode_errors = mpp_errors;

    std::lock_guard<std::mutex> lock(samples_mutex_);
    trimLocked(now);
    size_t rtp_bytes = 0;
    size_t wire_bytes = 0;
    for (std::deque<PacketSample>::const_iterator it = packet_samples_.begin();
         it != packet_samples_.end(); ++it) {
        rtp_bytes += it->bytes;
        wire_bytes += ethernetWireBytes(it->bytes);
        value.packet_max_bytes = std::max(value.packet_max_bytes, it->bytes);
    }
    if (!packet_samples_.empty()) {
        value.packet_last_bytes = packet_samples_.back().bytes;
        value.packet_average_bytes = static_cast<double>(rtp_bytes) /
                                     packet_samples_.size();
    }
    value.receive_fps = static_cast<double>(p_samples_.size() + i_samples_.size());
    value.decode_fps = static_cast<double>(decode_samples_.size());
    value.rtp_kbps = static_cast<double>(rtp_bytes) * 8.0 / 1000.0;
    value.wire_kbps = static_cast<double>(wire_bytes) * 8.0 / 1000.0;
    value.p_fps = static_cast<double>(p_samples_.size());
    value.i_fps = static_cast<double>(i_samples_.size());
    value.packets_per_second = static_cast<double>(packet_samples_.size());
    value.p_frames = p_frames_;
    value.i_frames = i_frames_;
    const int64_t now_ms = steadyMilliseconds(now);
    value.source_age_ms = last_source_ms_ ? now_ms - last_source_ms_ : -1;
    const int64_t idr = last_idr_ms.load();
    value.idr_age_ms = idr ? now_ms - idr : -1;
    return value;
}

std::vector<std::string> ReceiverStats::hudLines(uint64_t mpp_errors) {
    const ReceiverStatsSnapshot value = snapshot(mpp_errors);
    std::vector<std::string> lines;
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << "TX ";
    if (value.tx_wire_kbps == 0.0) out << "--";
    else out << value.tx_wire_kbps;
    out << "  RX " << value.wire_kbps << " kbps";
    lines.push_back(out.str()); out.str(""); out.clear();
    out << "P " << value.p_frames << "  I " << value.i_frames
        << "  TOTAL " << (value.p_frames + value.i_frames);
    lines.push_back(out.str()); out.str(""); out.clear();
    out << "PKT " << value.packets << "  LOSS " << value.lost
        << "  REO " << value.reordered << "  ERR " << value.decode_errors;
    lines.push_back(out.str()); out.str(""); out.clear();
    out << "PROFILE " << value.profile.name() << ' ' << value.profile.width << 'x'
        << value.profile.height << '@' << static_cast<int>(value.profile.fps)
        << "  G" << static_cast<int>(value.profile.generation);
    lines.push_back(out.str()); out.str(""); out.clear();
    out << "SRC " << (value.source_age_ms < 0 ? std::string("none") :
                       std::to_string(value.source_age_ms) + "ms")
        << "  IDR " << ageText(value.idr_age_ms);
    lines.push_back(out.str());
    return lines;
}

std::string ReceiverStats::report(uint64_t mpp_errors, int display_width,
                                  int display_height) {
    const ReceiverStatsSnapshot value = snapshot(mpp_errors);
    std::ostringstream out;
    out << std::fixed << std::setprecision(1)
        << "profile=" << value.profile.name() << " " << value.profile.width << 'x'
        << value.profile.height << '@' << static_cast<int>(value.profile.fps)
        << " generation=" << static_cast<int>(value.profile.generation)
        << " rtp=" << value.packets
        << " recv_fps=" << value.receive_fps
        << " decode_fps=" << value.decode_fps
        << " tx_wire_kbps=" << value.tx_wire_kbps
        << " rtp_kbps=" << value.rtp_kbps
        << " wire_kbps=" << value.wire_kbps
        << " lost=" << value.lost << " duplicate=" << value.duplicates
        << " reordered=" << value.reordered
        << " rejected_profile=" << value.rejected_profiles
        << " mpp_errors=" << value.decode_errors
        << " idr_age_ms=" << value.idr_age_ms
        << " display=" << display_width << 'x' << display_height
        << " decoder_restarts=" << value.decoder_restarts;
    return out.str();
}

}  // namespace board_receiver
