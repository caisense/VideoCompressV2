#include "receiver_stats.h"

#include <iomanip>
#include <sstream>

namespace board_receiver {

ReceiverStats::ReceiverStats()
    : packets(0), decoded_frames(0), displayed_frames(0), lost(0), duplicates(0),
      reordered(0), rejected_profiles(0), decoder_restarts(0), last_idr_ms(0),
      started_(std::chrono::steady_clock::now()), profile_bits_(0) {}

void ReceiverStats::setProfile(const StreamProfile& value) {
    uint64_t bits = value.id;
    bits |= static_cast<uint64_t>(value.generation) << 8;
    bits |= static_cast<uint64_t>(value.fps) << 16;
    bits |= static_cast<uint64_t>(value.width) << 24;
    bits |= static_cast<uint64_t>(value.height) << 40;
    profile_bits_.store(bits);
}

std::string ReceiverStats::report(uint64_t mpp_errors, int display_width,
                                  int display_height) {
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_).count();
    const uint64_t bits = profile_bits_.load();
    StreamProfile profile;
    profile.id = bits & 0xff;
    profile.generation = (bits >> 8) & 0xff;
    profile.fps = (bits >> 16) & 0xff;
    profile.width = (bits >> 24) & 0xffff;
    profile.height = (bits >> 40) & 0xffff;
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t idr = last_idr_ms.load();
    std::ostringstream out;
    out << std::fixed << std::setprecision(1)
        << "profile=" << profile.name() << " " << profile.width << 'x'
        << profile.height << '@' << static_cast<int>(profile.fps)
        << " generation=" << static_cast<int>(profile.generation)
        << " rtp=" << packets.load()
        << " recv_fps=" << packets.load() / (seconds > 0 ? seconds : 1)
        << " decode_fps=" << decoded_frames.load() / (seconds > 0 ? seconds : 1)
        << " lost=" << lost.load() << " duplicate=" << duplicates.load()
        << " reordered=" << reordered.load()
        << " rejected_profile=" << rejected_profiles.load()
        << " mpp_errors=" << mpp_errors
        << " idr_age_ms=" << (idr ? now - idr : -1)
        << " display=" << display_width << 'x' << display_height
        << " decoder_restarts=" << decoder_restarts.load();
    return out.str();
}

}  // namespace board_receiver

