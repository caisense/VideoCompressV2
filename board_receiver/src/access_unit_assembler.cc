#include "access_unit_assembler.h"

namespace board_receiver {

AccessUnit::AccessUnit()
    : timestamp(0), ssrc(0), has_vps(false), has_sps(false), has_pps(false),
      has_irap(false) {}

bool AccessUnit::completeRandomAccessPoint() const {
    return has_vps && has_sps && has_pps && has_irap && !bytes.empty();
}

AssemblyResult::AssemblyResult() : ready(false), dropped(false) {}

AccessUnitAssembler::AccessUnitAssembler() : active_(false), corrupt_(false) {}

void AccessUnitAssembler::reset() {
    active_ = false;
    corrupt_ = false;
    current_ = AccessUnit();
}

AssemblyResult AccessUnitAssembler::feed(const RtpPacket& packet,
                                         const StreamProfile& profile,
                                         const DepacketizedPayload& payload,
                                         bool discontinuity) {
    AssemblyResult result;
    const ProfileKey incoming_key(profile);
    if (discontinuity) {
        result.dropped = active_;
        reset();
    }
    if (active_ && (packet.timestamp != current_.timestamp ||
                    packet.ssrc != current_.ssrc ||
                    ProfileKey(current_.profile) != incoming_key)) {
        result.dropped = true;
        reset();
    }
    if (!active_) {
        active_ = true;
        current_.timestamp = packet.timestamp;
        current_.ssrc = packet.ssrc;
        current_.profile = profile;
    }

    if (!payload.valid) {
        corrupt_ = true;
    } else {
        current_.bytes.insert(current_.bytes.end(), payload.bytes.begin(), payload.bytes.end());
        for (size_t i = 0; i < payload.nal_types.size(); ++i) {
            const uint8_t type = payload.nal_types[i];
            current_.has_vps = current_.has_vps || type == 32;
            current_.has_sps = current_.has_sps || type == 33;
            current_.has_pps = current_.has_pps || type == 34;
            current_.has_irap = current_.has_irap || isHevcIrap(type);
        }
    }

    if (!packet.marker) return result;
    if (corrupt_ || payload.fu_in_progress || current_.bytes.empty()) {
        result.dropped = true;
        reset();
        return result;
    }

    result.ready = true;
    result.access_unit = current_;
    reset();
    return result;
}

}  // namespace board_receiver
