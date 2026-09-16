#include "profile_switch_gate.h"

namespace board_receiver {

ProfileSwitchGate::ProfileSwitchGate() : active_(false), active_ssrc_(0) {}

void ProfileSwitchGate::reset() {
    active_ = false;
    active_ssrc_ = 0;
    active_key_ = ProfileKey();
}

GateAction ProfileSwitchGate::evaluate(const AccessUnit& access_unit) {
    if (!access_unit.profile.supported()) return GATE_DROP;
    const ProfileKey key(access_unit.profile);
    if (active_ && active_ssrc_ == access_unit.ssrc && active_key_ == key) {
        return GATE_FORWARD;
    }
    if (active_ && active_ssrc_ == access_unit.ssrc &&
        key.generation != active_key_.generation) {
        const uint8_t forward = static_cast<uint8_t>(key.generation - active_key_.generation);
        if (forward >= 128) return GATE_DROP;
    }
    if (!access_unit.completeRandomAccessPoint()) return GATE_DROP;
    active_ = true;
    active_ssrc_ = access_unit.ssrc;
    active_key_ = key;
    return GATE_RESTART_AND_FORWARD;
}

}  // namespace board_receiver
