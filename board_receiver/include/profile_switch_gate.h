#ifndef BOARD_RECEIVER_PROFILE_SWITCH_GATE_H_
#define BOARD_RECEIVER_PROFILE_SWITCH_GATE_H_

#include <stdint.h>

#include "access_unit_assembler.h"

namespace board_receiver {

enum GateAction {
    GATE_DROP,
    GATE_FORWARD,
    GATE_RESTART_AND_FORWARD,
};

class ProfileSwitchGate {
public:
    ProfileSwitchGate();

    GateAction evaluate(const AccessUnit& access_unit);
    void reset();
    bool active() const { return active_; }
    const ProfileKey& activeKey() const { return active_key_; }

private:
    bool active_;
    uint32_t active_ssrc_;
    ProfileKey active_key_;
};

}  // namespace board_receiver

#endif  // BOARD_RECEIVER_PROFILE_SWITCH_GATE_H_
