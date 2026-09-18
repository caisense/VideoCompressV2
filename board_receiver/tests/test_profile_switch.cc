#include "test_support.h"

#include "profile_switch_gate.h"

board_receiver::AccessUnit accessUnit(uint8_t profile_id, uint8_t generation,
                                      bool random_access, uint32_t ssrc = 1) {
    board_receiver::AccessUnit unit;
    unit.ssrc = ssrc;
    unit.profile.id = profile_id;
    unit.profile.generation = generation;
    unit.profile.width = 320;
    unit.profile.height = 180;
    unit.profile.fps = 10;
    if (profile_id == board_receiver::PROFILE_RATE100) {
        unit.profile.width = 384; unit.profile.height = 216;
    } else if (profile_id == board_receiver::PROFILE_RATE120 ||
               profile_id == board_receiver::PROFILE_RATE150) {
        unit.profile.width = 480; unit.profile.height = 270;
        unit.profile.fps = profile_id == board_receiver::PROFILE_RATE150 ? 15 : 10;
    } else if (profile_id == board_receiver::PROFILE_RATE180 ||
               profile_id == board_receiver::PROFILE_RATE200) {
        unit.profile.width = 512; unit.profile.height = 288;
        unit.profile.fps = profile_id == board_receiver::PROFILE_RATE200 ? 18 : 15;
    } else if (profile_id == board_receiver::PROFILE_RATE300) {
        unit.profile.width = 640; unit.profile.height = 360; unit.profile.fps = 20;
    }
    unit.bytes.push_back(1);
    unit.has_vps = unit.has_sps = unit.has_pps = unit.has_irap = random_access;
    return unit;
}

int main() {
    using namespace board_receiver;
    ProfileSwitchGate gate;
    CHECK(gate.evaluate(accessUnit(PROFILE_RATE60, 1, false)) == GATE_DROP);
    CHECK(gate.evaluate(accessUnit(PROFILE_RATE60, 1, true)) == GATE_RESTART_AND_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_RATE60, 1, false)) == GATE_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_RATE300, 2, false)) == GATE_DROP);
    CHECK(gate.evaluate(accessUnit(PROFILE_RATE300, 2, true)) == GATE_RESTART_AND_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_RATE60, 1, true)) == GATE_DROP);
    CHECK(gate.evaluate(accessUnit(PROFILE_REBUILD, 3, true)) == GATE_DROP);
    const uint8_t supported[] = {PROFILE_RATE60, PROFILE_RATE80, PROFILE_RATE100,
        PROFILE_RATE120, PROFILE_RATE150, PROFILE_RATE180, PROFILE_RATE200,
        PROFILE_RATE300};
    for (size_t index = 0; index < sizeof(supported) / sizeof(supported[0]); ++index) {
        CHECK(accessUnit(supported[index], static_cast<uint8_t>(10 + index), true)
                  .profile.supported());
    }
    CHECK(!accessUnit(PROFILE_GAN, 20, true).profile.supported());
    AccessUnit geometry = accessUnit(PROFILE_RATE300, 2, true);
    geometry.profile.width = 608;
    CHECK(gate.evaluate(geometry) == GATE_RESTART_AND_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_RATE300, 2, true, 2)) == GATE_RESTART_AND_FORWARD);
    return EXIT_SUCCESS;
}
