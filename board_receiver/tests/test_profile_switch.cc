#include "test_support.h"

#include "profile_switch_gate.h"

board_receiver::AccessUnit accessUnit(uint8_t profile_id, uint8_t generation,
                                      bool random_access, uint32_t ssrc = 1) {
    board_receiver::AccessUnit unit;
    unit.ssrc = ssrc;
    unit.profile.id = profile_id;
    unit.profile.generation = generation;
    unit.profile.width = profile_id == board_receiver::PROFILE_LOW ? 320 :
                         profile_id == board_receiver::PROFILE_MEDIUM ? 480 : 640;
    unit.profile.height = profile_id == board_receiver::PROFILE_LOW ? 180 :
                          profile_id == board_receiver::PROFILE_MEDIUM ? 270 : 360;
    unit.profile.fps = profile_id == board_receiver::PROFILE_LOW ? 10 :
                       profile_id == board_receiver::PROFILE_MEDIUM ? 15 : 20;
    unit.bytes.push_back(1);
    unit.has_vps = unit.has_sps = unit.has_pps = unit.has_irap = random_access;
    return unit;
}

int main() {
    using namespace board_receiver;
    ProfileSwitchGate gate;
    CHECK(gate.evaluate(accessUnit(PROFILE_LOW, 1, false)) == GATE_DROP);
    CHECK(gate.evaluate(accessUnit(PROFILE_LOW, 1, true)) == GATE_RESTART_AND_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_LOW, 1, false)) == GATE_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_HIGH, 2, false)) == GATE_DROP);
    CHECK(gate.evaluate(accessUnit(PROFILE_HIGH, 2, true)) == GATE_RESTART_AND_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_LOW, 1, true)) == GATE_DROP);
    CHECK(gate.evaluate(accessUnit(PROFILE_REBUILD, 3, true)) == GATE_DROP);
    AccessUnit geometry = accessUnit(PROFILE_HIGH, 2, true);
    geometry.profile.width = 608;
    CHECK(gate.evaluate(geometry) == GATE_RESTART_AND_FORWARD);
    CHECK(gate.evaluate(accessUnit(PROFILE_HIGH, 2, true, 2)) == GATE_RESTART_AND_FORWARD);
    return EXIT_SUCCESS;
}
