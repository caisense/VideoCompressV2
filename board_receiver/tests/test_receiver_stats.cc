#include <cmath>
#include <string>
#include <vector>

#include "receiver_stats.h"
#include "test_support.h"

int main() {
    board_receiver::ReceiverStats stats;
    board_receiver::StreamProfile profile;
    profile.id = board_receiver::PROFILE_MEDIUM;
    profile.width = 480;
    profile.height = 270;
    profile.fps = 15;
    profile.generation = 7;
    stats.setProfile(profile);
    stats.recordPacket(1000);
    stats.recordAccessUnit(false);
    stats.recordAccessUnit(true);
    stats.recordDecodedFrame();
    stats.lost.store(3);
    stats.reordered.store(2);

    const board_receiver::ReceiverStatsSnapshot value = stats.snapshot(4);
    CHECK(value.profile.id == board_receiver::PROFILE_MEDIUM);
    CHECK(value.profile.width == 480 && value.profile.height == 270);
    CHECK(value.profile.fps == 15 && value.profile.generation == 7);
    CHECK(value.packets == 1 && value.packet_last_bytes == 1000);
    CHECK(value.packet_average_bytes == 1000.0 && value.packet_max_bytes == 1000);
    CHECK(std::fabs(value.rtp_kbps - 8.0) < 0.001);
    CHECK(std::fabs(value.wire_kbps - 8.528) < 0.001);
    CHECK(value.receive_fps == 2.0 && value.decode_fps == 1.0);
    CHECK(value.p_fps == 1.0 && value.i_fps == 1.0);
    CHECK(value.p_frames == 1 && value.i_frames == 1);
    CHECK(value.lost == 3 && value.reordered == 2 && value.decode_errors == 4);

    const std::vector<std::string> lines = stats.hudLines(4);
    CHECK(lines.size() == 7);
    CHECK(lines[0].find("RX 2.0") != std::string::npos);
    CHECK(lines[1].find("RTP 8.0") != std::string::npos);
    CHECK(lines[5].find("PROFILE medium 480x270@15  G7") != std::string::npos);
    return 0;
}
