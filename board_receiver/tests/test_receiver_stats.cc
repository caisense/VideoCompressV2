#include <cmath>
#include <string>
#include <vector>

#include "receiver_stats.h"
#include "test_support.h"

int main() {
    board_receiver::ReceiverStats stats;
    board_receiver::StreamProfile profile;
    profile.id = board_receiver::PROFILE_RATE150;
    profile.width = 480;
    profile.height = 270;
    profile.fps = 15;
    profile.generation = 7;
    stats.setProfile(profile);
    stats.setLocalTxWireBps(123400);
    stats.recordPacket(1000);
    stats.recordAccessUnit(false);
    stats.recordAccessUnit(true);
    stats.recordDecodedFrame();
    stats.lost.store(3);
    stats.reordered.store(2);

    const board_receiver::ReceiverStatsSnapshot value = stats.snapshot(4);
    CHECK(value.profile.id == board_receiver::PROFILE_RATE150);
    CHECK(value.profile.width == 480 && value.profile.height == 270);
    CHECK(value.profile.fps == 15 && value.profile.generation == 7);
    CHECK(value.packets == 1 && value.packet_last_bytes == 1000);
    CHECK(value.packet_average_bytes == 1000.0 && value.packet_max_bytes == 1000);
    CHECK(std::fabs(value.rtp_kbps - 8.0) < 0.001);
    CHECK(std::fabs(value.wire_kbps - 8.528) < 0.001);
    CHECK(value.receive_fps == 2.0 && value.decode_fps == 1.0);
    CHECK(std::fabs(value.tx_wire_kbps - 123.4) < 0.001);
    CHECK(value.p_fps == 1.0 && value.i_fps == 1.0);
    CHECK(value.p_frames == 1 && value.i_frames == 1);
    CHECK(value.lost == 3 && value.reordered == 2 && value.decode_errors == 4);

    const std::vector<std::string> lines = stats.hudLines(4);
    CHECK(lines.size() == 5);
    CHECK(lines[0] == "TX 123.4  RX 8.5 kbps");
    CHECK(lines[1] == "P 1  I 1  TOTAL 2");
    CHECK(lines[2] == "PKT 1  LOSS 3  REO 2  ERR 4");
    CHECK(lines[3] == "PROFILE rate150 480x270@15  G7");
    CHECK(lines[4].find("SRC ") == 0);
    CHECK(lines[4].find("  IDR ") != std::string::npos);

    stats.setLocalTxWireBps(0);
    CHECK(stats.hudLines(0)[0] == "TX --  RX 8.5 kbps");
    return 0;
}
