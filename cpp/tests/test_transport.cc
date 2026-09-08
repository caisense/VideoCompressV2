#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "transport/packetizer.h"
#include "transport/codec2_rtp_packetizer.h"
#include "transport/rate_pacer.h"

namespace {
int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        ++failures; \
    } \
} while (0)

void testH265RtpPacketization() {
    std::vector<uint8_t> access_unit;
    const uint8_t first[] = {0, 0, 0, 1, 0x02, 0x01, 0xaa, 0xbb};
    access_unit.insert(access_unit.end(), first, first + sizeof(first));
    access_unit.push_back(0);
    access_unit.push_back(0);
    access_unit.push_back(1);
    access_unit.push_back(0x26);  // H.265 NAL type 19 (IDR)
    access_unit.push_back(0x01);
    for (int i = 0; i < 80; ++i) access_unit.push_back(static_cast<uint8_t>(i));

    roi_h265::H265RtpPacketizer packetizer(7, 0x12345678U, 40);
    const std::vector<std::vector<uint8_t> > packets = packetizer.packetize(
        access_unit.data(), access_unit.size(), 90000);
    CHECK(packets.size() > 2);
    CHECK((packets.front()[0] >> 6) == 2);
    CHECK((packets.front()[1] & 0x7f) == 96);
    CHECK((packets.front()[1] & 0x80) == 0);
    CHECK((packets.back()[1] & 0x80) != 0);
    bool saw_fu_start = false;
    bool saw_fu_end = false;
    std::vector<uint8_t> reassembled;
    for (size_t i = 0; i < packets.size(); ++i) {
        CHECK(packets[i].size() <= 40);
        CHECK((static_cast<uint16_t>(packets[i][2]) << 8 | packets[i][3]) ==
              static_cast<uint16_t>(7 + i));
        CHECK((static_cast<uint32_t>(packets[i][4]) << 24 | static_cast<uint32_t>(packets[i][5]) << 16 |
               static_cast<uint32_t>(packets[i][6]) << 8 | packets[i][7]) == 90000U);
        CHECK((static_cast<uint32_t>(packets[i][8]) << 24 | static_cast<uint32_t>(packets[i][9]) << 16 |
               static_cast<uint32_t>(packets[i][10]) << 8 | packets[i][11]) == 0x12345678U);
        if (packets[i].size() > 15 && ((packets[i][12] >> 1) & 0x3f) == 49) {
            const bool start = (packets[i][14] & 0x80) != 0;
            const bool end = (packets[i][14] & 0x40) != 0;
            saw_fu_start = saw_fu_start || start;
            saw_fu_end = saw_fu_end || end;
            if (start) {
                reassembled.insert(reassembled.end(), {0, 0, 0, 1});
                reassembled.push_back(static_cast<uint8_t>((packets[i][12] & 0x81) |
                                                           ((packets[i][14] & 0x3f) << 1)));
                reassembled.push_back(packets[i][13]);
            }
            reassembled.insert(reassembled.end(), packets[i].begin() + 15, packets[i].end());
        } else {
            reassembled.insert(reassembled.end(), {0, 0, 0, 1});
            reassembled.insert(reassembled.end(), packets[i].begin() + 12, packets[i].end());
        }
    }
    CHECK(saw_fu_start && saw_fu_end);
    std::vector<uint8_t> normalized;
    normalized.insert(normalized.end(), {0, 0, 0, 1, 0x02, 0x01, 0xaa, 0xbb});
    normalized.insert(normalized.end(), {0, 0, 0, 1, 0x26, 0x01});
    for (int i = 0; i < 80; ++i) normalized.push_back(static_cast<uint8_t>(i));
    CHECK(reassembled == normalized);
}

void testRatePacerStartsWithoutLargeBurst() {
    roi_h265::RatePacer pacer(8000, 16);
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    pacer.waitForTokens(1);
    const long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
    CHECK(pacer.bitrateBps() == 8000);
    CHECK(elapsed_us >= 200);
    CHECK(elapsed_us < 100000);
}

void testAudioPacketPacerKeepsOnePacketPer80Ms() {
    // Codec2-1300 with two 40 ms frames is 26 B of UDP payload and costs
    // 92 physical Ethernet-wire bytes. Its reserved lane is 9.2 kbps, so four
    // consecutive packets must take roughly 320 ms instead of accumulating a
    // multi-second burst or delay.
    roi_h265::RatePacer pacer(9200, 92);
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int packet = 0; packet < 4; ++packet) pacer.waitForTokens(92);
    const long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
    CHECK(elapsed_us >= 260000);
    CHECK(elapsed_us < 800000);
}

void testPhysicalWireAccountingAndCodec2Packet() {
    CHECK(roi_h265::RatePacer::wireBytesForUdpPayload(20) == 86U);
    CHECK(roi_h265::RatePacer::wireBytesForUdpPayload(1200) == 1266U);

    const uint8_t codec2_payload[] = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e};
    roi_h265::Codec2RtpPacketizer packetizer(9, 0x41424344U, 97);
    const std::vector<uint8_t> packet = packetizer.packetize(codec2_payload,
        sizeof(codec2_payload), 640U);
    CHECK(packet.size() == sizeof(codec2_payload) + 12U);
    CHECK(packet[0] == 0x80);
    CHECK(packet[1] == 0xe1);  // marker + dynamic Codec2 payload type 97
    CHECK((static_cast<uint16_t>(packet[2]) << 8 | packet[3]) == 9U);
    CHECK((static_cast<uint32_t>(packet[4]) << 24 | static_cast<uint32_t>(packet[5]) << 16 |
           static_cast<uint32_t>(packet[6]) << 8 | packet[7]) == 640U);
    CHECK((static_cast<uint32_t>(packet[8]) << 24 | static_cast<uint32_t>(packet[9]) << 16 |
           static_cast<uint32_t>(packet[10]) << 8 | packet[11]) == 0x41424344U);
    CHECK(std::memcmp(packet.data() + 12, codec2_payload, sizeof(codec2_payload)) == 0);
}

void testProfileMetadataUsesRtpHeaderExtension() {
    const uint8_t access_unit[] = {0, 0, 0, 1, 0x26, 0x01, 0xaa};
    roi_h265::RtpStreamProfile profile;
    profile.valid = true;
    profile.profile = 4;
    profile.width = 320;
    profile.height = 180;
    profile.fps = 10;
    profile.generation = 7;
    profile.target_bitrate_kbps = 110;
    profile.link_cap_kbps = 150;
    roi_h265::H265RtpPacketizer packetizer(1, 2, 1200);
    const std::vector<std::vector<uint8_t> > packets = packetizer.packetize(
        access_unit, sizeof(access_unit), 1234, &profile);
    CHECK(packets.size() == 1);
    CHECK((packets[0][0] & 0x10) != 0);
    CHECK(packets[0][12] == 0x52 && packets[0][13] == 0x4f);
    CHECK(packets[0][14] == 0 && packets[0][15] == 3);
    CHECK(packets[0][16] == 1 && packets[0][17] == 4);
    CHECK((static_cast<int>(packets[0][18]) << 8 | packets[0][19]) == 320);
    CHECK((static_cast<int>(packets[0][20]) << 8 | packets[0][21]) == 180);
    CHECK(packets[0][22] == 10 && packets[0][23] == 7);
    CHECK((static_cast<int>(packets[0][24]) << 8 | packets[0][25]) == 110);
    CHECK((static_cast<int>(packets[0][26]) << 8 | packets[0][27]) == 150);
    CHECK(((packets[0][28] >> 1) & 0x3f) == 19);

    // The optional tail must reduce the H.265 payload budget by the same four
    // bytes, rather than let a fragmented RTP datagram exceed the caller MTU.
    std::vector<uint8_t> large_access_unit = {0, 0, 0, 1, 0x26, 0x01};
    for (int i = 0; i < 48; ++i) large_access_unit.push_back(static_cast<uint8_t>(i));
    roi_h265::H265RtpPacketizer constrained_packetizer(1, 2, 40);
    const std::vector<std::vector<uint8_t> > constrained_packets =
        constrained_packetizer.packetize(large_access_unit.data(), large_access_unit.size(),
                                         1234, &profile);
    CHECK(constrained_packets.size() > 1);
    for (size_t i = 0; i < constrained_packets.size(); ++i) {
        CHECK(constrained_packets[i].size() <= 40U);
        CHECK(((constrained_packets[i][28] >> 1) & 0x3f) == 49);
    }

    // Packetizer fields are network-order uint16 values.  Use non-preset
    // values here so both bytes are asserted independently; validation of
    // permitted CAP values belongs to the config test above.
    profile.target_bitrate_kbps = 0x1234U;
    profile.link_cap_kbps = 0x4567U;
    roi_h265::H265RtpPacketizer byte_order_packetizer(1, 2, 1200);
    const std::vector<std::vector<uint8_t> > byte_order_packets =
        byte_order_packetizer.packetize(access_unit, sizeof(access_unit), 1234, &profile);
    CHECK(byte_order_packets.size() == 1);
    CHECK(byte_order_packets[0][24] == 0x12 && byte_order_packets[0][25] == 0x34);
    CHECK(byte_order_packets[0][26] == 0x45 && byte_order_packets[0][27] == 0x67);

    // Current target-aware senders used the same tail with the two CAP bytes
    // reserved as zero.  Keep that twelve-byte layout valid for receivers
    // that now interpret CAP=0 as the historical 100 kbps GAN preset.
    profile.profile = 4;
    profile.width = 256;
    profile.height = 144;
    profile.fps = 8;
    profile.target_bitrate_kbps = 75;
    profile.link_cap_kbps = 0;
    roi_h265::H265RtpPacketizer current_gan_packetizer(1, 2, 1200);
    const std::vector<std::vector<uint8_t> > current_gan_packets =
        current_gan_packetizer.packetize(access_unit, sizeof(access_unit), 1234, &profile);
    CHECK(current_gan_packets.size() == 1);
    CHECK(current_gan_packets[0][14] == 0 && current_gan_packets[0][15] == 3);
    CHECK((static_cast<int>(current_gan_packets[0][24]) << 8 |
           current_gan_packets[0][25]) == 75);
    CHECK(current_gan_packets[0][26] == 0 && current_gan_packets[0][27] == 0);

    // Zero TARGET and CAP keep non-GAN profiles byte-for-byte compatible with
    // the original two-word RO extension.
    profile.profile = 1;
    profile.width = 480;
    profile.height = 270;
    profile.fps = 15;
    profile.target_bitrate_kbps = 0;
    profile.link_cap_kbps = 0;
    roi_h265::H265RtpPacketizer legacy_packetizer(1, 2, 1200);
    const std::vector<std::vector<uint8_t> > legacy_packets = legacy_packetizer.packetize(
        access_unit, sizeof(access_unit), 1234, &profile);
    CHECK(legacy_packets.size() == 1);
    CHECK(legacy_packets[0][14] == 0 && legacy_packets[0][15] == 2);
    CHECK(legacy_packets[0][16] == 1 && legacy_packets[0][17] == 1);
    CHECK(((legacy_packets[0][24] >> 1) & 0x3f) == 19);
}

}  // namespace

int main() {
    testH265RtpPacketization();
    testRatePacerStartsWithoutLargeBurst();
    testAudioPacketPacerKeepsOnePacketPer80Ms();
    testPhysicalWireAccountingAndCodec2Packet();
    testProfileMetadataUsesRtpHeaderExtension();
    if (failures) return EXIT_FAILURE;
    std::cout << "transport tests passed\n";
    return EXIT_SUCCESS;
}
