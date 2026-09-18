#include "test_support.h"

#include "rtp_reorder_buffer.h"

namespace {
std::vector<uint8_t> fixedHeader(uint8_t first, uint8_t second) {
    std::vector<uint8_t> bytes;
    bytes.push_back(first); bytes.push_back(second);
    appendBe16(&bytes, 7); appendBe32(&bytes, 8); appendBe32(&bytes, 9);
    return bytes;
}
}

int main() {
    using namespace board_receiver;
    const RtpPacket packet = makePacket(65535, 90000, true, nal(19, 0xaa),
                                        PROFILE_RATE300, 640, 360, 20, 7);
    CHECK(packet.marker);
    CHECK(packet.sequence == 65535);
    CHECK(packet.timestamp == 90000);
    CHECK(packet.payload_type == 96);
    CHECK(packet.extension_profile == 0x524f);
    CHECK(packet.payload_size == 3);

    StreamProfile profile;
    std::string error;
    CHECK(parseStreamProfile(packet, &profile, &error) == PROFILE_SUPPORTED);
    CHECK(profile.id == PROFILE_RATE300);
    CHECK(profile.width == 640 && profile.height == 360);
    CHECK(profile.fps == 20 && profile.generation == 7);

    const RtpPacket unsupported = makePacket(1, 2, true, nal(1, 0),
                                             PROFILE_REBUILD, 640, 360, 20, 1);
    CHECK(parseStreamProfile(unsupported, &profile, &error) == PROFILE_UNSUPPORTED);

    std::vector<uint8_t> invalid(12, 0);
    invalid[0] = 0x40;
    invalid[1] = 96;
    RtpPacket invalid_packet;
    CHECK(!parseRtpPacket(invalid.data(), invalid.size(), &invalid_packet, &error));

    std::vector<uint8_t> wrong_pt = fixedHeader(0x80, 97);
    wrong_pt.insert(wrong_pt.end(), 2, 1);
    CHECK(!parseRtpPacket(wrong_pt.data(), wrong_pt.size(), &invalid_packet, &error));

    std::vector<uint8_t> no_extension = fixedHeader(0x80, 96);
    no_extension.push_back(2); no_extension.push_back(1);
    CHECK(parseRtpPacket(no_extension.data(), no_extension.size(), &invalid_packet, &error));
    CHECK(invalid_packet.payload_offset == 12 && !invalid_packet.has_extension);
    CHECK(parseStreamProfile(invalid_packet, &profile, &error) == PROFILE_MISSING);

    std::vector<uint8_t> csrc = fixedHeader(0x81, 96);
    appendBe32(&csrc, 0xaabbccdd); csrc.push_back(2); csrc.push_back(1);
    CHECK(parseRtpPacket(csrc.data(), csrc.size(), &invalid_packet, &error));
    CHECK(invalid_packet.payload_offset == 16);

    std::vector<uint8_t> extension8 = fixedHeader(0x90, 96);
    appendBe16(&extension8, 0x524f); appendBe16(&extension8, 2);
    const std::vector<uint8_t> low8 = profileExtension(PROFILE_RATE60, 320, 180, 10, 3);
    extension8.insert(extension8.end(), low8.begin(), low8.begin() + 8);
    extension8.push_back(2); extension8.push_back(1);
    CHECK(parseRtpPacket(extension8.data(), extension8.size(), &invalid_packet, &error));
    CHECK(invalid_packet.payload_offset == 24);
    CHECK(parseStreamProfile(invalid_packet, &profile, &error) == PROFILE_SUPPORTED);

    std::vector<uint8_t> truncated = fixedHeader(0x90, 96);
    appendBe16(&truncated, 0x524f); appendBe16(&truncated, 3);
    truncated.insert(truncated.end(), 4, 0);
    CHECK(!parseRtpPacket(truncated.data(), truncated.size(), &invalid_packet, &error));

    std::vector<uint8_t> padded = fixedHeader(0xa0, 96);
    padded.push_back(2); padded.push_back(1); padded.push_back(0); padded.push_back(2);
    CHECK(parseRtpPacket(padded.data(), padded.size(), &invalid_packet, &error));
    CHECK(invalid_packet.payload_size == 2);
    padded.back() = 0;
    CHECK(!parseRtpPacket(padded.data(), padded.size(), &invalid_packet, &error));

    const uint8_t supported_profiles[] = {
        PROFILE_RATE60, PROFILE_RATE80, PROFILE_RATE100, PROFILE_RATE120,
        PROFILE_RATE150, PROFILE_RATE180, PROFILE_RATE200, PROFILE_RATE300};
    const char* supported_names[] = {
        "rate60", "rate80", "rate100", "rate120",
        "rate150", "rate180", "rate200", "rate300"};
    for (size_t i = 0; i < sizeof(supported_profiles) / sizeof(supported_profiles[0]); ++i) {
        const RtpPacket p = makePacket(static_cast<uint16_t>(20 + i), 2, true,
                                       nal(1, 0), supported_profiles[i]);
        CHECK(parseStreamProfile(p, &profile, &error) == PROFILE_SUPPORTED);
        CHECK(std::string(profile.name()) == supported_names[i]);
    }
    const uint8_t rejected_profiles[] = {PROFILE_REBUILD, PROFILE_GAN, 99};
    for (size_t i = 0; i < 3; ++i) {
        const RtpPacket p = makePacket(static_cast<uint16_t>(30 + i), 2, true,
                                       nal(1, 0), rejected_profiles[i]);
        CHECK(parseStreamProfile(p, &profile, &error) == PROFILE_UNSUPPORTED);
    }

    RtpReorderBuffer reorder(2);
    ReorderResult result = reorder.push(makePacket(65535, 1, false, nal(1, 1)));
    CHECK(result.ready.size() == 1);
    result = reorder.push(makePacket(1, 1, true, nal(1, 3)));
    CHECK(result.ready.empty());
    result = reorder.push(makePacket(0, 1, false, nal(1, 2)));
    CHECK(result.ready.size() == 2);
    CHECK(result.ready[0].sequence == 0 && result.ready[1].sequence == 1);
    return EXIT_SUCCESS;
}
