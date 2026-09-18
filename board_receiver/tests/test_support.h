#ifndef BOARD_RECEIVER_TEST_SUPPORT_H_
#define BOARD_RECEIVER_TEST_SUPPORT_H_

#include <stdint.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "rtp_packet.h"
#include "stream_profile.h"

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        std::cerr << __FILE__ << ':' << __LINE__                            \
                  << ": CHECK failed: " #condition << std::endl;           \
        return EXIT_FAILURE;                                                \
    }                                                                       \
} while (0)

inline void appendBe16(std::vector<uint8_t>* out, uint16_t value) {
    out->push_back(static_cast<uint8_t>(value >> 8));
    out->push_back(static_cast<uint8_t>(value));
}

inline void appendBe32(std::vector<uint8_t>* out, uint32_t value) {
    out->push_back(static_cast<uint8_t>(value >> 24));
    out->push_back(static_cast<uint8_t>(value >> 16));
    out->push_back(static_cast<uint8_t>(value >> 8));
    out->push_back(static_cast<uint8_t>(value));
}

inline std::vector<uint8_t> profileExtension(uint8_t profile, uint16_t width,
                                              uint16_t height, uint8_t fps,
                                              uint8_t generation) {
    std::vector<uint8_t> out;
    out.push_back(1);
    out.push_back(profile);
    appendBe16(&out, width);
    appendBe16(&out, height);
    out.push_back(fps);
    out.push_back(generation);
    appendBe16(&out, 80);
    appendBe16(&out, 100);
    return out;
}

inline board_receiver::RtpPacket makePacket(
        uint16_t sequence, uint32_t timestamp, bool marker,
        const std::vector<uint8_t>& payload, uint8_t profile = 0,
        uint16_t width = 320, uint16_t height = 180, uint8_t fps = 10,
        uint8_t generation = 1, uint32_t ssrc = 0x10203040) {
    std::vector<uint8_t> bytes;
    bytes.push_back(0x90);
    bytes.push_back(static_cast<uint8_t>(96 | (marker ? 0x80 : 0)));
    appendBe16(&bytes, sequence);
    appendBe32(&bytes, timestamp);
    appendBe32(&bytes, ssrc);
    appendBe16(&bytes, 0x524f);
    const std::vector<uint8_t> extension =
        profileExtension(profile, width, height, fps, generation);
    appendBe16(&bytes, 3);
    bytes.insert(bytes.end(), extension.begin(), extension.end());
    bytes.insert(bytes.end(), payload.begin(), payload.end());

    board_receiver::RtpPacket packet;
    std::string error;
    if (!board_receiver::parseRtpPacket(bytes.data(), bytes.size(), &packet, &error)) {
        std::cerr << "test packet parse failed: " << error << std::endl;
        std::abort();
    }
    return packet;
}

inline board_receiver::StreamProfile packetProfile(
        const board_receiver::RtpPacket& packet) {
    board_receiver::StreamProfile profile;
    std::string error;
    if (board_receiver::parseStreamProfile(packet, &profile, &error) ==
        board_receiver::PROFILE_INVALID) {
        std::cerr << "test profile parse failed: " << error << std::endl;
        std::abort();
    }
    return profile;
}

inline std::vector<uint8_t> nal(uint8_t type, uint8_t body) {
    std::vector<uint8_t> out;
    out.push_back(static_cast<uint8_t>(type << 1));
    out.push_back(1);
    out.push_back(body);
    return out;
}

#endif
