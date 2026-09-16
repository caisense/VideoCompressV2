#include "test_support.h"

#include "hevc_depacketizer.h"
#include "transport/packetizer.h"

#include <iomanip>
#include <sstream>

namespace {
std::string hex(const uint8_t* data, size_t size) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) out << std::setw(2) << static_cast<int>(data[i]);
    return out.str();
}
}

int main(int argc, char**) {
    using namespace board_receiver;
    std::vector<uint8_t> annex_b;
    const uint8_t types[] = {32, 33, 34, 19};
    for (size_t index = 0; index < 4; ++index) {
        annex_b.insert(annex_b.end(), 4, 0);
        annex_b.back() = 1;
        const std::vector<uint8_t> current = nal(types[index],
            static_cast<uint8_t>(0xa0 + index));
        annex_b.insert(annex_b.end(), current.begin(), current.end());
        if (types[index] == 19) annex_b.insert(annex_b.end(), 100, 0x5a);
    }

    roi_h265::RtpStreamProfile sender_profile;
    sender_profile.valid = true;
    sender_profile.profile = PROFILE_MEDIUM;
    sender_profile.width = 480;
    sender_profile.height = 270;
    sender_profile.fps = 15;
    sender_profile.generation = 9;
    roi_h265::H265RtpPacketizer packetizer(65533, 0x12345678, 48);
    const std::vector<std::vector<uint8_t> > datagrams = packetizer.packetize(
        annex_b.data(), annex_b.size(), 90000, &sender_profile);
    CHECK(datagrams.size() > 4);

    HevcRtpDepacketizer receiver;
    std::vector<uint8_t> reconstructed;
    for (size_t index = 0; index < datagrams.size(); ++index) {
        RtpPacket packet;
        std::string error;
        CHECK(parseRtpPacket(datagrams[index].data(), datagrams[index].size(),
                             &packet, &error));
        StreamProfile parsed;
        CHECK(parseStreamProfile(packet, &parsed, &error) == PROFILE_SUPPORTED);
        CHECK(parsed.id == PROFILE_MEDIUM && parsed.width == 480 &&
              parsed.height == 270 && parsed.fps == 15 && parsed.generation == 9);
        const DepacketizedPayload output = receiver.feed(packet);
        CHECK(output.valid);
        if (argc > 1) {
            std::cout << hex(datagrams[index].data(), datagrams[index].size()) << '|'
                      << packet.payload_offset << '|'
                      << static_cast<int>(parsed.id) << ',' << parsed.width << ','
                      << parsed.height << ',' << static_cast<int>(parsed.fps) << ','
                      << static_cast<int>(parsed.generation) << '|'
                      << static_cast<int>(hevcNalType(packet)) << '|'
                      << hex(output.bytes.data(), output.bytes.size()) << std::endl;
        }
        reconstructed.insert(reconstructed.end(), output.bytes.begin(), output.bytes.end());
    }
    CHECK(reconstructed == annex_b);
    return EXIT_SUCCESS;
}
