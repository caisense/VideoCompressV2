#include "rtp_packet.h"

#include <limits>

namespace board_receiver {
namespace {

uint16_t readBe16(const uint8_t* data) {
    return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8) |
                                 static_cast<uint16_t>(data[1]));
}

uint32_t readBe32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

bool fail(const char* message, std::string* error) {
    if (error) *error = message;
    return false;
}

}  // namespace

RtpPacket::RtpPacket()
    : marker(false), payload_type(0), sequence(0), timestamp(0), ssrc(0),
      has_extension(false), extension_profile(0), extension_offset(0),
      extension_size(0), payload_offset(0), payload_size(0) {}

const uint8_t* RtpPacket::payload() const {
    return payload_offset <= bytes.size() ? bytes.data() + payload_offset : NULL;
}

const uint8_t* RtpPacket::extensionData() const {
    return has_extension && extension_offset <= bytes.size()
        ? bytes.data() + extension_offset : NULL;
}

bool parseRtpPacket(const uint8_t* data, size_t size, RtpPacket* output,
                    std::string* error) {
    if (!data || !output) return fail("null RTP input or output", error);
    if (size < 12) return fail("truncated RTP fixed header", error);
    if ((data[0] >> 6) != 2) return fail("RTP version is not 2", error);

    const uint8_t payload_type = static_cast<uint8_t>(data[1] & 0x7f);
    if (payload_type != 96) return fail("RTP payload type is not H.265 PT 96", error);

    const size_t csrc_count = data[0] & 0x0f;
    if (csrc_count > (std::numeric_limits<size_t>::max() - 12) / 4) {
        return fail("RTP CSRC count overflows size", error);
    }
    size_t offset = 12 + csrc_count * 4;
    if (offset > size) return fail("truncated RTP CSRC list", error);

    RtpPacket parsed;
    parsed.bytes.assign(data, data + size);
    parsed.marker = (data[1] & 0x80) != 0;
    parsed.payload_type = payload_type;
    parsed.sequence = readBe16(data + 2);
    parsed.timestamp = readBe32(data + 4);
    parsed.ssrc = readBe32(data + 8);

    parsed.has_extension = (data[0] & 0x10) != 0;
    if (parsed.has_extension) {
        if (offset + 4 > size) return fail("truncated RTP extension header", error);
        parsed.extension_profile = readBe16(data + offset);
        const size_t extension_words = readBe16(data + offset + 2);
        if (extension_words > (std::numeric_limits<size_t>::max() - offset - 4) / 4) {
            return fail("RTP extension length overflows size", error);
        }
        parsed.extension_offset = offset + 4;
        parsed.extension_size = extension_words * 4;
        offset = parsed.extension_offset + parsed.extension_size;
        if (offset > size) return fail("truncated RTP extension payload", error);
    }

    size_t payload_end = size;
    if (data[0] & 0x20) {
        const size_t padding = data[size - 1];
        if (padding == 0 || padding > payload_end - offset) {
            return fail("invalid RTP padding", error);
        }
        payload_end -= padding;
    }
    if (payload_end <= offset) return fail("empty RTP H.265 payload", error);

    parsed.payload_offset = offset;
    parsed.payload_size = payload_end - offset;
    *output = parsed;
    if (error) error->clear();
    return true;
}

uint8_t hevcNalType(const RtpPacket& packet) {
    if (!packet.payload() || packet.payload_size < 2) return 0xff;
    uint8_t nal_type = static_cast<uint8_t>((packet.payload()[0] >> 1) & 0x3f);
    if (nal_type == 49) {
        if (packet.payload_size < 3) return 0xff;
        if ((packet.payload()[2] & 0x80) == 0) return 0xff;
        nal_type = static_cast<uint8_t>(packet.payload()[2] & 0x3f);
    }
    return nal_type;
}

bool isHevcIrap(uint8_t nal_type) {
    return nal_type >= 16 && nal_type <= 21;
}

}  // namespace board_receiver
