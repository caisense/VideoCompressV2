#ifndef BOARD_RECEIVER_RTP_PACKET_H_
#define BOARD_RECEIVER_RTP_PACKET_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace board_receiver {

struct RtpPacket {
    std::vector<uint8_t> bytes;
    bool marker;
    uint8_t payload_type;
    uint16_t sequence;
    uint32_t timestamp;
    uint32_t ssrc;
    bool has_extension;
    uint16_t extension_profile;
    size_t extension_offset;
    size_t extension_size;
    size_t payload_offset;
    size_t payload_size;

    RtpPacket();

    const uint8_t* payload() const;
    const uint8_t* extensionData() const;
};

// Parses the same RTP subset consumed by tools/live_h265_hud.py. The receiver
// is intentionally video-only and rejects payload types other than dynamic
// H.265 PT 96.
bool parseRtpPacket(const uint8_t* data, size_t size, RtpPacket* output,
                    std::string* error);

uint8_t hevcNalType(const RtpPacket& packet);
bool isHevcIrap(uint8_t nal_type);

}  // namespace board_receiver

#endif  // BOARD_RECEIVER_RTP_PACKET_H_
