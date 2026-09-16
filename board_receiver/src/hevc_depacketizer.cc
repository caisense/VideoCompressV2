#include "hevc_depacketizer.h"

namespace board_receiver {

DepacketizedPayload::DepacketizedPayload()
    : valid(false), fu_in_progress(false) {}

HevcRtpDepacketizer::HevcRtpDepacketizer()
    : fu_active_(false), fu_expected_sequence_(0), fu_nal_type_(0xff) {}

void HevcRtpDepacketizer::appendStartCode(std::vector<uint8_t>* output) {
    static const uint8_t kStartCode[] = {0, 0, 0, 1};
    output->insert(output->end(), kStartCode, kStartCode + sizeof(kStartCode));
}

DepacketizedPayload HevcRtpDepacketizer::invalid(const char* message) {
    reset();
    DepacketizedPayload result;
    result.error = message;
    return result;
}

void HevcRtpDepacketizer::reset() {
    fu_active_ = false;
    fu_expected_sequence_ = 0;
    fu_nal_type_ = 0xff;
}

DepacketizedPayload HevcRtpDepacketizer::feed(const RtpPacket& packet) {
    const uint8_t* payload = packet.payload();
    if (!payload || packet.payload_size < 2) return invalid("truncated HEVC RTP payload");

    DepacketizedPayload result;
    const uint8_t nal_type = static_cast<uint8_t>((payload[0] >> 1) & 0x3f);
    if (nal_type == 48) {
        reset();
        size_t cursor = 2;
        while (cursor + 2 <= packet.payload_size) {
            const size_t nal_size = (static_cast<size_t>(payload[cursor]) << 8) |
                                    static_cast<size_t>(payload[cursor + 1]);
            cursor += 2;
            if (nal_size < 2 || cursor + nal_size > packet.payload_size) {
                return invalid("invalid RFC 7798 aggregation packet length");
            }
            appendStartCode(&result.bytes);
            result.nal_types.push_back(
                static_cast<uint8_t>((payload[cursor] >> 1) & 0x3f));
            result.bytes.insert(result.bytes.end(), payload + cursor,
                                payload + cursor + nal_size);
            cursor += nal_size;
        }
        if (cursor != packet.payload_size || result.nal_types.empty()) {
            return invalid("truncated or empty RFC 7798 aggregation packet");
        }
        result.valid = true;
        return result;
    }

    if (nal_type != 49) {
        reset();
        appendStartCode(&result.bytes);
        result.bytes.insert(result.bytes.end(), payload, payload + packet.payload_size);
        result.nal_types.push_back(nal_type);
        result.valid = true;
        return result;
    }

    if (packet.payload_size < 4) return invalid("truncated RFC 7798 fragmentation unit");
    const uint8_t fu_header = payload[2];
    const bool start = (fu_header & 0x80) != 0;
    const bool end = (fu_header & 0x40) != 0;
    const uint8_t original_type = static_cast<uint8_t>(fu_header & 0x3f);

    if (start) {
        reset();
        appendStartCode(&result.bytes);
        const uint8_t first_byte = static_cast<uint8_t>(
            (payload[0] & 0x81) | (original_type << 1));
        result.bytes.push_back(first_byte);
        result.bytes.push_back(payload[1]);
        result.bytes.insert(result.bytes.end(), payload + 3,
                            payload + packet.payload_size);
        result.nal_types.push_back(original_type);
        if (!end) {
            fu_active_ = true;
            fu_expected_sequence_ = static_cast<uint16_t>(packet.sequence + 1);
            fu_nal_type_ = original_type;
        }
        result.valid = true;
        result.fu_in_progress = fu_active_;
        return result;
    }

    if (!fu_active_ || packet.sequence != fu_expected_sequence_ ||
        original_type != fu_nal_type_) {
        return invalid("RFC 7798 fragmentation sequence discontinuity");
    }
    result.bytes.insert(result.bytes.end(), payload + 3,
                        payload + packet.payload_size);
    result.valid = true;
    if (end) {
        reset();
    } else {
        fu_expected_sequence_ = static_cast<uint16_t>(packet.sequence + 1);
    }
    result.fu_in_progress = fu_active_;
    return result;
}

}  // namespace board_receiver
