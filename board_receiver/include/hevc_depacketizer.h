#ifndef BOARD_RECEIVER_HEVC_DEPACKETIZER_H_
#define BOARD_RECEIVER_HEVC_DEPACKETIZER_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "rtp_packet.h"

namespace board_receiver {

struct DepacketizedPayload {
    bool valid;
    bool fu_in_progress;
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> nal_types;
    std::string error;

    DepacketizedPayload();
};

class HevcRtpDepacketizer {
public:
    HevcRtpDepacketizer();

    DepacketizedPayload feed(const RtpPacket& packet);
    void reset();
    bool fuInProgress() const { return fu_active_; }

private:
    static void appendStartCode(std::vector<uint8_t>* output);
    DepacketizedPayload invalid(const char* message);

    bool fu_active_;
    uint16_t fu_expected_sequence_;
    uint8_t fu_nal_type_;
};

}  // namespace board_receiver

#endif  // BOARD_RECEIVER_HEVC_DEPACKETIZER_H_
