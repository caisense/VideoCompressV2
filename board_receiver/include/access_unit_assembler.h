#ifndef BOARD_RECEIVER_ACCESS_UNIT_ASSEMBLER_H_
#define BOARD_RECEIVER_ACCESS_UNIT_ASSEMBLER_H_

#include <stdint.h>

#include <vector>

#include "hevc_depacketizer.h"
#include "stream_profile.h"

namespace board_receiver {

struct AccessUnit {
    uint32_t timestamp;
    uint32_t ssrc;
    StreamProfile profile;
    std::vector<uint8_t> bytes;
    bool has_vps;
    bool has_sps;
    bool has_pps;
    bool has_irap;

    AccessUnit();
    bool completeRandomAccessPoint() const;
};

struct AssemblyResult {
    bool ready;
    bool dropped;
    AccessUnit access_unit;

    AssemblyResult();
};

class AccessUnitAssembler {
public:
    AccessUnitAssembler();

    AssemblyResult feed(const RtpPacket& packet,
                        const StreamProfile& profile,
                        const DepacketizedPayload& payload,
                        bool discontinuity);
    void reset();

private:
    bool active_;
    bool corrupt_;
    AccessUnit current_;
};

}  // namespace board_receiver

#endif  // BOARD_RECEIVER_ACCESS_UNIT_ASSEMBLER_H_
