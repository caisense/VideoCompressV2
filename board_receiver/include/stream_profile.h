#ifndef BOARD_RECEIVER_STREAM_PROFILE_H_
#define BOARD_RECEIVER_STREAM_PROFILE_H_

#include <stdint.h>

#include <string>

#include "rtp_packet.h"

namespace board_receiver {

enum ProfileId {
    PROFILE_LOW = 0,
    PROFILE_MEDIUM = 1,
    PROFILE_HIGH = 2,
    PROFILE_REBUILD = 3,
    PROFILE_GAN = 4,
};

enum ProfileParseStatus {
    PROFILE_MISSING,
    PROFILE_SUPPORTED,
    PROFILE_UNSUPPORTED,
    PROFILE_INVALID,
};

struct StreamProfile {
    uint8_t version;
    uint8_t id;
    uint16_t width;
    uint16_t height;
    uint8_t fps;
    uint8_t generation;
    uint16_t target_bitrate_kbps;
    uint16_t link_cap_kbps;

    StreamProfile();
    const char* name() const;
    bool supported() const;
};

struct ProfileKey {
    uint8_t generation;
    uint8_t profile;
    uint16_t width;
    uint16_t height;
    uint8_t fps;

    ProfileKey();
    explicit ProfileKey(const StreamProfile& value);
    bool operator==(const ProfileKey& other) const;
    bool operator!=(const ProfileKey& other) const { return !(*this == other); }
};

ProfileParseStatus parseStreamProfile(const RtpPacket& packet,
                                      StreamProfile* output,
                                      std::string* error);

}  // namespace board_receiver

#endif  // BOARD_RECEIVER_STREAM_PROFILE_H_
