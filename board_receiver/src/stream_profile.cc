#include "stream_profile.h"

namespace board_receiver {
namespace {

uint16_t readBe16(const uint8_t* data) {
    return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8) |
                                 static_cast<uint16_t>(data[1]));
}

ProfileParseStatus invalid(const char* message, std::string* error) {
    if (error) *error = message;
    return PROFILE_INVALID;
}

}  // namespace

StreamProfile::StreamProfile()
    : version(0), id(0xff), width(0), height(0), fps(0), generation(0),
      target_bitrate_kbps(0), link_cap_kbps(0) {}

const char* StreamProfile::name() const {
    switch (id) {
        case PROFILE_LOW: return "low";
        case PROFILE_MEDIUM: return "medium";
        case PROFILE_HIGH: return "high";
        case PROFILE_REBUILD: return "rebuild";
        case PROFILE_GAN: return "gan";
        default: return "unknown";
    }
}

bool StreamProfile::supported() const {
    return id == PROFILE_LOW || id == PROFILE_MEDIUM || id == PROFILE_HIGH;
}

ProfileKey::ProfileKey()
    : generation(0), profile(0xff), width(0), height(0), fps(0) {}

ProfileKey::ProfileKey(const StreamProfile& value)
    : generation(value.generation), profile(value.id), width(value.width),
      height(value.height), fps(value.fps) {}

bool ProfileKey::operator==(const ProfileKey& other) const {
    return generation == other.generation && profile == other.profile &&
           width == other.width && height == other.height && fps == other.fps;
}

ProfileParseStatus parseStreamProfile(const RtpPacket& packet,
                                      StreamProfile* output,
                                      std::string* error) {
    if (!output) return invalid("null stream profile output", error);
    if (!packet.has_extension || packet.extension_profile != 0x524f) {
        if (error) error->clear();
        return PROFILE_MISSING;
    }
    const uint8_t* payload = packet.extensionData();
    if (!payload || packet.extension_size < 8) {
        return invalid("RO extension is shorter than its 8-byte core", error);
    }
    if (payload[0] != 1) return invalid("unsupported RO metadata version", error);

    StreamProfile parsed;
    parsed.version = payload[0];
    parsed.id = payload[1];
    parsed.width = readBe16(payload + 2);
    parsed.height = readBe16(payload + 4);
    parsed.fps = payload[6];
    parsed.generation = payload[7];
    if (packet.extension_size >= 10) parsed.target_bitrate_kbps = readBe16(payload + 8);
    if (packet.extension_size >= 12) parsed.link_cap_kbps = readBe16(payload + 10);

    if (parsed.width == 0 || parsed.height == 0 || parsed.width > 8192 ||
        parsed.height > 8192 || parsed.fps == 0 || parsed.fps > 120) {
        return invalid("RO extension contains unsafe video geometry or FPS", error);
    }

    *output = parsed;
    if (error) error->clear();
    return parsed.supported() ? PROFILE_SUPPORTED : PROFILE_UNSUPPORTED;
}

}  // namespace board_receiver
