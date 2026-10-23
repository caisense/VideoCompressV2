#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "common/config.h"
#include "transport/packetizer.h"
using namespace roi_h265;
static bool parse(std::vector<std::string> args, AppConfig *config) {
    args.insert(args.begin(), "test");
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(&arg[0]);
    std::string error;
    return parseAppConfig(static_cast<int>(argv.size()), argv.data(), config, &error);
}
int main() {
    for (const std::string rate : {"rate60", "rate80", "rate100", "rate120", "rate150", "rate180", "rate200", "rate300"}) {
        AppConfig source;
        assert(parse({"--rate-profile="+rate}, &source));
        RuntimeProfileSelection choice;
        assert(parseRuntimeProfileSelection(rate, &choice));
        assert(choice.gan_scale == 0 && choice.output_fps == 0);
        for (int scale : {2, 4}) {
            const bool supported = scale == 2 || rate == "rate60" || rate == "rate80" || rate == "rate100" || rate == "rate120";
            const std::string enhanced = rate+"_x"+std::to_string(scale);
            assert(parseRuntimeProfileSelection(enhanced, &choice) == supported);
            const bool interpolates = supported && !(rate == "rate120" && scale == 4);
            assert(parseRuntimeProfileSelection(enhanced+"_fps30", &choice) == interpolates);
            if (!interpolates) continue;
            assert(choice.profile == source.rate_profile && choice.output_fps == 30 && choice.gan_scale == scale);
            assert(((choice.packed() >> 16) & 255) == 30);
            assert(!parseRuntimeProfileSelection(enhanced+"_fps"+std::to_string(source.encoder.fps), &choice));
        }
        assert(!parseRuntimeProfileSelection(rate+"_fps30", &choice));
    }
    for (const std::string bad : {"rate60_x2_fps0", "rate60_x2_fps121", "rate60_x3", "rate60_x2_fpsnan", "rate60_x2_fps15_extra", "rate60_x2_", "rate60_x2_fps999999999999999"}) {
        RuntimeProfileSelection ignored;
        assert(!parseRuntimeProfileSelection(bad, &ignored));
    }
    for (int scale : {0, 2, 4}) {
        RtpStreamProfile p;
        p.valid = true; p.runtime_selection = true; p.gan_scale = scale;
        p.receiver_output_fps = scale ? 30 : 0;
        p.width = 320; p.height = 180; p.fps = 10;
        const uint8_t nal[] = {0,0,1,38,1,42};
        H265RtpPacketizer packetizer(0, 1, 64);
        const auto packets = packetizer.packetize(nal, sizeof(nal), 0, &p);
        assert(packets[0][16] == 3 && packets[0][28] == scale);
        assert(packets[0][29] == (scale ? 30 : 0));
        assert(packets[0].size() == 35);
    }
    for (const std::string rate : {"rate60", "rate80", "rate100", "rate120", "rate150", "rate180", "rate200", "rate300"}) {
        AppConfig plain;
        assert(parse({"--rate-profile=" + rate}, &plain));
        for (const int scale : {0, 2, 4}) {
            const std::string gan = scale ? "--gan=x" + std::to_string(scale) : "--gan=off";
            for (bool reverse : {false, true}) {
                AppConfig enhanced;
                assert(parse(reverse ? std::vector<std::string>{gan, "--rate-profile=" + rate} :
                                       std::vector<std::string>{"--rate-profile=" + rate, gan}, &enhanced));
                assert(enhanced.gan.scale == scale && enhanced.rate_profile == plain.rate_profile);
                assert(enhanced.mode == plain.mode);
#define SAME(field) assert(enhanced.field == plain.field)
                SAME(encoder.width); SAME(encoder.height); SAME(encoder.fps);
                SAME(encoder.target_bitrate_bps); SAME(encoder.gop);
                SAME(encoder.qp_min); SAME(encoder.qp_max); SAME(encoder.qp_init);
                SAME(encoder.qp_min_i); SAME(encoder.qp_max_i); SAME(encoder.qp_ip);
                SAME(encoder.max_i_prop); SAME(encoder.min_i_prop); SAME(encoder.init_ip_ratio);
                SAME(encoder.fqp_min_p); SAME(encoder.fqp_max_p);
                SAME(encoder.debreath); SAME(encoder.debreath_strength);
                SAME(encoder.intra_refresh); SAME(encoder.intra_refresh_mode); SAME(encoder.intra_refresh_num);
                SAME(encoder.super_frame_mode); SAME(encoder.max_reencode_times); SAME(encoder.super_priority);
                SAME(encoder.super_i_frame_bits); SAME(encoder.super_p_frame_bits); SAME(encoder.grayscale_encode);
                SAME(roi.background_delta_qp); SAME(roi.core_delta_qp); SAME(roi.edge_delta_qp);
                SAME(transport.pacing_bitrate_bps); SAME(transport.send_queue_frames); SAME(transport.send_max_latency_ms);
                SAME(transport.event.enabled); SAME(audio.enabled); SAME(audio.reserve_wire_bitrate_bps);
#undef SAME
                RtpStreamProfile metadata;
                metadata.valid = true;
                metadata.profile = static_cast<uint8_t>(enhanced.rate_profile);
                metadata.width = static_cast<uint16_t>(enhanced.encoder.width);
                metadata.height = static_cast<uint16_t>(enhanced.encoder.height);
                metadata.fps = static_cast<uint8_t>(enhanced.encoder.fps);
                metadata.gan_scale = static_cast<uint8_t>(scale);
                if (scale) {
                    metadata.target_bitrate_kbps = enhanced.encoder.target_bitrate_bps / 1000;
                    metadata.link_cap_kbps = enhanced.transport.pacing_bitrate_bps / 1000;
                }
                const uint8_t frame[] = {0, 0, 1, 38, 1, 42};
                H265RtpPacketizer packer(0, 1, 64);
                const auto wire = packer.packetize(frame, sizeof(frame), 0, &metadata);
                assert(wire.size() == 1);
                assert(wire[0][17] == metadata.profile);
                assert(wire[0][18] * 256 + wire[0][19] == enhanced.encoder.width);
                assert(wire[0][20] * 256 + wire[0][21] == enhanced.encoder.height);
                assert(wire[0][22] == enhanced.encoder.fps);
                if (scale) {
                    assert(wire[0][16] == 2 && wire[0][28] == scale);
                    assert(wire[0][24] * 256 + wire[0][25] == metadata.target_bitrate_kbps);
                    assert(wire[0][26] * 256 + wire[0][27] == metadata.link_cap_kbps);
                } else {
                    assert(wire[0][16] == 1 && wire[0].size() == 27);
                }
                applyRateProfile(RATE_PROFILE_RATE150, &enhanced);
                assert(enhanced.gan.scale == scale && enhanced.encoder.fps == 15);
            }
        }
    }
    for (const std::string old : {"gan", "gan2", "gan4"}) {
        AppConfig config;
        assert(!parse({"--rate-profile=" + old}, &config));
        assert(!parse({"--mode=" + old}, &config));
        RateProfile rate;
        assert(!parseRateProfile(old, &rate));
    }
    for (const std::string option : {"--gan=x3", "--transport-mode=image", "--rate-profile=rebuild",
                                    "--rate-profile=ganq120", "--gan-fps=8", "--gan-link-cap-kbps=120"}) {
        AppConfig config;
        assert(!parse({"--rate-profile=rate120", "--gan=x4", option}, &config));
    }
    AppConfig custom;
    assert(parse({"--rate-profile=rate120", "--gan=x4", "--fps=8", "--target-bitrate=64000"}, &custom));
    assert(custom.encoder.fps == 8 && custom.encoder.target_bitrate_bps == 64000);
    // Byte-exact v2 metadata, FU fragmentation and MTU accounting.
    RtpStreamProfile profile;
    profile.valid = true; profile.profile = RATE_PROFILE_RATE120;
    profile.width = 480; profile.height = 270; profile.fps = 10;
    profile.gan_scale = 4; profile.target_bitrate_kbps = 88; profile.link_cap_kbps = 120;
    std::vector<uint8_t> annex = {0, 0, 0, 1, 38, 1};
    annex.resize(400, 0x55);
    H265RtpPacketizer packetizer(0, 123, 64);
    const auto packets = packetizer.packetize(annex.data(), annex.size(), 90000, &profile);
    assert(packets.size() > 1);
    for (const auto &packet : packets) {
        assert(packet.size() <= 64);
        assert(packet[15] == 4 && packet[16] == 2 && packet[17] == RATE_PROFILE_RATE120);
        assert(packet[24] == 0 && packet[25] == 88 && packet[26] == 0 && packet[27] == 120);
        assert(packet[28] == 4 && packet[29] == 0 && packet[30] == 0 && packet[31] == 0);
        assert((packet[32] >> 1) == 49);
    }
    std::cout << "GAN selection and v2 transport tests passed\n";
}
