#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "common/config.h"
#include "roi/adaptive_rate_controller.h"
#include "transport/packetizer.h"
#include "transport/rate_pacer.h"

using namespace roi_h265;

static bool parse(const std::vector<std::string> &options, AppConfig *config) {
    std::vector<std::string> args(1, "test");
    args.insert(args.end(), options.begin(), options.end());
    std::vector<char *> argv;
    for (size_t i = 0; i < args.size(); ++i) argv.push_back(&args[i][0]);
    std::string error;
    return parseAppConfig(static_cast<int>(argv.size()), argv.data(), config, &error);
}

int main() {
    for (int cap : {60, 120, 180}) {
        AppConfig config;
        const std::string profile = "--rate-profile=ganq" + std::to_string(cap);
        assert(parse({profile}, &config));
        assert(isGanRateProfile(config.rate_profile));
        assert(config.transport.pacing_bitrate_bps == cap * 1000);
        assert(!config.audio.enabled && config.audio.reserve_wire_bitrate_bps == 0);
        assert(!config.transport.event.enabled && config.mode == PIPELINE_GAN);
        AppConfig invalid;
        assert(!parse({profile, "--audio=on"}, &invalid));
        assert(!parse({profile, "--audio-reserve-bitrate=1000"}, &invalid));
        assert(!parse({profile, "--pacing-bitrate=300000"}, &invalid));
        assert(!parse({profile, "--gan-video-bitrate-kbps=" + std::to_string(cap)}, &invalid));
        assert(!parse({profile, "--encoder-width=318"}, &invalid));
        assert(!parse({profile, "--gan-native-720p=on"}, &invalid));
        const int limit = cap == 60 ? 50 : (cap == 120 ? 100 : 155);
        AppConfig boundary;
        assert(parse({profile, "--gan-video-bitrate-kbps=" + std::to_string(limit)}, &boundary));
        assert(!parse({profile, "--gan-video-bitrate-kbps=" + std::to_string(limit + 1)}, &invalid));
        AppConfig varied;
        assert(parse({profile, "--encoder-width=384", "--encoder-height=216", "--gan-fps=6",
                      "--adaptive-roi=on", "--roi-weight-edge=4"}, &varied));
        assert(varied.encoder.width == 384 && varied.encoder.fps == 6 && varied.roi.adaptive);
        assert(!parse({profile, "--roi-weight-edge=nan"}, &invalid));
        assert(!parse({profile, "--roi-weight-core=0"}, &invalid));
    }
    AppConfig legacy;
    assert(parse({"--rate-profile=gan4"}, &legacy));
    assert(legacy.gan.video_bitrate_kbps == 90 && legacy.encoder.qp_min_i == 36);
    AppConfig gan180;
    assert(parse({"--rate-profile=gan2", "--gan-link-cap-kbps=180"}, &gan180));
    assert(gan180.encoder.target_bitrate_bps == 145000);
    RoiMap map;
    map.frame_width = 32; map.frame_height = 18; map.cell_size = 16;
    map.grid_width = 2; map.grid_height = 2;
    map.cells.assign(4, ROI_BACKGROUND); map.cells[1] = ROI_CORE; map.cells[3] = ROI_EDGE;
    RoiConfig roi;
    AdaptiveRateController controller;
    int previous[4] = {0, 0, 0, 0};
    for (int n = 0; n < 80; ++n) {
        std::vector<RoiRegion> regions = RoiRegionMerger(roi).merge(map);
        controller.apply(map, roi, 90000, 8, &regions);
        double total = 0, area = 0;
        for (int i = 0; i < 4; ++i) {
            total += controller.budget(i); area += controller.area(i);
            assert(std::abs(controller.offset(i) - previous[i]) <= 2);
            previous[i] = controller.offset(i);
        }
        assert(std::abs(total - 11250) < 0.01 && area == 576);
        for (const RoiRegion &region : regions) assert(region.delta_qp >= -16 && region.delta_qp <= 16);
        controller.observe(5000, n % 16 == 0, 90000, 8, 16, 30, 3, 0.98);
    }
    RtpStreamProfile profile;
    roi.max_regions = 1;
    std::vector<RoiRegion> limited = RoiRegionMerger(roi).merge(map);
    controller.apply(map, roi, 90000, 8, &limited);
    assert(limited.size() == 1);
    double represented = 0;
    for (int i = 0; i < 4; ++i) {
        assert(controller.representedArea(i) <= controller.area(i));
        represented += controller.representedArea(i);
    }
    assert(represented < 576);
    profile.valid = true; profile.profile = 13; profile.width = 320; profile.height = 180;
    profile.fps = 10; profile.target_bitrate_kbps = 145; profile.link_cap_kbps = 180;
    H265RtpPacketizer packetizer(0, 1, 1172);
    const std::vector<uint8_t> nal = {0, 0, 0, 1, 0x26, 1, 2, 3};
    const auto packets = packetizer.packetize(nal.data(), nal.size(), 90000, &profile);
    assert(!packets.empty());
    assert(RatePacer::wireBytesForUdpPayload(packets[0].size()) >= packets[0].size() + 66);
    std::cout << "video quality profiles / budget / ROI / RTP PASS\n";
}
