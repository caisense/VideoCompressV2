#include <signal.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "access_unit_assembler.h"
#include "board_display.h"
#include "hevc_depacketizer.h"
#include "mpp_hevc_decoder.h"
#include "profile_switch_gate.h"
#include "receiver_stats.h"
#include "rtp_receiver.h"
#include "rtp_reorder_buffer.h"
#include "stream_profile.h"

namespace {
std::atomic<bool> g_running(true);
void stopHandler(int) { g_running.store(false); }

struct Options {
    uint16_t port = 5004;
    int receive_buffer = 1048576;
    int reorder_window = 32;
    int stats_interval_ms = 1000;
    int idle_timeout_ms = 3000;
    int max_frames = 0;
    bool headless = false;
    bool fullscreen = false;
    bool rotate_ccw = true;
};

bool integerOption(const std::string& argument, const char* name, int* value) {
    const std::string prefix = std::string("--") + name + "=";
    if (argument.compare(0, prefix.size(), prefix) != 0) return false;
    char* end = NULL;
    const long parsed = std::strtol(argument.c_str() + prefix.size(), &end, 10);
    if (!end || *end || parsed < 0 || parsed > 2147483647L) return false;
    *value = static_cast<int>(parsed);
    return true;
}

bool parseOptions(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        int number = 0;
        if (integerOption(arg, "udp-port", &number) && number <= 65535) options->port = number;
        else if (integerOption(arg, "receive-buffer-bytes", &number)) options->receive_buffer = number;
        else if (integerOption(arg, "reorder-window", &number)) options->reorder_window = number;
        else if (integerOption(arg, "stats-interval-ms", &number)) options->stats_interval_ms = number;
        else if (integerOption(arg, "idle-timeout-ms", &number)) options->idle_timeout_ms = number;
        else if (integerOption(arg, "max-frames", &number)) options->max_frames = number;
        else if (arg == "--headless") options->headless = true;
        else if (arg == "--fullscreen") options->fullscreen = true;
        else if (arg == "--display=wayland") options->headless = false;
        else if (arg == "--rotate=ccw") options->rotate_ccw = true;
        else if (arg == "--rotate=none") options->rotate_ccw = false;
        else if (arg.compare(0, 12, "--log-level=") == 0) {}
        else { std::cerr << "Unknown or invalid option: " << arg << std::endl; return false; }
    }
    return options->port && options->receive_buffer >= 1048576 &&
           options->reorder_window > 0 && options->stats_interval_ms > 0;
}

class LatestFrame {
public:
    void publish(const board_receiver::DecodedFrame& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        frame_ = frame;
        available_ = true;
        condition_.notify_one();
    }
    bool take(board_receiver::DecodedFrame* frame, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [this] { return available_ || !g_running.load(); });
        if (!available_) return false;
        *frame = frame_;
        available_ = false;
        return true;
    }
private:
    std::mutex mutex_;
    std::condition_variable condition_;
    board_receiver::DecodedFrame frame_;
    bool available_ = false;
};
}  // namespace

int main(int argc, char** argv) {
    using namespace board_receiver;
    Options options;
    if (!parseOptions(argc, argv, &options)) return EXIT_FAILURE;
    signal(SIGINT, stopHandler);
    signal(SIGTERM, stopHandler);
    ReceiverStats stats;
    LatestFrame latest;
    std::atomic<uint64_t> mpp_errors(0);
    std::atomic<int> display_width(0), display_height(0);

    std::thread worker([&] {
        UdpRtpReceiver receiver;
        std::string error;
        if (!receiver.open(options.port, options.receive_buffer, 100, &error)) {
            std::cerr << "UDP receiver open failed: " << error << std::endl;
            g_running.store(false); return;
        }
        RtpReorderBuffer reorder(options.reorder_window);
        HevcRtpDepacketizer depacketizer;
        AccessUnitAssembler assembler;
        ProfileSwitchGate gate;
        MppHevcDecoder decoder;
        int64_t last_packet_ms = 0;
        while (g_running.load()) {
            ReceivedDatagram datagram;
            bool timed_out = false;
            if (!receiver.receive(&datagram, &timed_out, &error)) {
                std::cerr << "UDP receive failed: " << error << std::endl; break;
            }
            if (timed_out) {
                if (last_packet_ms && options.idle_timeout_ms > 0) {
                    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    if (now - last_packet_ms >= options.idle_timeout_ms) {
                        reorder.reset(); depacketizer.reset(); assembler.reset(); gate.reset();
                        decoder.shutdown(); last_packet_ms = 0;
                    }
                }
                continue;
            }
            last_packet_ms = datagram.received_ms;
            ++stats.packets;
            RtpPacket packet;
            if (!parseRtpPacket(datagram.bytes.data(), datagram.bytes.size(), &packet, &error)) continue;
            ReorderResult ordered = reorder.push(packet);
            stats.lost += ordered.lost; stats.duplicates += ordered.duplicates;
            stats.reordered += ordered.reordered;
            for (size_t index = 0; index < ordered.ready.size(); ++index) {
                const RtpPacket& current = ordered.ready[index];
                StreamProfile profile;
                const ProfileParseStatus status = parseStreamProfile(current, &profile, &error);
                if (status != PROFILE_SUPPORTED) {
                    if (status == PROFILE_UNSUPPORTED) ++stats.rejected_profiles;
                    continue;
                }
                if (ordered.discontinuity) depacketizer.reset();
                const DepacketizedPayload payload = depacketizer.feed(current);
                const AssemblyResult assembled = assembler.feed(
                    current, profile, payload, ordered.discontinuity);
                if (!assembled.ready) continue;
                const GateAction action = gate.evaluate(assembled.access_unit);
                if (action == GATE_DROP) continue;
                if (action == GATE_RESTART_AND_FORWARD) {
                    if (!decoder.initialize(&error)) {
                        std::cerr << error << std::endl; g_running.store(false); break;
                    }
                    ++stats.decoder_restarts;
                    stats.setProfile(assembled.access_unit.profile);
                }
                if (assembled.access_unit.has_irap) stats.last_idr_ms.store(datagram.received_ms);
                std::vector<DecodedFrame> frames;
                if (!decoder.submit(assembled.access_unit.bytes, current.timestamp,
                                    &frames, &error)) {
                    std::cerr << error << std::endl; continue;
                }
                mpp_errors.store(decoder.errorFrames());
                for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
                    ++stats.decoded_frames;
                    latest.publish(frames[frame_index]);
                }
            }
        }
        receiver.close();
        decoder.shutdown();
        g_running.store(false);
    });

    BoardDisplay display(options.fullscreen, options.rotate_ccw);
    int64_t next_stats = 0;
    while (g_running.load()) {
        DecodedFrame frame;
        if (latest.take(&frame, 20)) {
            display_width.store(options.rotate_ccw ? frame.height : frame.width);
            display_height.store(options.rotate_ccw ? frame.width : frame.height);
            if (!options.headless) {
                std::string error;
                if (!display.show(frame, &error)) {
                    std::cerr << "Display stopped: " << error << std::endl;
                    g_running.store(false);
                }
            }
            const uint64_t shown = ++stats.displayed_frames;
            if (options.max_frames > 0 && shown >= static_cast<uint64_t>(options.max_frames))
                g_running.store(false);
        }
        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now >= next_stats) {
            std::cerr << stats.report(mpp_errors.load(), display_width.load(),
                                      display_height.load()) << std::endl;
            next_stats = now + options.stats_interval_ms;
        }
    }
    worker.join();
    return EXIT_SUCCESS;
}
