#include "mpp_hevc_decoder.h"

#include <cstring>
#include <deque>
#include <unistd.h>

extern "C" {
#include <rk_mpi.h>
}

namespace board_receiver {
namespace {
struct DecoderInputStorage {
    std::deque<std::vector<uint8_t> > access_units;
};
}

DecodedFrame::DecodedFrame()
    : width(0), height(0), horizontal_stride(0), vertical_stride(0), pts(0) {}

MppHevcDecoder::MppHevcDecoder()
    : context_(NULL), api_(NULL), input_storage_(NULL), error_frames_(0) {}
MppHevcDecoder::~MppHevcDecoder() { shutdown(); }

bool MppHevcDecoder::initialize(std::string* error) {
    shutdown();
    MppCtx context = NULL;
    MppApi* api = NULL;
    MPP_RET result = mpp_create(&context, &api);
    if (result == MPP_OK) result = mpp_init(context, MPP_CTX_DEC, MPP_VIDEO_CodingHEVC);
    if (result != MPP_OK) {
        if (context) mpp_destroy(context);
        if (error) *error = "Rockchip MPP HEVC decoder initialization failed: " +
                            std::to_string(result);
        return false;
    }
    RK_U32 split = 1;
    api->control(context, MPP_DEC_SET_PARSER_SPLIT_MODE, &split);
    MppPollType output_timeout = MPP_POLL_NON_BLOCK;
    api->control(context, MPP_SET_OUTPUT_TIMEOUT, &output_timeout);
    context_ = context;
    api_ = api;
    input_storage_ = new DecoderInputStorage;
    return true;
}

bool MppHevcDecoder::submit(const std::vector<uint8_t>& annex_b, int64_t pts,
                            std::vector<DecodedFrame>* frames, std::string* error) {
    if (!context_ || !api_ || annex_b.empty() || !frames) {
        if (error) *error = "invalid MPP decoder state or access unit";
        return false;
    }
    DecoderInputStorage* storage = static_cast<DecoderInputStorage*>(input_storage_);
    if (!storage || storage->access_units.size() >= 256) {
        if (error) *error = "MPP input lifetime queue reached its 256-AU bound";
        return false;
    }
    storage->access_units.push_back(annex_b);
    std::vector<uint8_t>& owned_input = storage->access_units.back();
    MppPacket packet = NULL;
    MPP_RET result = mpp_packet_init(&packet,
        owned_input.data(), owned_input.size());
    if (result != MPP_OK) {
        storage->access_units.pop_back();
        if (error) *error = "mpp_packet_init failed";
        return false;
    }
    mpp_packet_set_pts(packet, pts);
    for (int retry = 0; retry < 200 && mpp_packet_get_length(packet) > 0; ++retry) {
        result = static_cast<MppApi*>(api_)->decode_put_packet(
            static_cast<MppCtx>(context_), packet);
        if (result != MPP_OK) usleep(1000);
    }
    const bool consumed = mpp_packet_get_length(packet) == 0;
    mpp_packet_deinit(&packet);
    if (!consumed) {
        storage->access_units.pop_back();
        if (error) *error = "MPP input queue did not consume access unit: " +
                            std::to_string(result);
        return false;
    }

    for (int attempt = 0; attempt < 16; ++attempt) {
        MppFrame frame = NULL;
        result = static_cast<MppApi*>(api_)->decode_get_frame(
            static_cast<MppCtx>(context_), &frame);
        if (result != MPP_OK) {
            if (error) *error = "decode_get_frame failed: " + std::to_string(result);
            return false;
        }
        if (!frame) break;
        if (mpp_frame_get_info_change(frame)) {
            static_cast<MppApi*>(api_)->control(static_cast<MppCtx>(context_),
                                                 MPP_DEC_SET_INFO_CHANGE_READY, NULL);
            mpp_frame_deinit(&frame);
            continue;
        }
        if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
            ++error_frames_;
            if (!storage->access_units.empty()) storage->access_units.pop_front();
            mpp_frame_deinit(&frame);
            continue;
        }
        MppBuffer buffer = mpp_frame_get_buffer(frame);
        const uint8_t* source = buffer
            ? static_cast<const uint8_t*>(mpp_buffer_get_ptr(buffer)) : NULL;
        const int width = mpp_frame_get_width(frame);
        const int height = mpp_frame_get_height(frame);
        const int h_stride = mpp_frame_get_hor_stride(frame);
        const int v_stride = mpp_frame_get_ver_stride(frame);
        const size_t bytes = h_stride > 0 && v_stride > 0
            ? static_cast<size_t>(h_stride) * v_stride * 3 / 2 : 0;
        if (source && width > 0 && height > 0 && width <= h_stride &&
            height <= v_stride && bytes <= 128U * 1024U * 1024U) {
            DecodedFrame output;
            output.width = width;
            output.height = height;
            output.horizontal_stride = h_stride;
            output.vertical_stride = v_stride;
            output.pts = mpp_frame_get_pts(frame);
            output.nv12.assign(source, source + bytes);
            frames->push_back(output);
            if (!storage->access_units.empty()) storage->access_units.pop_front();
        } else {
            ++error_frames_;
            if (!storage->access_units.empty()) storage->access_units.pop_front();
        }
        mpp_frame_deinit(&frame);
    }
    return true;
}

void MppHevcDecoder::shutdown() {
    if (context_) {
        // The board's older MPP runtime can dereference decoder slot state
        // during mpp_destroy unless reset has first released all HEVC refs.
        if (api_) static_cast<MppApi*>(api_)->reset(static_cast<MppCtx>(context_));
        mpp_destroy(static_cast<MppCtx>(context_));
    }
    delete static_cast<DecoderInputStorage*>(input_storage_);
    context_ = NULL;
    api_ = NULL;
    input_storage_ = NULL;
}

}  // namespace board_receiver
