#ifndef BOARD_RECEIVER_MPP_HEVC_DECODER_H_
#define BOARD_RECEIVER_MPP_HEVC_DECODER_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace board_receiver {

struct DecodedFrame {
    int width;
    int height;
    int horizontal_stride;
    int vertical_stride;
    int64_t pts;
    std::vector<uint8_t> nv12;
    DecodedFrame();
};

class MppHevcDecoder {
public:
    MppHevcDecoder();
    ~MppHevcDecoder();
    bool initialize(std::string* error);
    bool submit(const std::vector<uint8_t>& annex_b, int64_t pts,
                std::vector<DecodedFrame>* frames, std::string* error);
    void shutdown();
    uint64_t errorFrames() const { return error_frames_; }

private:
    void* context_;
    void* api_;
    void* input_storage_;
    uint64_t error_frames_;
};

}  // namespace board_receiver
#endif
