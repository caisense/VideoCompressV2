#ifndef ROI_H265_CAPTURE_CAMERA_CAPTURE_H_
#define ROI_H265_CAPTURE_CAMERA_CAPTURE_H_

#include <string>
#include <mutex>

#include "common/config.h"
#include "common/frame_meta.h"

namespace roi_h265 {

// V4L2 capture followed by the existing RGA-backed image conversion path. The
// emitted FramePacket contains RGB for RKNN and encoder-sized NV12 for MPP.
// Quality capture bounds RGA waits to 20ms and labels temporary CPU conversion
// recovery; persistent accelerator unavailability over 5s fails explicitly.
class CameraCapture {
public:
    CameraCapture(const CameraConfig &camera, const EncoderConfig &encoder,
                  bool quality_capture = false);
    ~CameraCapture();

    bool open(std::string *error);
    bool read(FramePacket *frame, std::string *error);
    void updateEncoderConfig(const EncoderConfig &encoder);
    void close();

private:
    struct Impl;
    CameraConfig camera_;
    EncoderConfig encoder_;
    bool quality_capture_;
    mutable std::mutex encoder_mutex_;
    Impl *impl_;
};

}  // namespace roi_h265

#endif  // ROI_H265_CAPTURE_CAMERA_CAPTURE_H_
