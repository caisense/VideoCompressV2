#include "capture/camera_capture.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/videodev2.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image_utils.h"
#include "im2d.h"
#include "capture/capture_policy.h"
#include "capture/bounded_conversion.h"
#include "capture/planar_to_nv12.h"

namespace roi_h265 {

namespace {

struct V4l2MappedBuffer {
    void *address;
    size_t length;

    V4l2MappedBuffer() : address(MAP_FAILED), length(0) {}
};

int ioctlRetry(int fd, unsigned long request, void *argument) {
    int result;
    do {
        result = ioctl(fd, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
}

std::string errnoMessage(const std::string &operation) {
    return operation + ": " + std::strerror(errno);
}

}  // namespace

struct CameraCapture::Impl {
    cv::VideoCapture device;
    uint64_t next_frame_id;
    int v4l2_fd;
    bool v4l2_streaming;
    int v4l2_width;
    int v4l2_height;
    int v4l2_stride;
    std::vector<V4l2MappedBuffer> v4l2_buffers;
    uint64_t driver_capture_us;
    uint32_t driver_sequence;
    uint32_t driver_flags;
    unsigned int discarded_buffers;
    bool warned_timestamp_fallback;
    std::unique_ptr<BoundedConversion> converter;
    uint64_t conversion_degraded_since_us;

    Impl()
        : next_frame_id(0), v4l2_fd(-1), v4l2_streaming(false),
          v4l2_width(0), v4l2_height(0), v4l2_stride(0), driver_capture_us(0),
          driver_sequence(0), driver_flags(0), discarded_buffers(0),
          warned_timestamp_fallback(false), conversion_degraded_since_us(0) {}

    bool queueV4l2Buffer(unsigned int index, std::string *error) {
        if (index >= v4l2_buffers.size()) {
            if (error) *error = "V4L2 returned an out-of-range buffer index";
            return false;
        }
        v4l2_buffer buffer;
        v4l2_plane plane;
        std::memset(&buffer, 0, sizeof(buffer));
        std::memset(&plane, 0, sizeof(plane));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        buffer.length = 1;
        buffer.m.planes = &plane;
        plane.length = v4l2_buffers[index].length;
        if (ioctlRetry(v4l2_fd, VIDIOC_QBUF, &buffer) < 0) {
            if (error) *error = errnoMessage("VIDIOC_QBUF");
            return false;
        }
        return true;
    }

    void closeV4l2() {
        if (v4l2_fd < 0) return;
        if (v4l2_streaming) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            ioctlRetry(v4l2_fd, VIDIOC_STREAMOFF, &type);
        }
        for (size_t i = 0; i < v4l2_buffers.size(); ++i) {
            if (v4l2_buffers[i].address != MAP_FAILED)
                munmap(v4l2_buffers[i].address, v4l2_buffers[i].length);
        }
        v4l2_buffers.clear();
        ::close(v4l2_fd);
        v4l2_fd = -1;
        v4l2_streaming = false;
        v4l2_width = 0;
        v4l2_height = 0;
        v4l2_stride = 0;
    }

    bool openV4l2(const std::string &path, int requested_width, int requested_height,
                  bool latest_only, std::string *error) {
        closeV4l2();
        v4l2_fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC | (latest_only ? O_NONBLOCK : 0));
        if (v4l2_fd < 0) {
            if (error) *error = errnoMessage("open " + path);
            return false;
        }
        v4l2_capability capability;
        std::memset(&capability, 0, sizeof(capability));
        if (ioctlRetry(v4l2_fd, VIDIOC_QUERYCAP, &capability) < 0 ||
            !(capability.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) ||
            !(capability.device_caps & V4L2_CAP_STREAMING)) {
            if (error) *error = "not a streaming multi-planar V4L2 capture node: " + path;
            closeV4l2();
            return false;
        }
        v4l2_format format;
        std::memset(&format, 0, sizeof(format));
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        format.fmt.pix_mp.width = requested_width;
        format.fmt.pix_mp.height = requested_height;
        format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
        format.fmt.pix_mp.field = V4L2_FIELD_NONE;
        format.fmt.pix_mp.num_planes = 1;
        if (ioctlRetry(v4l2_fd, VIDIOC_S_FMT, &format) < 0 ||
            format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 ||
            format.fmt.pix_mp.num_planes != 1) {
            if (error) *error = "V4L2 node does not accept single-plane NV12: " + path;
            closeV4l2();
            return false;
        }
        v4l2_width = static_cast<int>(format.fmt.pix_mp.width);
        v4l2_height = static_cast<int>(format.fmt.pix_mp.height);
        v4l2_stride = static_cast<int>(format.fmt.pix_mp.plane_fmt[0].bytesperline);
        if (v4l2_width <= 0 || v4l2_height <= 0 || v4l2_stride < v4l2_width) {
            if (error) *error = "invalid NV12 layout returned by " + path;
            closeV4l2();
            return false;
        }
        v4l2_requestbuffers request;
        std::memset(&request, 0, sizeof(request));
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        request.memory = V4L2_MEMORY_MMAP;
        // Cover one 6fps sampling period of the 30fps camera, while remaining
        // bounded and selecting the latest completed buffer on each read.
        request.count = latest_only ? 8 : 3;
        if (ioctlRetry(v4l2_fd, VIDIOC_REQBUFS, &request) < 0 || request.count < 2) {
            if (error) *error = errnoMessage("VIDIOC_REQBUFS");
            closeV4l2();
            return false;
        }
        v4l2_buffers.resize(request.count);
        for (unsigned int i = 0; i < request.count; ++i) {
            v4l2_buffer buffer;
            v4l2_plane plane;
            std::memset(&buffer, 0, sizeof(buffer));
            std::memset(&plane, 0, sizeof(plane));
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = i;
            buffer.length = 1;
            buffer.m.planes = &plane;
            if (ioctlRetry(v4l2_fd, VIDIOC_QUERYBUF, &buffer) < 0) {
                if (error) *error = errnoMessage("VIDIOC_QUERYBUF");
                closeV4l2();
                return false;
            }
            v4l2_buffers[i].length = plane.length;
            v4l2_buffers[i].address = mmap(NULL, plane.length, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, v4l2_fd, plane.m.mem_offset);
            if (v4l2_buffers[i].address == MAP_FAILED) {
                if (error) *error = errnoMessage("mmap V4L2 buffer");
                closeV4l2();
                return false;
            }
            if (!queueV4l2Buffer(i, error)) {
                closeV4l2();
                return false;
            }
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctlRetry(v4l2_fd, VIDIOC_STREAMON, &type) < 0) {
            if (error) *error = errnoMessage("VIDIOC_STREAMON");
            closeV4l2();
            return false;
        }
        v4l2_streaming = true;
        std::fprintf(stderr, "CAPTURE_CONFIG native=1 buffers=%zu latest_only=%d\n",
                     v4l2_buffers.size(), latest_only ? 1 : 0);
        return true;
    }

    bool readV4l2(cv::Mat *bgr, bool latest_only, std::string *error) {
        pollfd descriptor;
        std::memset(&descriptor, 0, sizeof(descriptor));
        descriptor.fd = v4l2_fd;
        descriptor.events = POLLIN;
        const int poll_result = poll(&descriptor, 1, 2000);
        if (poll_result <= 0) {
            if (error) *error = poll_result == 0 ? "V4L2 frame acquisition timed out" : errnoMessage("poll V4L2");
            return false;
        }
        v4l2_buffer buffer;
        v4l2_plane plane;
        std::memset(&buffer, 0, sizeof(buffer));
        std::memset(&plane, 0, sizeof(plane));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.length = 1;
        buffer.m.planes = &plane;
        if (ioctlRetry(v4l2_fd, VIDIOC_DQBUF, &buffer) < 0) {
            if (error) *error = errnoMessage("VIDIOC_DQBUF");
            return false;
        }
        if (buffer.index >= v4l2_buffers.size() ||
            plane.bytesused < static_cast<unsigned int>(v4l2_stride * v4l2_height * 3 / 2)) {
            if (error) *error = "V4L2 returned an incomplete NV12 frame";
            if (buffer.index < v4l2_buffers.size()) queueV4l2Buffer(buffer.index, NULL);
            return false;
        }
        discarded_buffers = 0;
        if (latest_only) {
            const CaptureDrainResult drained = drainReadyCaptureFrames(
                static_cast<unsigned int>(v4l2_buffers.size() - 1), [&]() {
                    v4l2_buffer next;
                    v4l2_plane next_plane;
                    std::memset(&next, 0, sizeof(next));
                    std::memset(&next_plane, 0, sizeof(next_plane));
                    next.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                    next.memory = V4L2_MEMORY_MMAP;
                    next.length = 1;
                    next.m.planes = &next_plane;
                    if (ioctlRetry(v4l2_fd, VIDIOC_DQBUF, &next) < 0) {
                        if (errno == EAGAIN) return 0;
                        if (error) *error = errnoMessage("VIDIOC_DQBUF latest");
                        return -1;
                    }
                    if (next.index >= v4l2_buffers.size() ||
                        next_plane.bytesused < static_cast<unsigned int>(
                            v4l2_stride * v4l2_height * 3 / 2)) {
                        if (error) *error = "V4L2 returned an incomplete latest NV12 frame";
                        if (next.index < v4l2_buffers.size()) queueV4l2Buffer(next.index, NULL);
                        return -1;
                    }
                    if (!queueV4l2Buffer(buffer.index, error)) {
                        queueV4l2Buffer(next.index, NULL);
                        return -1;
                    }
                    buffer = next;
                    plane = next_plane;
                    buffer.m.planes = &plane;
                    return 1;
                });
            discarded_buffers = drained.discarded;
            if (!drained.ok) {
                queueV4l2Buffer(buffer.index, NULL);
                return false;
            }
        }
        const uint64_t read_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        driver_capture_us = monotonicCaptureTimestamp(buffer.timestamp.tv_sec,
            buffer.timestamp.tv_usec, (buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) ==
                V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, read_us);
        driver_sequence = buffer.sequence;
        driver_flags = buffer.flags;
        if (latest_only && !driver_capture_us && !warned_timestamp_fallback) {
            std::fprintf(stderr,
                "CAPTURE_TIMESTAMP_FALLBACK source=camera_read driver_flags=%u; "
                "driver queue age cannot be verified\n", buffer.flags);
            warned_timestamp_fallback = true;
        }
        const uint8_t *source = static_cast<const uint8_t *>(v4l2_buffers[buffer.index].address);
        cv::Mat tightly_packed(v4l2_height * 3 / 2, v4l2_width, CV_8UC1);
        for (int row = 0; row < v4l2_height; ++row) {
            std::memcpy(tightly_packed.data + static_cast<size_t>(row) * v4l2_width,
                        source + static_cast<size_t>(row) * v4l2_stride, v4l2_width);
        }
        const size_t source_uv = static_cast<size_t>(v4l2_stride) * v4l2_height;
        const size_t destination_uv = static_cast<size_t>(v4l2_width) * v4l2_height;
        for (int row = 0; row < v4l2_height / 2; ++row) {
            std::memcpy(tightly_packed.data + destination_uv + static_cast<size_t>(row) * v4l2_width,
                        source + source_uv + static_cast<size_t>(row) * v4l2_stride, v4l2_width);
        }
        if (!queueV4l2Buffer(buffer.index, error)) return false;
        cv::cvtColor(tightly_packed, *bgr, cv::COLOR_YUV2BGR_NV12);
        return !bgr->empty();
    }
};

namespace {

uint64_t nowMicros() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool numericDevice(const std::string &device, int *index) {
    if (device.empty()) return false;
    char *end = NULL;
    const long parsed = std::strtol(device.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed < 0) return false;
    *index = static_cast<int>(parsed);
    return true;
}

}  // namespace

CameraCapture::CameraCapture(const CameraConfig &camera, const EncoderConfig &encoder,
                             bool quality_capture)
    : camera_(camera), encoder_(encoder), quality_capture_(quality_capture), impl_(NULL) {}

CameraCapture::~CameraCapture() { close(); }

bool CameraCapture::open(std::string *error) {
    close();
    impl_ = new Impl;
    int index = 0;
    const std::string v4l2_path = numericDevice(camera_.device, &index)
        ? "/dev/video" + std::to_string(index) : camera_.device;
    // RKISP mainpath nodes are V4L2 multi-planar.  OpenCV 4.5.4's V4L2
    // backend rejects them even though they stream correctly, so try the
    // native NV12 capture path first.  Files remain on the OpenCV path.
    if (camera_.input_video.empty() &&
        impl_->openV4l2(v4l2_path, camera_.width, camera_.height, quality_capture_, error)) {
        return true;
    }
    if (camera_.input_video.empty() && quality_capture_) {
        if (error) *error = "quality capture requires native NV12 V4L2: " + *error;
        close();
        return false;
    }
    const bool opened = !camera_.input_video.empty()
        ? impl_->device.open(camera_.input_video)
        : (numericDevice(camera_.device, &index)
            ? impl_->device.open(index, cv::CAP_V4L2)
            : impl_->device.open(camera_.device, cv::CAP_V4L2));
    if (!opened) {
        if (error) *error = "cannot open " + (camera_.input_video.empty() ? camera_.device : camera_.input_video);
        close();
        return false;
    }
    impl_->device.set(cv::CAP_PROP_FRAME_WIDTH, camera_.width);
    impl_->device.set(cv::CAP_PROP_FRAME_HEIGHT, camera_.height);
    EncoderConfig encoder;
    {
        std::lock_guard<std::mutex> lock(encoder_mutex_);
        encoder = encoder_;
    }
    impl_->device.set(cv::CAP_PROP_FPS, encoder.fps);
    return true;
}

bool CameraCapture::read(FramePacket *frame, std::string *error) {
    if (!impl_ || !frame) {
        if (error) *error = "camera is not open";
        return false;
    }
    EncoderConfig encoder;
    {
        std::lock_guard<std::mutex> lock(encoder_mutex_);
        encoder = encoder_;
    }
    cv::Mat bgr;
    const bool read_ok = impl_->v4l2_fd >= 0
        ? impl_->readV4l2(&bgr, quality_capture_, error)
        : impl_->device.read(bgr);
    if (!read_ok || bgr.empty()) {
        if (error) *error = "V4L2 frame acquisition failed";
        return false;
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();

    FramePacket converted;
    converted.meta.frame_id = impl_->next_frame_id++;
    converted.meta.pts_us = nowMicros();
    converted.capture_read_time_us = converted.meta.pts_us;
    converted.native_capture = impl_->v4l2_fd >= 0;
    converted.file_input = !camera_.input_video.empty();
    if (converted.native_capture) {
        converted.driver_capture_time_us = impl_->driver_capture_us;
        converted.driver_sequence = impl_->driver_sequence;
        converted.driver_buffer_flags = impl_->driver_flags;
        converted.discarded_capture_buffers = impl_->discarded_buffers;
        converted.driver_timestamp_used = quality_capture_ && impl_->driver_capture_us > 0;
        if (converted.driver_timestamp_used) converted.meta.pts_us = impl_->driver_capture_us;
    }
    converted.meta.capture_time_us = converted.meta.pts_us;
    converted.source_width = rgb.cols;
    converted.source_height = rgb.rows;
    converted.encoder_width = encoder.width;
    converted.encoder_height = encoder.height;
    converted.rgb.assign(rgb.data, rgb.data + static_cast<size_t>(rgb.cols) * rgb.rows * 3);
    converted.nv12.assign(static_cast<size_t>(encoder.width) * encoder.height * 3 / 2, 0);

    image_buffer_t source;
    std::memset(&source, 0, sizeof(source));
    source.width = converted.source_width;
    source.height = converted.source_height;
    source.width_stride = converted.source_width;
    source.height_stride = converted.source_height;
    source.format = IMAGE_FORMAT_RGB888;
    source.virt_addr = converted.rgb.data();
    source.size = static_cast<int>(converted.rgb.size());
    source.fd = -1;
    image_buffer_t destination;
    std::memset(&destination, 0, sizeof(destination));
    destination.width = encoder.width;
    destination.height = encoder.height;
    destination.width_stride = encoder.width;
    destination.height_stride = encoder.height;
    destination.format = IMAGE_FORMAT_YUV420SP_NV12;
    destination.virt_addr = converted.nv12.data();
    destination.size = static_cast<int>(converted.nv12.size());
    destination.fd = -1;
    image_rect_t source_rect;
    source_rect.left = 0;
    source_rect.top = 0;
    source_rect.right = source.width - 1;
    source_rect.bottom = source.height - 1;
    image_rect_t destination_rect;
    destination_rect.left = 0;
    destination_rect.top = 0;
    destination_rect.right = destination.width - 1;
    destination_rect.bottom = destination.height - 1;
    if (quality_capture_ && converted.meta.frame_id == 0) {
        std::fprintf(stderr, "CAPTURE_RGA_CORE requested=RGA3_CORE0 mask=%d\n",
                     IM_SCHEDULER_RGA3_CORE0);
        std::fprintf(stderr, "CAPTURE_CONVERT_POLICY rga_budget_ms=20 recovery=OPENCV_RGB_NV12 max_recovery_ms=5000 max_inflight=1\n");
    }
    if (quality_capture_) {
        if (!impl_->converter) impl_->converter.reset(new BoundedConversion);
        converted.bounded_conversion = true;
        const uint64_t wait_start = nowMicros();
        BoundedConversion::Result result = {BoundedConversion::BUSY, -1};
        std::shared_ptr<FramePacket> job;
        if (!impl_->converter->busy()) {
            // The driver may continue DMA after the frame deadline. Never
            // borrow the caller's RGB/NV12 storage for that operation.
            job.reset(new FramePacket(converted));
            result = impl_->converter->run([job, source, destination, source_rect, destination_rect]() {
                static thread_local bool named = false;
                if (!named) {
                    pthread_setname_np(pthread_self(), "capture-rga");
                    named = true;
                }
                image_buffer_t src = source, dst = destination;
                image_rect_t src_rect = source_rect, dst_rect = destination_rect;
                src.virt_addr = job->rgb.data();
                dst.virt_addr = job->nv12.data();
                return convert_image_rga_core(&src, &dst, &src_rect, &dst_rect, 0,
                                              IM_SCHEDULER_RGA3_CORE0);
            }, std::chrono::milliseconds(20));
        }
        converted.conversion_wait_us = nowMicros() - wait_start;
        if (result.outcome == BoundedConversion::COMPLETE && result.status == 0) {
            converted.nv12.swap(job->nv12);
            if (impl_->conversion_degraded_since_us) {
                std::fprintf(stderr, "CAPTURE_CONVERT_RECOVERED seq=%llu degraded_us=%llu\n",
                    static_cast<unsigned long long>(converted.meta.frame_id),
                    static_cast<unsigned long long>(nowMicros() - impl_->conversion_degraded_since_us));
                impl_->conversion_degraded_since_us = 0;
            }
        } else {
            converted.conversion_cpu_fallback = true;
            converted.conversion_recovery_reason = result.outcome == BoundedConversion::BUSY ? 1 :
                (result.outcome == BoundedConversion::DEADLINE ? 2 : 3);
            if (!impl_->conversion_degraded_since_us) {
                impl_->conversion_degraded_since_us = nowMicros();
                std::fprintf(stderr, "CAPTURE_CONVERT_RECOVERY seq=%llu backend=OPENCV_RGB_NV12 reason=%d wait_us=%llu\n",
                    static_cast<unsigned long long>(converted.meta.frame_id), converted.conversion_recovery_reason,
                    static_cast<unsigned long long>(converted.conversion_wait_us));
            }
            if (nowMicros() - impl_->conversion_degraded_since_us > 5000000) {
                if (error) *error = "RGA unavailable for over 5s; refusing permanent CPU conversion recovery";
                return false;
            }
            cv::Mat resized, planar;
            cv::resize(rgb, resized, cv::Size(encoder.width, encoder.height), 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(resized, planar, cv::COLOR_RGB2YUV_I420);
            if (!planar.isContinuous()) planar = planar.clone();
            if (!planarI420ToNv12(planar.data, planar.total() * planar.elemSize(),
                                  encoder.width, encoder.height, &converted.nv12)) {
                if (error) *error = "CPU conversion recovery produced invalid I420 storage";
                return false;
            }
        }
    } else if (convert_image(&source, &destination, &source_rect, &destination_rect, 0) != 0) {
        if (error) *error = "RGA RGB-to-NV12 scale failed";
        return false;
    }
    // Keep the board preview and RKNN input in color, but make the encoded
    // stream effectively monochrome for the 60 kbps visual profile. NV12 UV
    // is interleaved; neutral chroma preserves luma detail and compresses
    // color variation to nearly zero without changing the PC decoder.
    if (encoder.grayscale_encode) {
        const size_t luma_bytes = static_cast<size_t>(encoder.width) * encoder.height;
        std::fill(converted.nv12.begin() + luma_bytes, converted.nv12.end(), 128);
    }
    *frame = converted;
    return true;
}

void CameraCapture::updateEncoderConfig(const EncoderConfig &encoder) {
    std::lock_guard<std::mutex> lock(encoder_mutex_);
    encoder_ = encoder;
}

void CameraCapture::close() {
    if (!impl_) return;
    impl_->closeV4l2();
    if (impl_->device.isOpened()) impl_->device.release();
    delete impl_;
    impl_ = NULL;
}

}  // namespace roi_h265
