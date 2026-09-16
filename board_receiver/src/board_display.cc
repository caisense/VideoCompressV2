#include "board_display.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <opencv2/opencv.hpp>

namespace board_receiver {
namespace { const char kWindow[] = "Board H.265 Receiver"; }

BoardDisplay::BoardDisplay(bool fullscreen, bool rotate_ccw)
    : fullscreen_(fullscreen), rotate_ccw_(rotate_ccw), opened_(false) {}
BoardDisplay::~BoardDisplay() { close(); }

void BoardDisplay::prepareWaylandEnvironment() {
    if (access("/run/wayland-0", F_OK) == 0) {
        if (!std::getenv("XDG_RUNTIME_DIR")) setenv("XDG_RUNTIME_DIR", "/run", 0);
        if (!std::getenv("WAYLAND_DISPLAY")) setenv("WAYLAND_DISPLAY", "wayland-0", 0);
    }
}

bool BoardDisplay::open(std::string* error) {
    if (opened_) return true;
    prepareWaylandEnvironment();
    try {
        cv::namedWindow(kWindow, cv::WINDOW_NORMAL);
        if (fullscreen_) cv::setWindowProperty(kWindow, cv::WND_PROP_FULLSCREEN,
                                                cv::WINDOW_FULLSCREEN);
        opened_ = true;
        return true;
    } catch (const cv::Exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

bool BoardDisplay::show(const DecodedFrame& frame,
                        const std::vector<std::string>& hud_lines,
                        std::string* error) {
    if (!open(error)) return false;
    if (frame.nv12.size() < static_cast<size_t>(frame.horizontal_stride) *
                            frame.vertical_stride * 3 / 2) {
        if (error) *error = "short NV12 decoder frame";
        return false;
    }
    try {
        cv::Mat y(frame.height, frame.width, CV_8UC1);
        cv::Mat uv(frame.height / 2, frame.width, CV_8UC1);
        const uint8_t* source = frame.nv12.data();
        for (int row = 0; row < frame.height; ++row)
            std::memcpy(y.ptr(row), source + static_cast<size_t>(row) * frame.horizontal_stride,
                        frame.width);
        source += static_cast<size_t>(frame.horizontal_stride) * frame.vertical_stride;
        for (int row = 0; row < frame.height / 2; ++row)
            std::memcpy(uv.ptr(row), source + static_cast<size_t>(row) * frame.horizontal_stride,
                        frame.width);
        cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1);
        y.copyTo(nv12.rowRange(0, frame.height));
        uv.copyTo(nv12.rowRange(frame.height, frame.height * 3 / 2));
        cv::Mat bgr;
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
        if (rotate_ccw_) cv::rotate(bgr, bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
        if (!hud_lines.empty()) {
            const int font = cv::FONT_HERSHEY_SIMPLEX;
            const int thickness = 1;
            double font_scale = 0.42;
            int baseline = 0;
            int widest = 0;
            for (size_t index = 0; index < hud_lines.size(); ++index) {
                widest = std::max(widest, cv::getTextSize(
                    hud_lines[index], font, font_scale, thickness, &baseline).width);
            }
            const int available = std::max(1, bgr.cols - 16);
            if (widest > available) font_scale *= static_cast<double>(available) / widest;
            font_scale = std::max(0.25, font_scale);
            const int line_height = std::max(15, cv::getTextSize(
                "Ag", font, font_scale, thickness, &baseline).height + 7);
            for (size_t index = 0; index < hud_lines.size(); ++index) {
                const int y = 5 + line_height * static_cast<int>(index + 1) - baseline;
                cv::putText(bgr, hud_lines[index], cv::Point(8, y), font,
                            font_scale, cv::Scalar(80, 255, 80), thickness,
                            cv::LINE_AA);
            }
        }
        cv::imshow(kWindow, bgr);
        const int key = cv::waitKey(1);
        return key != 'q' && key != 27;
    } catch (const cv::Exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

void BoardDisplay::close() {
    if (!opened_) return;
    try { cv::destroyWindow(kWindow); } catch (const cv::Exception&) {}
    opened_ = false;
}

}  // namespace board_receiver
