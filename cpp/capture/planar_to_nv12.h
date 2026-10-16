#ifndef ROI_H265_CAPTURE_PLANAR_TO_NV12_H_
#define ROI_H265_CAPTURE_PLANAR_TO_NV12_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace roi_h265 {

inline bool planarI420ToNv12(const uint8_t *source, size_t source_size,
                            int width, int height, std::vector<uint8_t> *destination) {
    if (!source || !destination || width <= 0 || height <= 0 || width % 2 || height % 2) return false;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (source_size < pixels + pixels / 2) return false;
    destination->resize(pixels + pixels / 2);
    std::memcpy(destination->data(), source, pixels);
    for (size_t i = 0; i < pixels / 4; ++i) {
        (*destination)[pixels + i * 2] = source[pixels + i];
        (*destination)[pixels + i * 2 + 1] = source[pixels + pixels / 4 + i];
    }
    return true;
}

}  // namespace roi_h265
#endif
