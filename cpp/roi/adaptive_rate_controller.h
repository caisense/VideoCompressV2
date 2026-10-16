#ifndef ROI_H265_ADAPTIVE_RATE_CONTROLLER_H_
#define ROI_H265_ADAPTIVE_RATE_CONTROLLER_H_

#include <algorithm>
#include <cmath>
#include "common/config.h"
#include "roi/roi_region_merger.h"

namespace roi_h265 {

// MPP relative-QP approximation. Whole-frame observations are never treated
// as measured per-region bits. Budgets use disjoint cells before truncation.
class AdaptiveRateController {
public:
    AdaptiveRateController() : ratio_(1.0), pressure_(0.0), previous_area_(0.0), average_qp_(-1.0) {
        for (int i = 0; i < 4; ++i) { offsets_[i] = 0; budgets_[i] = 0; areas_[i] = 0; represented_[i] = 0; }
    }

    void observe(size_t bytes, bool idr, int target, int fps, int gop,
                 int qp, size_t queued, double wire_fraction, int gop_position = -1) {
        // Amortize IDR over a GOP; do not react to its full size as a P frame.
        const double frame_bits = static_cast<double>(target) / std::max(1, fps);
        const double measured = bytes * 8.0 / std::max(1.0, frame_bits);
        const double gain = idr ? 1.0 / std::max(1, gop) : (gop_position == 1 ? 0.05 : 0.12);
        ratio_ += gain * (std::min(4.0, measured) - ratio_);
        pressure_ = std::max(queued >= 2 ? 1.0 : 0.0,
            std::max(0.0, (wire_fraction - 0.90) * 10.0));
        // QP is a global observation and only limits aggression near saturation.
        if (qp >= 0) average_qp_ = average_qp_ < 0 ? qp : average_qp_ * 0.9 + qp * 0.1;
        if (average_qp_ >= 48) pressure_ = std::max(pressure_, 1.0);
    }

    void apply(const RoiMap &map, const RoiConfig &config, int target, int fps,
               std::vector<RoiRegion> *regions) {
        for (int i = 0; i < 4; ++i) { areas_[i] = 0; represented_[i] = 0; }
        for (int y = 0; y < map.grid_height; ++y) {
            for (int x = 0; x < map.grid_width; ++x) {
                const int width = std::min(map.cell_size, map.frame_width - x * map.cell_size);
                const int height = std::min(map.cell_size, map.frame_height - y * map.cell_size);
                areas_[static_cast<int>(map.at(x, y))] += width * height;
            }
        }
        double weighted_area = 0;
        for (int i = 0; i < 4; ++i) weighted_area += areas_[i] * config.weights[i];
        const double roi_area = areas_[1] + areas_[2] + areas_[3];
        for (int i = 0; i < 4; ++i) {
            budgets_[i] = weighted_area > 0 ? static_cast<double>(target) /
                std::max(1, fps) * areas_[i] * config.weights[i] / weighted_area : 0;
            // Relative density approximation: bits ~ 2^(-QP/6).
            const double mean_weight = weighted_area / std::max(1, map.frame_width * map.frame_height);
            int desired = static_cast<int>(std::lround(-6.0 * std::log2(config.weights[i] /
                std::max(0.001, mean_weight)) + (ratio_ - 1.0) * 2.0 + pressure_ * (i < 2 ? 2 : 1)));
            desired = std::max(-8, std::min(8, desired));
            const int step = roi_area > previous_area_ && i >= 2 && desired < offsets_[i] ? 2 : 1;
            offsets_[i] += std::max(-step, std::min(step, desired - offsets_[i]));
        }
        previous_area_ = roi_area;
        for (size_t i = 0; i < regions->size(); ++i) {
            RoiRegion &region = (*regions)[i];
            represented_[static_cast<int>(region.level)] += region.width * region.height;
            region.delta_qp = std::max(-16, std::min(16,
                region.delta_qp + offsets_[static_cast<int>(region.level)]));
        }
    }

    double budget(int level) const { return budgets_[level]; }
    double area(int level) const { return areas_[level]; }
    double representedArea(int level) const { return represented_[level]; }
    int offset(int level) const { return offsets_[level]; }

private:
    double ratio_, pressure_, previous_area_, average_qp_;
    int offsets_[4];
    double budgets_[4], areas_[4], represented_[4];
};

} // namespace roi_h265
#endif
