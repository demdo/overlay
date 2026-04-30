#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>

#include "checkerboard_detector.hpp"

namespace hydramarker {

    struct DotCell {
        int i = -1;
        int j = -1;

        int value = -1;          // 1 = dot, 0 = no dot, -1 = invalid / uncertain
        double score = 0.0;      // signed contrast: ring_mean - center_mean

        cv::Point2f center_uv;
    };

    struct DotDetection {
        std::vector<DotCell> cells;

        std::vector<int8_t> grid;   // row-major: grid[j * cols + i]
        int cols = 0;
        int rows = 0;

        int origin_i = 0;
        int origin_j = 0;

        bool valid() const {
            return cols > 0 && rows > 0 && !grid.empty();
        }

        int8_t at(int i, int j) const {
            return grid[j * cols + i];
        }
    };

    struct DotDetectorConfig {
        int warp_size = 64;

        // Relative radii in the normalized warped cell.
        // Center is the expected dot region.
        // Ring is the local background around the expected dot.
        float center_radius = 0.18f;
        float ring_inner_radius = 0.28f;
        float ring_outer_radius = 0.45f;

        // Polarity-invariant dot contrast:
        // abs(mean_ring - mean_center) >= min_dot_contrast -> dot
        double min_dot_contrast = 18.0;

        // If the contrast is close to the threshold, mark uncertain.
        double ambiguous_margin = 6.0;

        double min_cell_size_px = 8.0;
    };

    class DotDetector {
    public:
        DotDetector() = default;
        explicit DotDetector(DotDetectorConfig config);

        std::optional<DotDetection> detect(
            const cv::Mat& image,
            const CheckerboardDetection& checkerboard
        ) const;

    private:
        DotDetectorConfig config_;

        static cv::Mat toGray(const cv::Mat& image);

        bool evaluateCell(
            const cv::Mat& gray,
            const CheckerboardDetection& checkerboard,
            const GridCell& cell,
            DotCell& out
        ) const;
    };

} // namespace hydramarker