#pragma once

#include <array>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>

#include "marker_field.hpp"

namespace hydramarker {

    /*
     * One detected checkerboard/grid corner in image coordinates.
     *
     * uv:
     *   Subpixel-refined image position.
     *
     * i, j:
     *   Local grid indices assigned during grid reconstruction.
     *   These indices are local only. They do not yet describe the global
     *   HydraMarker field position.
     */
    struct GridCorner {
        int i = -1;
        int j = -1;
        cv::Point2f uv;
    };

    /*
     * One valid checkerboard cell.
     *
     * A cell is formed by four neighboring grid corners:
     *
     *   0: (i,   j)
     *   1: (i+1, j)
     *   2: (i,   j+1)
     *   3: (i+1, j+1)
     *
     * The cell center is the sampling location for the next pipeline stage,
     * the DotDetector.
     */
    struct GridCell {
        int i = -1;
        int j = -1;

        std::array<int, 4> corner_indices = { -1, -1, -1, -1 };

        cv::Point2f center_uv;
    };

    /*
     * Complete CheckerboardDetector result.
     *
     * This is the output needed by the DotDetector:
     *
     *   - detected corners
     *   - local grid indices
     *   - valid cells with image-space centers
     */
    struct CheckerboardDetection {
        std::vector<GridCorner> corners;
        std::vector<GridCell> cells;

        int cols = 0;
        int rows = 0;

        bool valid() const {
            return !corners.empty() && !cells.empty();
        }
    };

    /*
     * Configuration for the CheckerboardDetector.
     */
    struct CheckerboardDetectorConfig {
        cv::Size pattern_size;
        int det_width = 640;

        cv::Size subpix_window = cv::Size(5, 5);
        int subpix_max_iter = 30;
        double subpix_eps = 1e-3;

        bool use_sb_fallback = false;
    };

    /*
     * Detects the local checkerboard/grid structure of the HydraMarker board.
     *
     * Pipeline:
     *
     *   1. Convert image to grayscale.
     *   2. Enhance contrast.
     *   3. Segment black marker cells.
     *   4. Extract cell corners.
     *   5. Cluster duplicate corner candidates.
     *   6. Refine corners to subpixel accuracy.
     *   7. Reconstruct local grid connectivity.
     *   8. Build valid cells from four neighboring corners.
     */
    class CheckerboardDetector {
    public:
        explicit CheckerboardDetector(const MarkerField& field);
        explicit CheckerboardDetector(CheckerboardDetectorConfig config);

        std::optional<CheckerboardDetection> detect(
            const cv::Mat& image
        ) const;

    private:
        CheckerboardDetectorConfig config_;

        static cv::Mat toGray(const cv::Mat& image);

        bool detectClassic(
            const cv::Mat& gray,
            std::vector<cv::Point2f>& corners
        ) const;

        void refine(
            const cv::Mat& gray,
            std::vector<cv::Point2f>& corners
        ) const;

        CheckerboardDetection build(
            const std::vector<cv::Point2f>& corners
        ) const;
    };

}