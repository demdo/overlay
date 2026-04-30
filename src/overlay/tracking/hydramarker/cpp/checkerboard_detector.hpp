#pragma once

#include <array>
#include <optional>
#include <vector>

#include <opencv2/opencv.hpp>

#include "marker_field.hpp"

namespace hydramarker {

    struct GridCorner {
        int i = -1;
        int j = -1;
        cv::Point2f uv;
    };

    struct GridCell {
        int i = -1;
        int j = -1;

        std::array<int, 4> corner_indices = { -1, -1, -1, -1 };

        cv::Point2f center_uv;
    };

    struct CheckerboardDetection {
        std::vector<GridCorner> corners;
        std::vector<GridCell> cells;

        int cols = 0;
        int rows = 0;

        bool tracking = false;

        bool valid() const {
            return !corners.empty() && !cells.empty();
        }
    };

    struct CheckerboardDetectorConfig {
        cv::Size pattern_size;
        int det_width = 640;

        cv::Size subpix_window = cv::Size(5, 5);
        int subpix_max_iter = 30;
        double subpix_eps = 1e-3;

        bool use_sb_fallback = false;

        // Important:
        // Tracking should only start from a strong full detection.
        int min_detection_cells = 80;

        // Tracking may continue with fewer cells than needed for initialization.
        int min_tracking_cells = 40;
        int min_tracking_points = 50;

        int max_lost_frames = 2;

        double max_tracking_reproj_error_px = 8.0;
    };

    class CheckerboardDetector {
    public:
        explicit CheckerboardDetector(const MarkerField& field);
        explicit CheckerboardDetector(CheckerboardDetectorConfig config);

        std::optional<CheckerboardDetection> detect(
            const cv::Mat& image
        );

        void resetTracking();

        bool isTracking() const {
            return tracking_active_;
        }

    private:
        CheckerboardDetectorConfig config_;

        bool tracking_active_ = false;
        int lost_counter_ = 0;

        cv::Mat prev_gray_;

        std::vector<cv::Point2f> tracked_grid_points_;
        std::vector<cv::Point2f> tracked_image_points_;
        cv::Mat H_grid_to_image_;

        static cv::Mat toGray(const cv::Mat& image);

        void refine(
            const cv::Mat& gray,
            std::vector<cv::Point2f>& corners
        ) const;

        std::optional<CheckerboardDetection> runFullDetection(
            const cv::Mat& gray
        ) const;

        CheckerboardDetection build(
            const std::vector<cv::Point2f>& corners
        ) const;

        void initializeTracking(
            const cv::Mat& gray,
            const CheckerboardDetection& det
        );

        std::optional<CheckerboardDetection> trackFromPrevious(
            const cv::Mat& gray
        );

        CheckerboardDetection buildDetectionFromTrackedModel(
            const cv::Mat& gray,
            const std::vector<cv::Point2f>& grid_points,
            const std::vector<cv::Point2f>& image_points,
            const cv::Mat& H
        ) const;
    };

} // namespace hydramarker