#include "dot_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hydramarker {

    namespace {

        double dist(const cv::Point2f& a, const cv::Point2f& b)
        {
            const double dx = a.x - b.x;
            const double dy = a.y - b.y;
            return std::sqrt(dx * dx + dy * dy);
        }

        bool insideImage(const cv::Mat& img, const cv::Point2f& p)
        {
            return p.x >= 0 && p.y >= 0 &&
                p.x < img.cols - 1 &&
                p.y < img.rows - 1;
        }

    } // namespace

    DotDetector::DotDetector(DotDetectorConfig config)
        : config_(config)
    {
    }

    cv::Mat DotDetector::toGray(const cv::Mat& image)
    {
        if (image.channels() == 1)
            return image;

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    std::optional<DotDetection> DotDetector::detect(
        const cv::Mat& image,
        const CheckerboardDetection& checkerboard
    ) const
    {
        if (image.empty() || !checkerboard.valid())
            return std::nullopt;

        const cv::Mat gray = toGray(image);

        DotDetection out;

        int min_i = std::numeric_limits<int>::max();
        int min_j = std::numeric_limits<int>::max();
        int max_i = std::numeric_limits<int>::lowest();
        int max_j = std::numeric_limits<int>::lowest();

        for (const auto& c : checkerboard.cells) {
            min_i = std::min(min_i, c.i);
            min_j = std::min(min_j, c.j);
            max_i = std::max(max_i, c.i);
            max_j = std::max(max_j, c.j);
        }

        if (min_i > max_i || min_j > max_j)
            return std::nullopt;

        out.origin_i = min_i;
        out.origin_j = min_j;
        out.cols = max_i - min_i + 1;
        out.rows = max_j - min_j + 1;
        out.grid.assign(out.cols * out.rows, int8_t(-1));

        for (const auto& cell : checkerboard.cells) {
            DotCell dc;
            dc.i = cell.i;
            dc.j = cell.j;
            dc.center_uv = cell.center_uv;

            evaluateCell(gray, checkerboard, cell, dc);

            const int gi = cell.i - min_i;
            const int gj = cell.j - min_j;

            if (gi >= 0 && gj >= 0 && gi < out.cols && gj < out.rows) {
                out.grid[gj * out.cols + gi] = static_cast<int8_t>(dc.value);
            }

            out.cells.push_back(dc);
        }

        return out;
    }

    bool DotDetector::evaluateCell(
        const cv::Mat& gray,
        const CheckerboardDetection& checkerboard,
        const GridCell& cell,
        DotCell& out
    ) const
    {
        std::vector<cv::Point2f> src;
        src.reserve(4);

        for (int idx : cell.corner_indices) {
            if (idx < 0 || idx >= static_cast<int>(checkerboard.corners.size())) {
                out.value = -1;
                return false;
            }

            const auto& p = checkerboard.corners[idx].uv;

            if (!insideImage(gray, p)) {
                out.value = -1;
                return false;
            }

            src.push_back(p);
        }

        const double s01 = dist(src[0], src[1]);
        const double s02 = dist(src[0], src[2]);
        const double s13 = dist(src[1], src[3]);
        const double s23 = dist(src[2], src[3]);

        const double min_s = std::min({ s01, s02, s13, s23 });

        if (min_s < config_.min_cell_size_px) {
            out.value = -1;
            return false;
        }

        const int N = config_.warp_size;

        std::vector<cv::Point2f> dst = {
            {0.f, 0.f},
            {float(N - 1), 0.f},
            {0.f, float(N - 1)},
            {float(N - 1), float(N - 1)}
        };

        const cv::Mat H = cv::getPerspectiveTransform(src, dst);

        cv::Mat patch;
        cv::warpPerspective(
            gray,
            patch,
            H,
            cv::Size(N, N),
            cv::INTER_LINEAR,
            cv::BORDER_REPLICATE
        );

        const cv::Point2f center(
            float(N - 1) * 0.5f,
            float(N - 1) * 0.5f
        );

        const float r_center = config_.center_radius * float(N);
        const float r_inner = config_.ring_inner_radius * float(N);
        const float r_outer = config_.ring_outer_radius * float(N);

        double sum_center = 0.0;
        double sum_ring = 0.0;
        int n_center = 0;
        int n_ring = 0;

        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                const float dx = float(x) - center.x;
                const float dy = float(y) - center.y;
                const float r = std::sqrt(dx * dx + dy * dy);

                const uint8_t val = patch.at<uint8_t>(y, x);

                if (r <= r_center) {
                    sum_center += val;
                    n_center++;
                }
                else if (r >= r_inner && r <= r_outer) {
                    sum_ring += val;
                    n_ring++;
                }
            }
        }

        if (n_center == 0 || n_ring == 0) {
            out.value = -1;
            return false;
        }

        const double mean_center = sum_center / double(n_center);
        const double mean_ring = sum_ring / double(n_ring);

        /*
         * Polarity-invariant dot detection.
         *
         * signed_contrast = mean_ring - mean_center
         *
         * black dot on bright cell:
         *     center dark, ring bright -> signed_contrast > 0
         *
         * white dot on dark cell:
         *     center bright, ring dark -> signed_contrast < 0
         *
         * Therefore the dot decision uses abs(signed_contrast).
         */
        const double signed_contrast = mean_ring - mean_center;
        const double abs_contrast = std::abs(signed_contrast);

        out.score = signed_contrast;

        if (abs_contrast >= config_.min_dot_contrast) {
            out.value = 1;
            return true;
        }

        if (std::abs(abs_contrast - config_.min_dot_contrast)
            <= config_.ambiguous_margin)
        {
            out.value = -1;
            return false;
        }

        out.value = 0;
        return true;
    }

} // namespace hydramarker