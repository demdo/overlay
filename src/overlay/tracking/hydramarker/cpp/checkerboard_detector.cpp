/*
 * CheckerboardDetector
 *
 * Purpose:
 *   Detect the local checkerboard/grid structure of the HydraMarker board
 *   from an RGB image.
 *
 * Output:
 *   - Grid corners (image space, subpixel refined)
 *   - Local grid indices (i, j)
 *   - Valid cells (each defined by 4 neighboring corners)
 *
 * Pipeline:
 *
 *   1. Convert image to grayscale
 *   2. Contrast enhancement (CLAHE)
 *   3. Adaptive thresholding (segment black cells)
 *   4. Contour extraction
 *   5. Filter valid square-like cells (rotation invariant)
 *   6. Extract corner candidates
 *   7. Cluster duplicates
 *   8. Subpixel refinement
 *   9. Grid reconstruction (BFS)
 *  10. Build valid cells
 *
 * Design philosophy:
 *   Corner detection is intentionally permissive.
 *   Structural consistency is enforced later via grid reconstruction.
 */

#include "checkerboard_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>

namespace hydramarker {

    namespace {

        // ========================= BASIC GEOMETRY =========================

        double dist2(const cv::Point2f& a, const cv::Point2f& b)
        {
            double dx = a.x - b.x;
            double dy = a.y - b.y;
            return dx * dx + dy * dy;
        }

        double norm(const cv::Point2f& v)
        {
            return std::sqrt(v.x * v.x + v.y * v.y);
        }

        cv::Point2f normalizeVec(const cv::Point2f& v)
        {
            double n = norm(v);
            if (n < 1e-9)
                return { 1.f, 0.f };

            return { float(v.x / n), float(v.y / n) };
        }

        double dot2(const cv::Point2f& a, const cv::Point2f& b)
        {
            return a.x * b.x + a.y * b.y;
        }

        double cross2(const cv::Point2f& a, const cv::Point2f& b)
        {
            return a.x * b.y - a.y * b.x;
        }

        std::int64_t gridKey(int i, int j)
        {
            return (std::int64_t(i) << 32) ^ std::uint32_t(j);
        }

        // ========================= CLUSTER =========================

        std::vector<cv::Point2f> clusterPoints(
            const std::vector<cv::Point2f>& pts,
            float radius
        )
        {
            std::vector<cv::Point2f> out;
            std::vector<int> counts;

            float r2 = radius * radius;

            for (const auto& p : pts) {
                int best = -1;
                double best_d = r2;

                for (int i = 0; i < (int)out.size(); ++i) {
                    double d = dist2(p, out[i]);
                    if (d < best_d) {
                        best = i;
                        best_d = d;
                    }
                }

                if (best >= 0) {
                    int n = counts[best];
                    out[best] = (out[best] * n + p) * (1.0f / (n + 1));
                    counts[best]++;
                }
                else {
                    out.push_back(p);
                    counts.push_back(1);
                }
            }

            return out;
        }

        // ========================= CELL FILTER =========================

        bool isReasonableBlackCell(
            const std::vector<cv::Point>& contour,
            std::vector<cv::Point>& approx_out,
            double img_area
        )
        {
            double area = cv::contourArea(contour);

            if (area < std::max(150.0, 0.00002 * img_area))
                return false;

            cv::RotatedRect rr = cv::minAreaRect(contour);

            float w = rr.size.width;
            float h = rr.size.height;

            if (w < 10 || h < 10)
                return false;

            float min_s = std::min(w, h);
            float max_s = std::max(w, h);

            if (max_s / min_s > 1.35)
                return false;

            double fill = area / (w * h);

            if (fill < 0.60 || fill > 1.15)
                return false;

            std::vector<cv::Point> approx;

            cv::approxPolyDP(
                contour,
                approx,
                0.025 * cv::arcLength(contour, true),
                true
            );

            if (approx.size() != 4)
                return false;

            if (!cv::isContourConvex(approx))
                return false;

            approx_out = approx;
            return true;
        }

        // ========================= SPACING =========================

        double estimateGridSpacing(const std::vector<cv::Point2f>& pts)
        {
            std::vector<double> nn;

            for (int i = 0; i < (int)pts.size(); ++i) {
                double best = 1e12;

                for (int j = 0; j < (int)pts.size(); ++j) {
                    if (i == j) continue;
                    best = std::min(best, std::sqrt(dist2(pts[i], pts[j])));
                }

                nn.push_back(best);
            }

            std::nth_element(nn.begin(), nn.begin() + nn.size() / 2, nn.end());
            return nn[nn.size() / 2];
        }

        // ========================= AXES =========================

        bool estimateGridAxes(
            const std::vector<cv::Point2f>& pts,
            double spacing,
            cv::Point2f& axis_i,
            cv::Point2f& axis_j
        )
        {
            std::vector<cv::Point2f> dirs;

            for (int i = 0; i < (int)pts.size(); ++i) {
                for (int j = i + 1; j < (int)pts.size(); ++j) {
                    cv::Point2f v = pts[j] - pts[i];
                    double d = norm(v);

                    if (d < 0.5 * spacing || d > 1.5 * spacing)
                        continue;

                    dirs.push_back(normalizeVec(v));
                }
            }

            if (dirs.size() < 4)
                return false;

            axis_i = normalizeVec(dirs[0]);
            axis_j = cv::Point2f(-axis_i.y, axis_i.x);

            return true;
        }

        // ========================= NEIGHBOR =========================

        int findNeighborInDirection(
            const std::vector<cv::Point2f>& pts,
            int src,
            const cv::Point2f& dir,
            double spacing
        )
        {
            int best = -1;
            double best_score = 1e12;

            for (int k = 0; k < (int)pts.size(); ++k) {
                if (k == src) continue;

                cv::Point2f v = pts[k] - pts[src];

                double parallel = dot2(v, dir);
                double perp = std::abs(cross2(v, dir));

                if (parallel < 0.6 * spacing || parallel > 1.4 * spacing)
                    continue;

                if (perp > 0.4 * spacing)
                    continue;

                double score = std::abs(parallel - spacing) + 2 * perp;

                if (score < best_score) {
                    best = k;
                    best_score = score;
                }
            }

            return best;
        }

    } // namespace

    // ========================= CONSTRUCTOR =========================

    CheckerboardDetector::CheckerboardDetector(const MarkerField& field)
    {
        config_.pattern_size = cv::Size(field.width() - 1, field.height() - 1);
    }

    CheckerboardDetector::CheckerboardDetector(CheckerboardDetectorConfig config)
        : config_(config)
    {
    }

    // ========================= GRAY =========================

    cv::Mat CheckerboardDetector::toGray(const cv::Mat& image)
    {
        if (image.channels() == 1)
            return image;

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    // ========================= SUBPIX =========================

    void CheckerboardDetector::refine(
        const cv::Mat& gray,
        std::vector<cv::Point2f>& corners
    ) const
    {
        if (corners.empty())
            return;

        const int mx = config_.subpix_window.width + 2;
        const int my = config_.subpix_window.height + 2;

        std::vector<cv::Point2f> valid;

        for (const auto& p : corners) {
            if (p.x > mx && p.y > my &&
                p.x < gray.cols - mx &&
                p.y < gray.rows - my)
            {
                valid.push_back(p);
            }
        }

        if (valid.empty()) {
            corners.clear();
            return;
        }

        cv::cornerSubPix(
            gray,
            valid,
            config_.subpix_window,
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                config_.subpix_max_iter,
                config_.subpix_eps
            )
        );

        corners = std::move(valid);
    }

    // ========================= DETECT =========================

    std::optional<CheckerboardDetection>
        CheckerboardDetector::detect(const cv::Mat& image) const
    {
        if (image.empty())
            return std::nullopt;

        cv::Mat gray = toGray(image);

        cv::Mat eq;
        cv::createCLAHE(2.0)->apply(gray, eq);

        cv::Mat blur;
        cv::GaussianBlur(eq, blur, { 3,3 }, 0);

        cv::Mat bin;
        cv::adaptiveThreshold(
            blur, bin, 255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            81, 9
        );

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point2f> raw;
        double img_area = gray.cols * gray.rows;

        for (auto& c : contours) {
            std::vector<cv::Point> approx;

            if (!isReasonableBlackCell(c, approx, img_area))
                continue;

            for (auto& p : approx)
                raw.emplace_back(p);
        }

        if (raw.empty())
            return std::nullopt;

        double spacing = estimateGridSpacing(raw);

        float radius = static_cast<float>(std::clamp(0.18 * spacing, 5.0, 14.0));
        auto corners = clusterPoints(raw, radius);

        refine(gray, corners);

        return build(corners);
    }

    // ========================= BUILD =========================

    CheckerboardDetection
        CheckerboardDetector::build(const std::vector<cv::Point2f>& pts) const
    {
        CheckerboardDetection out;

        for (auto& p : pts)
            out.corners.push_back({ -1, -1, p });

        if (pts.size() < 4)
            return out;

        double spacing = estimateGridSpacing(pts);

        cv::Point2f ai, aj;
        if (!estimateGridAxes(pts, spacing, ai, aj))
            return out;

        std::vector<bool> assigned(pts.size(), false);

        int start = 0;
        assigned[start] = true;

        out.corners[start].i = 0;
        out.corners[start].j = 0;

        std::queue<int> q;
        q.push(start);

        std::unordered_map<std::int64_t, int> map;
        map[gridKey(0, 0)] = start;

        while (!q.empty()) {
            int s = q.front(); q.pop();

            for (auto [dir, di, dj] :
                { std::tuple{ai,1,0}, std::tuple{-ai,-1,0},
                  std::tuple{aj,0,1}, std::tuple{-aj,0,-1} }) {

                int nb = findNeighborInDirection(pts, s, dir, spacing);
                if (nb < 0) continue;

                int ni = out.corners[s].i + di;
                int nj = out.corners[s].j + dj;

                if (!assigned[nb]) {
                    assigned[nb] = true;
                    out.corners[nb].i = ni;
                    out.corners[nb].j = nj;
                    map[gridKey(ni, nj)] = nb;
                    q.push(nb);
                }
            }
        }

        for (auto& kv : map) {
            int i = (int)(kv.first >> 32);
            int j = (int)(kv.first & 0xffffffff);

            auto k00 = map.find(gridKey(i, j));
            auto k10 = map.find(gridKey(i + 1, j));
            auto k01 = map.find(gridKey(i, j + 1));
            auto k11 = map.find(gridKey(i + 1, j + 1));

            if (k00 == map.end() || k10 == map.end() ||
                k01 == map.end() || k11 == map.end())
                continue;

            GridCell c;
            c.i = i;
            c.j = j;
            c.corner_indices = {
                k00->second,
                k10->second,
                k01->second,
                k11->second
            };

            c.center_uv = 0.25f * (
                out.corners[k00->second].uv +
                out.corners[k10->second].uv +
                out.corners[k01->second].uv +
                out.corners[k11->second].uv
                );

            out.cells.push_back(c);
        }

        return out;
    }

} // namespace hydramarker