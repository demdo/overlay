#include "checkerboard_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace hydramarker {

    namespace {

        double dist2(const cv::Point2f& a, const cv::Point2f& b)
        {
            const double dx = a.x - b.x;
            const double dy = a.y - b.y;
            return dx * dx + dy * dy;
        }

        double norm(const cv::Point2f& v)
        {
            return std::sqrt(double(v.x) * double(v.x) + double(v.y) * double(v.y));
        }

        double dot2(const cv::Point2f& a, const cv::Point2f& b)
        {
            return double(a.x) * double(b.x) + double(a.y) * double(b.y);
        }

        double cross2(const cv::Point2f& a, const cv::Point2f& b)
        {
            return double(a.x) * double(b.y) - double(a.y) * double(b.x);
        }

        cv::Point2f normalizeVec(const cv::Point2f& v)
        {
            const double n = norm(v);
            if (n < 1e-9)
                return { 1.f, 0.f };

            return { float(double(v.x) / n), float(double(v.y) / n) };
        }

        std::int64_t gridKey(int i, int j)
        {
            return (std::int64_t(i) << 32) ^ std::uint32_t(j);
        }

        int keyI(std::int64_t k)
        {
            return int(k >> 32);
        }

        int keyJ(std::int64_t k)
        {
            return int(std::uint32_t(k & 0xffffffff));
        }

        double angleModPi(const cv::Point2f& v)
        {
            double a = std::atan2(v.y, v.x);

            if (a < 0.0)
                a += CV_PI;

            if (a >= CV_PI)
                a -= CV_PI;

            return a;
        }

        double angleDiffModPi(double a, double b)
        {
            double d = std::abs(a - b);
            return std::min(d, CV_PI - d);
        }

        cv::Point2f signedToReference(cv::Point2f v, const cv::Point2f& ref)
        {
            if (dot2(v, ref) < 0.0)
                v = -v;

            return v;
        }

        cv::Point2f centroid(const std::vector<cv::Point2f>& pts)
        {
            cv::Point2f c(0.f, 0.f);

            for (const auto& p : pts)
                c += p;

            return c * (1.0f / float(pts.size()));
        }

        std::vector<cv::Point2f> clusterPoints(
            const std::vector<cv::Point2f>& pts,
            float radius
        )
        {
            std::vector<cv::Point2f> out;
            std::vector<int> counts;

            const float r2 = radius * radius;

            for (const auto& p : pts) {
                int best = -1;
                double best_d = r2;

                for (int i = 0; i < int(out.size()); ++i) {
                    const double d = dist2(p, out[i]);

                    if (d < best_d) {
                        best = i;
                        best_d = d;
                    }
                }

                if (best >= 0) {
                    const int n = counts[best];
                    out[best] =
                        (out[best] * float(n) + p) *
                        (1.0f / float(n + 1));
                    counts[best]++;
                }
                else {
                    out.push_back(p);
                    counts.push_back(1);
                }
            }

            return out;
        }

        bool isReasonableBlackCell(
            const std::vector<cv::Point>& contour,
            std::vector<cv::Point>& approx_out,
            double img_area
        )
        {
            const double area = cv::contourArea(contour);

            if (area < std::max(150.0, 0.00002 * img_area))
                return false;

            const cv::RotatedRect rr = cv::minAreaRect(contour);

            const float w = rr.size.width;
            const float h = rr.size.height;

            if (w < 10 || h < 10)
                return false;

            const float min_s = std::min(w, h);
            const float max_s = std::max(w, h);

            if (max_s / min_s > 1.35f)
                return false;

            const double fill = area / double(w * h);

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

        double estimateGridSpacing(const std::vector<cv::Point2f>& pts)
        {
            if (pts.size() < 2)
                return 1.0;

            std::vector<double> nn;
            nn.reserve(pts.size());

            for (int i = 0; i < int(pts.size()); ++i) {
                double best = 1e12;

                for (int j = 0; j < int(pts.size()); ++j) {
                    if (i == j)
                        continue;

                    best = std::min(best, std::sqrt(dist2(pts[i], pts[j])));
                }

                if (std::isfinite(best))
                    nn.push_back(best);
            }

            if (nn.empty())
                return 1.0;

            std::nth_element(nn.begin(), nn.begin() + nn.size() / 2, nn.end());
            return nn[nn.size() / 2];
        }

        bool averageDirectionNearAngle(
            const std::vector<cv::Point2f>& dirs,
            double target_angle,
            double max_angle_error,
            cv::Point2f& out
        )
        {
            cv::Point2f ref(
                float(std::cos(target_angle)),
                float(std::sin(target_angle))
            );

            cv::Point2f sum(0.f, 0.f);
            int count = 0;

            for (auto d : dirs) {
                const double a = angleModPi(d);

                if (angleDiffModPi(a, target_angle) > max_angle_error)
                    continue;

                d = signedToReference(d, ref);
                sum += d;
                count++;
            }

            if (count < 2)
                return false;

            out = normalizeVec(sum);
            return true;
        }

        bool estimateGridAxesRobust(
            const std::vector<cv::Point2f>& pts,
            double spacing,
            cv::Point2f& axis_i,
            cv::Point2f& axis_j
        )
        {
            std::vector<cv::Point2f> dirs;

            for (int i = 0; i < int(pts.size()); ++i) {
                for (int j = i + 1; j < int(pts.size()); ++j) {
                    const cv::Point2f v = pts[j] - pts[i];
                    const double d = norm(v);

                    if (d < 0.45 * spacing || d > 1.70 * spacing)
                        continue;

                    dirs.push_back(normalizeVec(v));
                }
            }

            if (dirs.size() < 6)
                return false;

            constexpr int bins = 36;
            std::vector<int> hist(bins, 0);

            for (const auto& d : dirs) {
                const double a = angleModPi(d);
                int b = int(std::floor(a / CV_PI * bins));
                b = std::clamp(b, 0, bins - 1);
                hist[b]++;
            }

            const int b0 =
                int(std::max_element(hist.begin(), hist.end()) - hist.begin());

            int b1 = -1;
            int best_vote = -1;

            for (int b = 0; b < bins; ++b) {
                const double a0 = (double(b0) + 0.5) * CV_PI / double(bins);
                const double a1 = (double(b) + 0.5) * CV_PI / double(bins);
                const double sep = angleDiffModPi(a0, a1);

                if (sep < 35.0 * CV_PI / 180.0)
                    continue;

                if (sep > 145.0 * CV_PI / 180.0)
                    continue;

                if (hist[b] > best_vote) {
                    best_vote = hist[b];
                    b1 = b;
                }
            }

            if (b1 < 0)
                return false;

            const double a0 = (double(b0) + 0.5) * CV_PI / double(bins);
            const double a1 = (double(b1) + 0.5) * CV_PI / double(bins);

            if (!averageDirectionNearAngle(dirs, a0, 17.0 * CV_PI / 180.0, axis_i))
                return false;

            if (!averageDirectionNearAngle(dirs, a1, 17.0 * CV_PI / 180.0, axis_j))
                return false;

            if (cross2(axis_i, axis_j) < 0.0)
                axis_j = -axis_j;

            return true;
        }

        bool solveLatticeCoordinates(
            const cv::Point2f& p,
            const cv::Point2f& origin,
            const cv::Point2f& step_i,
            const cv::Point2f& step_j,
            double& u,
            double& v
        )
        {
            const cv::Point2f q = p - origin;

            const double a = step_i.x;
            const double b = step_j.x;
            const double c = step_i.y;
            const double d = step_j.y;

            const double det = a * d - b * c;

            if (std::abs(det) < 1e-9)
                return false;

            u = (d * q.x - b * q.y) / det;
            v = (-c * q.x + a * q.y) / det;

            return true;
        }

        std::vector<int> chooseOriginCandidates(
            const std::vector<cv::Point2f>& pts,
            int max_count
        )
        {
            const cv::Point2f c = centroid(pts);

            std::vector<int> ids(pts.size());
            std::iota(ids.begin(), ids.end(), 0);

            std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                const double da = std::sqrt(dist2(pts[a], c));
                const double db = std::sqrt(dist2(pts[b], c));

                return da < db;
                });

            if (int(ids.size()) > max_count)
                ids.resize(max_count);

            return ids;
        }

        bool isReasonableCellGeometry(
            const cv::Point2f& p00,
            const cv::Point2f& p10,
            const cv::Point2f& p01,
            const cv::Point2f& p11,
            double spacing
        )
        {
            const double a = norm(p10 - p00);
            const double b = norm(p11 - p01);
            const double c = norm(p01 - p00);
            const double d = norm(p11 - p10);

            const double min_len = 0.35 * spacing;
            const double max_len = 2.30 * spacing;

            if (a < min_len || b < min_len || c < min_len || d < min_len)
                return false;

            if (a > max_len || b > max_len || c > max_len || d > max_len)
                return false;

            const double area1 = std::abs(cross2(p10 - p00, p01 - p00));
            const double area2 = std::abs(cross2(p11 - p10, p01 - p10));

            if (area1 + area2 < 0.15 * spacing * spacing)
                return false;

            return true;
        }

        bool insideImage(const cv::Mat& img, const cv::Point2f& p, float margin = 4.f)
        {
            return p.x >= margin &&
                p.y >= margin &&
                p.x < float(img.cols) - margin &&
                p.y < float(img.rows) - margin;
        }

    } // namespace

    CheckerboardDetector::CheckerboardDetector(const MarkerField& field)
    {
        config_.pattern_size = cv::Size(field.width() - 1, field.height() - 1);
    }

    CheckerboardDetector::CheckerboardDetector(CheckerboardDetectorConfig config)
        : config_(config)
    {
    }

    void CheckerboardDetector::resetTracking()
    {
        tracking_active_ = false;
        lost_counter_ = 0;
        prev_gray_.release();
        tracked_grid_points_.clear();
        tracked_image_points_.clear();
        H_grid_to_image_.release();
    }

    cv::Mat CheckerboardDetector::toGray(const cv::Mat& image)
    {
        if (image.channels() == 1)
            return image.clone();

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

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

    std::optional<CheckerboardDetection>
        CheckerboardDetector::detect(const cv::Mat& image)
    {
        if (image.empty())
            return std::nullopt;

        cv::Mat gray = toGray(image);

        if (tracking_active_) {
            auto tracked = trackFromPrevious(gray);

            if (tracked && int(tracked->cells.size()) >= config_.min_tracking_cells) {
                tracked->tracking = true;
                gray.copyTo(prev_gray_);
                lost_counter_ = 0;
                return tracked;
            }

            lost_counter_++;

            if (lost_counter_ > config_.max_lost_frames) {
                resetTracking();
            }
        }

        auto detected = runFullDetection(gray);

        if (detected) {
            detected->tracking = false;

            if (int(detected->cells.size()) >= config_.min_detection_cells) {
                initializeTracking(gray, *detected);
            }
        }

        return detected;
    }

    std::optional<CheckerboardDetection>
        CheckerboardDetector::runFullDetection(const cv::Mat& gray) const
    {
        cv::Mat eq;
        cv::createCLAHE(2.0)->apply(gray, eq);

        cv::Mat blur;
        cv::GaussianBlur(eq, blur, { 3, 3 }, 0);

        cv::Mat bin;
        cv::adaptiveThreshold(
            blur,
            bin,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            81,
            9
        );

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            bin,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        std::vector<cv::Point2f> raw;
        const double img_area = double(gray.cols) * double(gray.rows);

        for (auto& c : contours) {
            std::vector<cv::Point> approx;

            if (!isReasonableBlackCell(c, approx, img_area))
                continue;

            for (auto& p : approx)
                raw.emplace_back(float(p.x), float(p.y));
        }

        if (raw.empty())
            return std::nullopt;

        const double raw_spacing = estimateGridSpacing(raw);
        const float radius =
            float(std::clamp(0.18 * raw_spacing, 5.0, 14.0));

        auto corners = clusterPoints(raw, radius);
        refine(gray, corners);

        return build(corners);
    }

    CheckerboardDetection
        CheckerboardDetector::build(const std::vector<cv::Point2f>& pts) const
    {
        CheckerboardDetection base;

        for (const auto& p : pts)
            base.corners.push_back({ -1, -1, p });

        if (pts.size() < 4)
            return base;

        const double spacing = estimateGridSpacing(pts);

        cv::Point2f axis_i;
        cv::Point2f axis_j;

        if (!estimateGridAxesRobust(pts, spacing, axis_i, axis_j))
            return base;

        const cv::Point2f step_i = axis_i * float(spacing);
        const cv::Point2f step_j = axis_j * float(spacing);

        const std::vector<int> origins = chooseOriginCandidates(pts, 20);

        CheckerboardDetection best = base;
        int best_score = -1;

        for (int origin_idx : origins) {
            CheckerboardDetection current = base;

            const cv::Point2f origin = pts[origin_idx];

            struct Candidate {
                int point_idx = -1;
                int i = 0;
                int j = 0;
                double score = 0.0;
            };

            std::vector<Candidate> candidates;
            candidates.reserve(pts.size());

            for (int idx = 0; idx < int(pts.size()); ++idx) {
                double u = 0.0;
                double v = 0.0;

                if (!solveLatticeCoordinates(
                    pts[idx],
                    origin,
                    step_i,
                    step_j,
                    u,
                    v
                ))
                {
                    continue;
                }

                const int gi = int(std::llround(u));
                const int gj = int(std::llround(v));

                const double du = u - double(gi);
                const double dv = v - double(gj);

                const double err = std::max(std::abs(du), std::abs(dv));

                if (err > 0.65)
                    continue;

                const double score =
                    err + 0.08 * (std::abs(u) + std::abs(v));

                candidates.push_back({ idx, gi, gj, score });
            }

            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                    return a.score < b.score;
                }
            );

            std::unordered_map<std::int64_t, int> map;
            std::vector<bool> used_point(pts.size(), false);

            for (const auto& c : candidates) {
                if (used_point[c.point_idx])
                    continue;

                const std::int64_t key = gridKey(c.i, c.j);

                if (map.find(key) != map.end())
                    continue;

                map[key] = c.point_idx;
                used_point[c.point_idx] = true;

                current.corners[c.point_idx].i = c.i;
                current.corners[c.point_idx].j = c.j;
            }

            int min_i = std::numeric_limits<int>::max();
            int max_i = std::numeric_limits<int>::lowest();
            int min_j = std::numeric_limits<int>::max();
            int max_j = std::numeric_limits<int>::lowest();

            for (const auto& kv : map) {
                const int i = keyI(kv.first);
                const int j = keyJ(kv.first);

                min_i = std::min(min_i, i);
                max_i = std::max(max_i, i);
                min_j = std::min(min_j, j);
                max_j = std::max(max_j, j);
            }

            for (const auto& kv : map) {
                const int i = keyI(kv.first);
                const int j = keyJ(kv.first);

                const auto k00 = map.find(gridKey(i, j));
                const auto k10 = map.find(gridKey(i + 1, j));
                const auto k01 = map.find(gridKey(i, j + 1));
                const auto k11 = map.find(gridKey(i + 1, j + 1));

                if (k00 == map.end() || k10 == map.end() ||
                    k01 == map.end() || k11 == map.end())
                {
                    continue;
                }

                const cv::Point2f& p00 = current.corners[k00->second].uv;
                const cv::Point2f& p10 = current.corners[k10->second].uv;
                const cv::Point2f& p01 = current.corners[k01->second].uv;
                const cv::Point2f& p11 = current.corners[k11->second].uv;

                if (!isReasonableCellGeometry(p00, p10, p01, p11, spacing))
                    continue;

                GridCell cell;
                cell.i = i;
                cell.j = j;
                cell.corner_indices = {
                    k00->second,
                    k10->second,
                    k01->second,
                    k11->second
                };
                cell.center_uv = 0.25f * (p00 + p10 + p01 + p11);

                current.cells.push_back(cell);
            }

            if (!map.empty()) {
                current.cols = max_i - min_i + 1;
                current.rows = max_j - min_j + 1;
            }

            const int score =
                int(current.cells.size()) * 100
                + int(map.size());

            if (score > best_score) {
                best_score = score;
                best = std::move(current);
            }
        }

        return best;
    }

    void CheckerboardDetector::initializeTracking(
        const cv::Mat& gray,
        const CheckerboardDetection& det
    )
    {
        tracked_grid_points_.clear();
        tracked_image_points_.clear();
        H_grid_to_image_.release();

        for (const auto& c : det.corners) {
            if (c.i < 0 || c.j < 0)
                continue;

            if (!insideImage(gray, c.uv))
                continue;

            tracked_grid_points_.push_back(
                cv::Point2f(float(c.i), float(c.j))
            );

            tracked_image_points_.push_back(c.uv);
        }

        if (tracked_grid_points_.size() < 4) {
            resetTracking();
            return;
        }

        H_grid_to_image_ = cv::findHomography(
            tracked_grid_points_,
            tracked_image_points_,
            cv::RANSAC,
            5.0
        );

        if (H_grid_to_image_.empty()) {
            resetTracking();
            return;
        }

        gray.copyTo(prev_gray_);
        tracking_active_ = true;
        lost_counter_ = 0;
    }

    std::optional<CheckerboardDetection>
        CheckerboardDetector::trackFromPrevious(const cv::Mat& gray)
    {
        if (!tracking_active_ ||
            prev_gray_.empty() ||
            tracked_image_points_.size() < 4)
        {
            resetTracking();
            return std::nullopt;
        }

        std::vector<cv::Point2f> next_points;
        std::vector<unsigned char> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(
            prev_gray_,
            gray,
            tracked_image_points_,
            next_points,
            status,
            err,
            cv::Size(21, 21),
            3,
            cv::TermCriteria(
                cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                30,
                0.01
            )
        );

        std::vector<cv::Point2f> good_grid;
        std::vector<cv::Point2f> good_img;

        for (int k = 0; k < int(next_points.size()); ++k) {
            if (!status[k])
                continue;

            if (!insideImage(gray, next_points[k]))
                continue;

            if (err[k] > 35.0f)
                continue;

            good_grid.push_back(tracked_grid_points_[k]);
            good_img.push_back(next_points[k]);
        }

        if (int(good_grid.size()) < config_.min_tracking_points) {
            return std::nullopt;
        }

        cv::Mat inlier_mask;

        cv::Mat H = cv::findHomography(
            good_grid,
            good_img,
            cv::RANSAC,
            config_.max_tracking_reproj_error_px,
            inlier_mask
        );

        if (H.empty())
            return std::nullopt;

        std::vector<cv::Point2f> inlier_grid;
        std::vector<cv::Point2f> inlier_img;

        for (int k = 0; k < int(good_grid.size()); ++k) {
            if (inlier_mask.empty() || inlier_mask.at<unsigned char>(k)) {
                inlier_grid.push_back(good_grid[k]);
                inlier_img.push_back(good_img[k]);
            }
        }

        if (int(inlier_grid.size()) < config_.min_tracking_points)
            return std::nullopt;

        tracked_grid_points_ = inlier_grid;
        tracked_image_points_ = inlier_img;
        H_grid_to_image_ = H;

        return buildDetectionFromTrackedModel(
            gray,
            tracked_grid_points_,
            tracked_image_points_,
            H_grid_to_image_
        );
    }

    CheckerboardDetection
        CheckerboardDetector::buildDetectionFromTrackedModel(
            const cv::Mat& gray,
            const std::vector<cv::Point2f>& grid_points,
            const std::vector<cv::Point2f>& image_points,
            const cv::Mat& H
        ) const
    {
        CheckerboardDetection out;
        out.tracking = true;

        if (grid_points.empty() || image_points.empty() || H.empty())
            return out;

        std::unordered_map<std::int64_t, int> map;

        int min_i = std::numeric_limits<int>::max();
        int max_i = std::numeric_limits<int>::lowest();
        int min_j = std::numeric_limits<int>::max();
        int max_j = std::numeric_limits<int>::lowest();

        for (int k = 0; k < int(grid_points.size()); ++k) {
            const int i = int(std::llround(grid_points[k].x));
            const int j = int(std::llround(grid_points[k].y));

            const cv::Point2f& p = image_points[k];

            if (!insideImage(gray, p))
                continue;

            const int idx = int(out.corners.size());

            out.corners.push_back({ i, j, p });
            map[gridKey(i, j)] = idx;

            min_i = std::min(min_i, i);
            max_i = std::max(max_i, i);
            min_j = std::min(min_j, j);
            max_j = std::max(max_j, j);
        }

        if (out.corners.empty())
            return out;

        out.cols = max_i - min_i + 1;
        out.rows = max_j - min_j + 1;

        const double spacing = estimateGridSpacing(image_points);

        for (const auto& kv : map) {
            const int i = keyI(kv.first);
            const int j = keyJ(kv.first);

            const auto k00 = map.find(gridKey(i, j));
            const auto k10 = map.find(gridKey(i + 1, j));
            const auto k01 = map.find(gridKey(i, j + 1));
            const auto k11 = map.find(gridKey(i + 1, j + 1));

            if (k00 == map.end() || k10 == map.end() ||
                k01 == map.end() || k11 == map.end())
            {
                continue;
            }

            const cv::Point2f& p00 = out.corners[k00->second].uv;
            const cv::Point2f& p10 = out.corners[k10->second].uv;
            const cv::Point2f& p01 = out.corners[k01->second].uv;
            const cv::Point2f& p11 = out.corners[k11->second].uv;

            if (!isReasonableCellGeometry(p00, p10, p01, p11, spacing))
                continue;

            GridCell cell;
            cell.i = i;
            cell.j = j;
            cell.corner_indices = {
                k00->second,
                k10->second,
                k01->second,
                k11->second
            };
            cell.center_uv = 0.25f * (p00 + p10 + p01 + p11);

            out.cells.push_back(cell);
        }

        out.tracking = true;
        return out;
    }

} // namespace hydramarker