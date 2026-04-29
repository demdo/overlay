/*
 * Pybind11 bindings for the HydraMarker C++ backend.
 *
 * This file exposes selected C++ classes to Python so that the Python overlay
 * pipeline can use the performance-critical C++ implementation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "marker_field.hpp"
#include "checkerboard_detector.hpp"

namespace py = pybind11;
using namespace hydramarker;

PYBIND11_MODULE(hydramarker_cpp, m)
{
    py::class_<PatchMatch>(m, "PatchMatch")
        .def_readonly("x", &PatchMatch::x)
        .def_readonly("y", &PatchMatch::y)
        .def_readonly("rotation_deg", &PatchMatch::rotation_deg);

    py::class_<MarkerField>(m, "MarkerField")
        .def(py::init([](const std::string& path) {
        return MarkerField::loadFromFile(path);
            }))
        .def("width", &MarkerField::width)
        .def("height", &MarkerField::height)
        .def("patch_size", &MarkerField::patchSize)
        .def("find_patch", &MarkerField::findPatch);

    py::class_<GridCorner>(m, "GridCorner")
        .def_readonly("i", &GridCorner::i)
        .def_readonly("j", &GridCorner::j)
        .def_property_readonly("uv", [](const GridCorner& c) {
        return py::make_tuple(c.uv.x, c.uv.y);
            });

    py::class_<GridCell>(m, "GridCell")
        .def_readonly("i", &GridCell::i)
        .def_readonly("j", &GridCell::j)
        .def_readonly("corner_indices", &GridCell::corner_indices)
        .def_property_readonly("center_uv", [](const GridCell& c) {
        return py::make_tuple(c.center_uv.x, c.center_uv.y);
            });

    py::class_<CheckerboardDetection>(m, "CheckerboardDetection")
        .def_readonly("corners", &CheckerboardDetection::corners)
        .def_readonly("cells", &CheckerboardDetection::cells)
        .def_readonly("cols", &CheckerboardDetection::cols)
        .def_readonly("rows", &CheckerboardDetection::rows)
        .def_property_readonly("valid", [](const CheckerboardDetection& d) {
        return d.valid();
            });

    py::class_<CheckerboardDetector>(m, "CheckerboardDetector")
        .def(py::init<const MarkerField&>())
        .def(
            "detect",
            [](
                const CheckerboardDetector& self,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img
                ) -> py::object
            {
                auto buf = img.request();

                if (buf.ndim != 2 && buf.ndim != 3) {
                    throw std::runtime_error(
                        "Image must be HxW grayscale or HxWx3 uint8 color."
                    );
                }

                const int h = static_cast<int>(buf.shape[0]);
                const int w = static_cast<int>(buf.shape[1]);

                cv::Mat mat;

                if (buf.ndim == 2) {
                    mat = cv::Mat(h, w, CV_8UC1, buf.ptr);
                }
                else {
                    const int channels = static_cast<int>(buf.shape[2]);

                    if (channels != 3) {
                        throw std::runtime_error(
                            "Color image must have exactly 3 channels."
                        );
                    }

                    mat = cv::Mat(h, w, CV_8UC3, buf.ptr);
                }

                auto result = self.detect(mat);

                if (!result) {
                    return py::none();
                }

                return py::cast(*result);
            }
        );
}