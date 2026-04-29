cmake_minimum_required(VERSION 3.14)
project(hydramarker_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Dependencies
# ============================================================

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

# ============================================================
# Python module (pybind11)
# ============================================================

pybind11_add_module(hydramarker_cpp
    bindings.cpp
    marker_field.cpp
    checkerboard_detector.cpp
)

target_link_libraries(hydramarker_cpp
    PRIVATE
    ${OpenCV_LIBS}
)

# ============================================================
# Example executable
# ============================================================

add_executable(example
    example.cpp
    generator_HydraMarker.cpp
    marker_field.cpp
    checkerboard_detector.cpp
)

target_link_libraries(example
    PRIVATE
    ${OpenCV_LIBS}
)