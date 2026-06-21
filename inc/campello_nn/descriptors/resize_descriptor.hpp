#pragma once

#include <cstdint>
#include <campello_nn/constants/resize_mode.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Parameters for `GraphBuilder::resize()`.
     *
     * Input/output are NCHW, matching `Conv2dDescriptor`/`Pool2dDescriptor`. Defaults
     * (`centerResult = true`, `alignCorners = false`) match OpenCV's and TensorFlow
     * v2's resize behavior, per `MPSGraphResizeOps`'s own documented convention.
     * `alignCorners = true` ignores `centerResult`.
     */
    struct ResizeDescriptor
    {
        int64_t outputHeight;
        int64_t outputWidth;
        ResizeMode mode = ResizeMode::Bilinear;
        bool centerResult = true;
        bool alignCorners = false;
        // Nearest mode only: round down (floor) instead of round-to-nearest when
        // picking the source pixel. Matches ONNX Resize's `nearest_mode="floor"`,
        // which differs from this struct's default (round-to-nearest, matching
        // MPSGraph's own default `RoundPreferCeil` for the generic resize call).
        bool nearestRoundsDown = false;
    };

} // namespace systems::leal::campello_nn
