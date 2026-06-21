#pragma once

#include <cstdint>

namespace systems::leal::campello_nn
{

    /**
     * @brief Parameters for `GraphBuilder::maxPool2d()` / `avgPool2d()`.
     *
     * Input is NCHW (`[batch, channels, height, width]`), matching `Conv2dDescriptor`.
     * Padding is explicit.
     */
    struct Pool2dDescriptor
    {
        int64_t kernelHeight = 1;
        int64_t kernelWidth = 1;
        int64_t strideX = 1;
        int64_t strideY = 1;
        int64_t paddingLeft = 0;
        int64_t paddingRight = 0;
        int64_t paddingTop = 0;
        int64_t paddingBottom = 0;
    };

} // namespace systems::leal::campello_nn
