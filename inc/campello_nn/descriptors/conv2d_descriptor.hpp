#pragma once

#include <cstdint>

namespace systems::leal::campello_nn
{

    /**
     * @brief Parameters for `GraphBuilder::conv2d()`.
     *
     * Input is NCHW (`[batch, channels, height, width]`); weights are OIHW
     * (`[outChannels, inChannels/groups, kernelHeight, kernelWidth]`). Padding is
     * explicit, matching the rest of the op set (no auto "same"/"valid" styles).
     * Bias is not fused — add it separately with `add()` after broadcasting it to
     * `[1, outChannels, 1, 1]`.
     */
    struct Conv2dDescriptor
    {
        int64_t strideX = 1;
        int64_t strideY = 1;
        int64_t dilationX = 1;
        int64_t dilationY = 1;
        int64_t paddingLeft = 0;
        int64_t paddingRight = 0;
        int64_t paddingTop = 0;
        int64_t paddingBottom = 0;
        int64_t groups = 1;
    };

} // namespace systems::leal::campello_nn
