#pragma once

namespace systems::leal::campello_nn
{

    /**
     * @brief Sampling mode for `GraphBuilder::resize()`. Mirrors `MPSGraphResizeMode`.
     */
    enum class ResizeMode
    {
        Nearest,
        Bilinear
    };

}
