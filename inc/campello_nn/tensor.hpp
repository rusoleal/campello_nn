#pragma once

#include <cstddef>
#include <vector>
#include <campello_nn/constants/data_type.hpp>
#include <campello_nn/descriptors/tensor_descriptor.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief A device-resident tensor — backend-owned storage for a fixed shape/dtype.
     *
     * Tensors are created exclusively by `Context::createTensor()`. They are bound
     * by name as graph inputs/outputs to `Context::dispatch()`. `write()`/`read()`
     * require the tensor to have been created with `writable`/`readable` set on its
     * `TensorDescriptor`.
     */
    class Tensor
    {
        friend class Context;
        void *native;
        TensorDescriptor descriptor;

        Tensor(void *pd, const TensorDescriptor &desc);

    public:
        ~Tensor();

        const std::vector<int64_t> &shape() const;
        DataType dataType() const;

        /**
         * @brief Uploads CPU data into the tensor. Requires `writable` at creation.
         * @param data Pointer to source data.
         * @param size Number of bytes to copy.
         */
        void write(const void *data, size_t size);

        /**
         * @brief Downloads tensor data to CPU memory. Requires `readable` at creation.
         * @param data Pointer to destination memory (must be at least `size` bytes).
         * @param size Number of bytes to copy.
         */
        void read(void *data, size_t size) const;
    };

} // namespace systems::leal::campello_nn
