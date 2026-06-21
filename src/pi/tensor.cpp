#include <campello_nn/tensor.hpp>
#include "resource_data.hpp"

using namespace systems::leal::campello_nn;

Tensor::Tensor(void *pd, const TensorDescriptor &desc) : native(pd), descriptor(desc) {}

Tensor::~Tensor()
{
    auto data = (TensorData *)native;
    data->backend->destroyTensor(data->native);
    delete data;
}

const std::vector<int64_t> &Tensor::shape() const
{
    return descriptor.shape;
}

DataType Tensor::dataType() const
{
    return descriptor.dataType;
}

void Tensor::write(const void *data, size_t size)
{
    auto td = (TensorData *)native;
    td->backend->writeTensor(td->native, data, size);
}

void Tensor::read(void *data, size_t size) const
{
    auto td = (TensorData *)native;
    td->backend->readTensor(td->native, data, size);
}
