#include <iostream>
#include <campello_nn/context.hpp>
#include "context_data.hpp"
#include "resource_data.hpp"
#include "../cpu/cpu_backend.hpp"
#ifdef __APPLE__
#include "../metal/mps_backend.hpp"
#endif

using namespace systems::leal::campello_nn;

Context::Context(void *pd)
{
    native = pd;
}

Context::~Context()
{
    delete (ContextData *)native;
}

std::shared_ptr<Context> Context::create(const ContextDescriptor &desc)
{
    std::unique_ptr<Backend> backend;

#ifdef __APPLE__
    // MPSGraph's own placement pass (MPSGraphOptimizationLevel1, the framework
    // default) can already dispatch eligible ops to the GPU or ANE, so Gpu/Npu/
    // Default all route to the same MPSGraph backend; only Cpu stays on CpuBackend.
    if (desc.deviceType == DeviceType::Cpu)
        backend = std::make_unique<CpuBackend>();
    else
        backend = std::make_unique<MpsBackend>();
#else
    if (desc.deviceType != DeviceType::Cpu && desc.deviceType != DeviceType::Default)
    {
        std::cerr << "campello_nn: no accelerator backend implemented yet for the requested "
                     "DeviceType; falling back to Cpu.\n";
    }
    backend = std::make_unique<CpuBackend>();
#endif

    auto data = new ContextData{std::move(backend), desc.deviceType};
    return std::shared_ptr<Context>(new Context((void *)data));
}

std::shared_ptr<Tensor> Context::createTensor(const TensorDescriptor &desc)
{
    auto data = (ContextData *)native;
    void *raw = data->backend->createTensor(desc);
    auto td = new TensorData{data->backend.get(), raw};
    return std::shared_ptr<Tensor>(new Tensor((void *)td, desc));
}

std::shared_ptr<Fence> Context::dispatch(
    const Graph &graph,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &inputs,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &outputs)
{
    auto data = (ContextData *)native;
    auto gdata = (GraphData *)graph.native;

    std::unordered_map<std::string, void *> inputNatives;
    for (auto &[name, tensor] : inputs)
        inputNatives[name] = ((TensorData *)tensor->native)->native;

    std::unordered_map<std::string, void *> outputNatives;
    for (auto &[name, tensor] : outputs)
        outputNatives[name] = ((TensorData *)tensor->native)->native;

    void *fenceNative = data->backend->dispatch(gdata->native, inputNatives, outputNatives);
    auto fd = new FenceData{data->backend.get(), fenceNative};
    return std::shared_ptr<Fence>(new Fence((void *)fd));
}
