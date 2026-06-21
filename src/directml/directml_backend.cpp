#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
// DirectML.h gates DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC/DML_RESAMPLE2_OPERATOR_DESC
// (and DML_OPERATOR_ACTIVATION_SOFTMAX1/DML_OPERATOR_RESAMPLE2) behind
// `#if DML_TARGET_VERSION >= 0x5100`; without this, the default resolved from
// <windows.h>'s NTDDI_VERSION can land below that. This is the redistributable
// NuGet package, not the in-box SDK, so always target its latest feature level.
#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "directml_backend.hpp"
#include "../pi/ir.hpp"

using namespace systems::leal::campello_nn;
using Microsoft::WRL::ComPtr;

// DirectML requires DML_FEATURE_LEVEL_5_1 (ACTIVATION_SOFTMAX1, RESAMPLE2) —
// checked explicitly at device creation rather than discovered op-by-op.
static const DML_FEATURE_LEVEL kRequiredFeatureLevel = DML_FEATURE_LEVEL_5_1;

namespace
{
    void throwIfFailed(HRESULT hr, const char *what)
    {
        if (FAILED(hr))
        {
            char hex[16];
            std::snprintf(hex, sizeof(hex), "%08lX", (unsigned long)hr);
            throw std::runtime_error(std::string("campello_nn: DirectML backend: ") + what +
                                      " failed (hr=0x" + hex + ")");
        }
    }

    // Same as throwIfFailed, but on DXGI_ERROR_DEVICE_REMOVED also queries the
    // device's *actual* removal reason — the outer HRESULT is always just
    // 0x887A0005 regardless of cause, which isn't actionable on its own.
    void throwIfFailed(HRESULT hr, const char *what, ID3D12Device *device)
    {
        if (FAILED(hr))
        {
            if (hr == DXGI_ERROR_DEVICE_REMOVED && device)
            {
                HRESULT reason = device->GetDeviceRemovedReason();
                char hex[16];
                std::snprintf(hex, sizeof(hex), "%08lX", (unsigned long)reason);
                throw std::runtime_error(std::string("campello_nn: DirectML backend: ") + what +
                                          " failed: device removed (reason=0x" + hex + ")");
            }
            throwIfFailed(hr, what);
        }
    }

    size_t elementByteSize(DataType dt)
    {
        switch (dt)
        {
        case DataType::Float32:
        case DataType::Int32:
        case DataType::Uint32:
            return 4;
        case DataType::Float16:
            return 2;
        case DataType::Int8:
            return 1;
        }
        throw std::runtime_error("campello_nn: unknown DataType");
    }

    int64_t numElements(const std::vector<int64_t> &shape)
    {
        int64_t n = 1;
        for (auto d : shape)
            n *= d;
        return n;
    }

    // DirectML requires every bound buffer's size (and DML_BUFFER_TENSOR_DESC's
    // TotalTensorSizeInBytes, see TensorDescBuilder::finish below) to be
    // DWORD-aligned. Element counts times Float16 (2 bytes)/Int8 (1 byte) can
    // produce odd byte sizes that violate this, so every GPU buffer backing
    // tensor data (node outputs, user-facing Tensors) rounds up through this.
    UINT64 dmlAlignedByteSize(UINT64 raw)
    {
        return (raw + 3) & ~UINT64(3);
    }

    DML_TENSOR_DATA_TYPE dmlDataType(DataType dt)
    {
        switch (dt)
        {
        case DataType::Float32: return DML_TENSOR_DATA_TYPE_FLOAT32;
        case DataType::Float16: return DML_TENSOR_DATA_TYPE_FLOAT16;
        case DataType::Int32: return DML_TENSOR_DATA_TYPE_INT32;
        case DataType::Uint32: return DML_TENSOR_DATA_TYPE_UINT32;
        case DataType::Int8: return DML_TENSOR_DATA_TYPE_INT8;
        }
        throw std::runtime_error("campello_nn: unknown DataType");
    }

    std::vector<UINT> toUintVec(const std::vector<int64_t> &v)
    {
        std::vector<UINT> r(v.size());
        for (size_t i = 0; i < v.size(); i++)
            r[i] = (UINT)v[i];
        return r;
    }

    // Row-major contiguous strides for `shape` (same convention as src/cpu/strides.hpp).
    std::vector<int64_t> rowMajorStrides(const std::vector<int64_t> &shape)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
        return strides;
    }

    // Owns the Sizes/Strides arrays for one DML_BUFFER_TENSOR_DESC. DML copies the
    // desc contents during CreateOperator, so this only needs to outlive that call.
    struct TensorDescBuilder
    {
        std::vector<UINT> sizes;
        std::vector<UINT> strides;
        DML_BUFFER_TENSOR_DESC buffer{};
        DML_TENSOR_DESC tensor{};

        // Plain (non-broadcast) tensor desc: packed/contiguous strides.
        void setPacked(const std::vector<int64_t> &shape, DataType dt)
        {
            sizes = toUintVec(shape);
            strides.clear();
            finish(dt);
        }

        // Broadcasting tensor desc: `shape` (the operand's own, possibly lower-rank
        // or size-1-dim shape) read as if it had `outputShape`'s shape, via explicit
        // strides with 0 in every broadcast dimension. NumPy/ONNX right-aligned
        // broadcasting, matching computeBroadcastShape()'s semantics.
        void setBroadcast(const std::vector<int64_t> &shape, const std::vector<int64_t> &outputShape, DataType dt)
        {
            // No actual broadcasting needed (operand's own shape already matches
            // the output) — emit a plain packed descriptor (Strides=nullptr)
            // rather than an explicit-but-trivial strides array. Functionally
            // identical, matches the encoding used elsewhere (e.g. Gelu's
            // internal multiply) when no broadcast is required.
            if (shape == outputShape)
            {
                setPacked(outputShape, dt);
                return;
            }

            size_t rank = outputShape.size();
            std::vector<int64_t> padded(rank, 1);
            for (size_t i = 0; i < shape.size(); i++)
                padded[rank - shape.size() + i] = shape[i];
            auto packedStrides = rowMajorStrides(padded);

            sizes = toUintVec(outputShape);
            strides.resize(rank);
            for (size_t i = 0; i < rank; i++)
                strides[i] = (UINT)((padded[i] == 1 && outputShape[i] != 1) ? 0 : packedStrides[i]);
            finish(dt);
        }

        // Strided (e.g. materializing Transpose) tensor desc: explicit strides given
        // directly, iteration domain `shape`.
        void setStrided(const std::vector<int64_t> &shape, const std::vector<int64_t> &explicitStrides, DataType dt)
        {
            sizes = toUintVec(shape);
            strides = toUintVec(explicitStrides);
            finish(dt);
        }

        // BatchNorm/InstanceNorm's per-channel [C] mean/variance/scale/bias broadcast
        // against NCHW's *second* axis — not NumPy right-alignment (that would land
        // [C] against the last/W axis instead). Mirrors MPSGraph's reshapeChannel:
        // logically [C] -> [1,C,1,1], expressed here as explicit strides instead of
        // an actual reshape op.
        void setChannelBroadcast(int64_t channelCount, const std::vector<int64_t> &outputShape, DataType dt)
        {
            (void)channelCount;
            sizes = toUintVec(outputShape);
            strides.assign(outputShape.size(), 0);
            strides[1] = 1; // NCHW channel axis is index 1
            finish(dt);
        }

    private:
        void finish(DataType dt)
        {
            buffer = {};
            buffer.DataType = dmlDataType(dt);
            buffer.Flags = DML_TENSOR_FLAG_NONE;
            buffer.DimensionCount = (UINT)sizes.size();
            buffer.Sizes = sizes.data();
            buffer.Strides = strides.empty() ? nullptr : strides.data();

            UINT64 elemSize = elementByteSizeFromDmlType(buffer.DataType);
            UINT64 byteSize;
            if (strides.empty())
            {
                UINT64 elemCount = 1;
                for (auto s : sizes)
                    elemCount *= s;
                byteSize = elemCount * elemSize;
            }
            else
            {
                // Minimum buffer size implied by the strides actually used, not
                // simply product(sizes) — required when Strides has zero entries
                // (broadcast) or is otherwise non-packed (Transpose materialization).
                UINT64 lastElementOffset = 0;
                for (size_t i = 0; i < sizes.size(); i++)
                    if (sizes[i] > 0)
                        lastElementOffset += (UINT64)(sizes[i] - 1) * strides[i];
                byteSize = (lastElementOffset + 1) * elemSize;
            }
            // DirectML requires TotalTensorSizeInBytes to be DWORD-aligned (a
            // multiple of 4) — CreateOperator returns E_INVALIDARG otherwise. Bites
            // odd-count Float16 (2 bytes/elem) or Int8 (1 byte/elem) tensors, which
            // product(sizes)*elemSize alone doesn't guarantee.
            byteSize = (byteSize + 3) & ~UINT64(3);
            buffer.TotalTensorSizeInBytes = byteSize;
            buffer.GuaranteedBaseOffsetAlignment = 0;
            tensor.Type = DML_TENSOR_TYPE_BUFFER;
            tensor.Desc = &buffer;
        }

        static UINT64 elementByteSizeFromDmlType(DML_TENSOR_DATA_TYPE dt)
        {
            switch (dt)
            {
            case DML_TENSOR_DATA_TYPE_FLOAT32:
            case DML_TENSOR_DATA_TYPE_INT32:
            case DML_TENSOR_DATA_TYPE_UINT32:
                return 4;
            case DML_TENSOR_DATA_TYPE_FLOAT16:
                return 2;
            case DML_TENSOR_DATA_TYPE_INT8:
                return 1;
            default:
                throw std::runtime_error("campello_nn: unknown DML_TENSOR_DATA_TYPE");
            }
        }
    };
} // namespace

struct DmlTensor
{
    ComPtr<ID3D12Resource> buffer; // DEFAULT heap, always kept in UNORDERED_ACCESS state
    TensorDescriptor desc;
    UINT64 byteSize;
};

// One input slot for a compiled operator's binding table: either resolved fresh
// on every dispatch() call (graph Input, or anything reachable through a chain of
// Reshape aliases back to one) or fixed once at compile time (a Constant node's
// buffer, an intermediate node's own output buffer, or a synthetic constant this
// backend created itself — e.g. QuantizeLinear's scale/zeroPoint tensors).
struct DmlBindingSource
{
    bool dynamic = false;
    size_t irNodeIndex = 0;        // valid when dynamic
    ComPtr<ID3D12Resource> fixed;  // valid when !dynamic
    UINT64 byteSize = 0;
    // GEMM's CTensor is _Maybenull_ in DML_GEMM_OPERATOR_DESC, but the operator's
    // *binding* slot count is fixed by its schema (A, B, C) regardless — omitting
    // the slot entirely from BindInputs left it unbound and produced zeroed
    // output. A `none` source still occupies its index, bound as
    // DML_BINDING_TYPE_NONE.
    bool none = false;
};

struct DmlOpNode
{
    ComPtr<IDMLCompiledOperator> compiledOp;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<IDMLBindingTable> bindingTable;
    ComPtr<ID3D12Resource> persistent;
    UINT64 persistentSize = 0;
    ComPtr<ID3D12Resource> temporary;
    UINT64 temporarySize = 0;
    std::vector<DmlBindingSource> inputs;
    ComPtr<ID3D12Resource> output;
    UINT64 outputByteSize = 0;
};

struct DmlCompiledGraph
{
    // Indexed by IR node index. Populated for every node kind: Input/Reshape stay
    // null forever (resolved dynamically, see resolveBuffer below); Constant and
    // every real-op node get a fixed buffer allocated at compile time.
    std::vector<ComPtr<ID3D12Resource>> nodeOutput;
    std::vector<UINT64> nodeByteSize;
    std::vector<DmlOpNode> opNodes; // every real compute op, in dispatch order
    GraphIR ir; // owned copy, so node.kind/inputs/shape survive past compileGraph()
    std::unordered_map<std::string, size_t> outputNodeForName;
};

struct DmlFence
{
    bool signaled = true;
};

struct DirectMlBackend::Impl
{
    ComPtr<ID3D12Device> device;
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<IDMLCommandRecorder> recorder;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 0;

    // One-shot resources (e.g. operator-initializer scratch buffers) that only
    // need to outlive the command list batch currently being recorded — kept
    // alive here since a *resource* referenced by a recorded command isn't
    // implicitly kept alive by the command list. Released once executeAndWait()
    // confirms the GPU has finished with them.
    std::vector<ComPtr<ID3D12Resource>> pendingScratch;

    // Same lifetime requirement as pendingScratch, but for descriptor heaps
    // (e.g. an operator initializer's one-shot heap) — a *heap* bound via
    // SetDescriptorHeaps is a genuine GPU-visible reference recorded into the
    // command list, unlike resources. Letting one go out of scope before the
    // matching executeAndWait() happens to work on real hardware drivers
    // (which don't reclaim the freed virtual memory immediately) but WARP
    // correctly detects it and faults the device — confirmed via the D3D12
    // debug layer ("An ID3D12DescriptorHeap object ... was deleted prior to
    // closing the command list").
    std::vector<ComPtr<ID3D12DescriptorHeap>> pendingScratchHeaps;

    // Submits whatever has been recorded on `cmdList`, waits synchronously for
    // completion, then resets the allocator/list for the next batch of recording.
    // Every public Backend method that touches the GPU (tensor I/O, compileGraph's
    // constant uploads + operator initialization, dispatch) ends with this, so the
    // whole backend is synchronous from the caller's perspective, matching the
    // CPU/MPSGraph backends.
    void executeAndWait()
    {
        throwIfFailed(cmdList->Close(), "ID3D12GraphicsCommandList::Close");
        ID3D12CommandList *lists[] = {cmdList.Get()};
        queue->ExecuteCommandLists(1, lists);
        fenceValue++;
        throwIfFailed(queue->Signal(fence.Get(), fenceValue), "ID3D12CommandQueue::Signal");
        if (fence->GetCompletedValue() < fenceValue)
        {
            throwIfFailed(fence->SetEventOnCompletion(fenceValue, fenceEvent), "ID3D12Fence::SetEventOnCompletion");
            WaitForSingleObject(fenceEvent, INFINITE);
        }
        throwIfFailed(allocator->Reset(), "ID3D12CommandAllocator::Reset");
        throwIfFailed(cmdList->Reset(allocator.Get(), nullptr), "ID3D12GraphicsCommandList::Reset");
        pendingScratch.clear();
        pendingScratchHeaps.clear();
    }
};

namespace
{
    ComPtr<IDXGIAdapter1> pickAdapter()
    {
        ComPtr<IDXGIFactory6> factory;
        throwIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

        // EnumAdapterByGpuPreference can still hand back a software-flagged
        // adapter (e.g. the "Microsoft Basic Render Driver" some CI runners
        // expose as a display-only placeholder, which is not the same thing as
        // — and far less robust for compute than — DirectML's real WARP
        // fallback below) even when asked for DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE.
        // Reject anything DXGI_ADAPTER_FLAG_SOFTWARE so we only ever take this
        // branch for a genuine hardware GPU.
        ComPtr<IDXGIAdapter1> adapter;
        if (SUCCEEDED(factory->EnumAdapterByGpuPreference(
                0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter))))
        {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
                return adapter;
        }

        // No (real) hardware adapter (e.g. a CI runner with no GPU) — fall back
        // to WARP, which fully supports this op set at our required feature level.
        throwIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "IDXGIFactory4::EnumWarpAdapter");
        return adapter;
    }

    ComPtr<ID3D12Resource> createBuffer(
        ID3D12Device *device, UINT64 byteSize, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_FLAGS flags,
        D3D12_RESOURCE_STATES initialState)
    {
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = heapType;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Alignment = 0;
        desc.Width = byteSize > 0 ? byteSize : 1;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;

        ComPtr<ID3D12Resource> resource;
        throwIfFailed(device->CreateCommittedResource(
                          &heapProps, D3D12_HEAP_FLAG_NONE, &desc, initialState, nullptr, IID_PPV_ARGS(&resource)),
                      "ID3D12Device::CreateCommittedResource", device);
        return resource;
    }

    // DEFAULT-heap, UAV-capable buffer — used for every tensor DirectML touches
    // (graph inputs/outputs, intermediate node outputs, constants, persistent and
    // temporary operator resources). Always kept in UNORDERED_ACCESS state except
    // for the brief COPY_DEST/COPY_SOURCE window around a staging copy.
    ComPtr<ID3D12Resource> createDmlBuffer(ID3D12Device *device, UINT64 byteSize)
    {
        return createBuffer(device, byteSize, D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    void uavBarrier(ID3D12GraphicsCommandList *cmdList)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = nullptr; // all UAVs — conservative, see directml backend decisions
        cmdList->ResourceBarrier(1, &barrier);
    }

    void transition(ID3D12GraphicsCommandList *cmdList, ID3D12Resource *resource,
                     D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.StateBefore = from;
        barrier.Transition.StateAfter = to;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList->ResourceBarrier(1, &barrier);
    }
} // namespace

DirectMlBackend::DirectMlBackend()
{
    impl = new Impl();

    ComPtr<IDXGIAdapter1> adapter = pickAdapter();
    throwIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&impl->device)),
                  "D3D12CreateDevice");

    throwIfFailed(DMLCreateDevice(impl->device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&impl->dmlDevice)),
                  "DMLCreateDevice");

    DML_FEATURE_LEVEL levels[] = {kRequiredFeatureLevel};
    DML_FEATURE_QUERY_FEATURE_LEVELS query{(UINT)1, levels};
    DML_FEATURE_DATA_FEATURE_LEVELS support{};
    throwIfFailed(impl->dmlDevice->CheckFeatureSupport(DML_FEATURE_FEATURE_LEVELS, sizeof(query), &query,
                                                        sizeof(support), &support),
                  "IDMLDevice::CheckFeatureSupport");
    if (support.MaxSupportedFeatureLevel < kRequiredFeatureLevel)
        throw std::runtime_error("campello_nn: DirectML backend requires feature level 5.1 or higher");

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    throwIfFailed(impl->device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&impl->queue)),
                  "ID3D12Device::CreateCommandQueue");

    throwIfFailed(impl->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&impl->allocator)),
                  "ID3D12Device::CreateCommandAllocator");
    throwIfFailed(impl->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, impl->allocator.Get(), nullptr,
                                                   IID_PPV_ARGS(&impl->cmdList)),
                  "ID3D12Device::CreateCommandList");

    throwIfFailed(impl->dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&impl->recorder)),
                  "IDMLDevice::CreateCommandRecorder");

    throwIfFailed(impl->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&impl->fence)),
                  "ID3D12Device::CreateFence");
    impl->fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!impl->fenceEvent)
        throw std::runtime_error("campello_nn: DirectML backend: CreateEvent failed");
}

DirectMlBackend::~DirectMlBackend()
{
    if (impl->fenceEvent)
        CloseHandle(impl->fenceEvent);
    delete impl;
}

void *DirectMlBackend::createTensor(const TensorDescriptor &desc)
{
    UINT64 byteSize = dmlAlignedByteSize((UINT64)elementByteSize(desc.dataType) * (UINT64)numElements(desc.shape));
    auto t = new DmlTensor{createDmlBuffer(impl->device.Get(), byteSize), desc, byteSize};
    return t;
}

void DirectMlBackend::destroyTensor(void *native)
{
    delete (DmlTensor *)native;
}

void DirectMlBackend::writeTensor(void *native, const void *data, size_t size)
{
    auto t = (DmlTensor *)native;
    if ((UINT64)size > t->byteSize)
        throw std::runtime_error("campello_nn: write() exceeds tensor capacity");

    ComPtr<ID3D12Resource> upload =
        createBuffer(impl->device.Get(), size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                     D3D12_RESOURCE_STATE_GENERIC_READ);
    void *mapped = nullptr;
    throwIfFailed(upload->Map(0, nullptr, &mapped), "ID3D12Resource::Map");
    memcpy(mapped, data, size);
    upload->Unmap(0, nullptr);

    transition(impl->cmdList.Get(), t->buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
    impl->cmdList->CopyBufferRegion(t->buffer.Get(), 0, upload.Get(), 0, size);
    transition(impl->cmdList.Get(), t->buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl->executeAndWait();
}

void DirectMlBackend::readTensor(void *native, void *data, size_t size)
{
    auto t = (DmlTensor *)native;
    if ((UINT64)size > t->byteSize)
        throw std::runtime_error("campello_nn: read() exceeds tensor capacity");

    ComPtr<ID3D12Resource> readback =
        createBuffer(impl->device.Get(), size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                     D3D12_RESOURCE_STATE_COPY_DEST);

    transition(impl->cmdList.Get(), t->buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    impl->cmdList->CopyBufferRegion(readback.Get(), 0, t->buffer.Get(), 0, size);
    transition(impl->cmdList.Get(), t->buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    impl->executeAndWait();

    void *mapped = nullptr;
    throwIfFailed(readback->Map(0, nullptr, &mapped), "ID3D12Resource::Map");
    memcpy(data, mapped, size);
    readback->Unmap(0, nullptr);
}

namespace
{
    ComPtr<IDMLCompiledOperator> compileOp(DirectMlBackend::Impl *impl, const DML_OPERATOR_DESC &desc)
    {
        ComPtr<IDMLOperator> op;
        throwIfFailed(impl->dmlDevice->CreateOperator(&desc, IID_PPV_ARGS(&op)), "IDMLDevice::CreateOperator");
        ComPtr<IDMLCompiledOperator> compiled;
        throwIfFailed(impl->dmlDevice->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&compiled)),
                      "IDMLDevice::CompileOperator");
        return compiled;
    }

    // Runs IDMLOperatorInitializer once for a freshly compiled operator — required
    // before any operator (even a stateless one) can be dispatched. Writes the
    // operator's persistent resource (if any) and records onto impl->cmdList;
    // the caller flushes via impl->executeAndWait() once for the whole graph.
    void initializeOperator(DirectMlBackend::Impl *impl, DmlOpNode &opNode)
    {
        IDMLCompiledOperator *ops[] = {opNode.compiledOp.Get()};
        ComPtr<IDMLOperatorInitializer> initializer;
        throwIfFailed(impl->dmlDevice->CreateOperatorInitializer(1, ops, IID_PPV_ARGS(&initializer)),
                      "IDMLDevice::CreateOperatorInitializer");

        DML_BINDING_PROPERTIES initProps = initializer->GetBindingProperties();
        UINT descCount = initProps.RequiredDescriptorCount > 0 ? initProps.RequiredDescriptorCount : 1;

        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = descCount;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> initHeap;
        throwIfFailed(impl->device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&initHeap)),
                      "ID3D12Device::CreateDescriptorHeap");
        // Bound on the command list below via SetDescriptorHeaps and referenced
        // by the dispatch recorded against it — must outlive the matching
        // executeAndWait() at the end of compileGraph(), not just this function.
        impl->pendingScratchHeaps.push_back(initHeap);

        DML_BINDING_TABLE_DESC tableDesc{};
        tableDesc.Dispatchable = initializer.Get();
        tableDesc.CPUDescriptorHandle = initHeap->GetCPUDescriptorHandleForHeapStart();
        tableDesc.GPUDescriptorHandle = initHeap->GetGPUDescriptorHandleForHeapStart();
        tableDesc.SizeInDescriptors = descCount;
        ComPtr<IDMLBindingTable> initTable;
        throwIfFailed(impl->dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initTable)),
                      "IDMLDevice::CreateBindingTable");

        if (initProps.TemporaryResourceSize > 0)
        {
            ComPtr<ID3D12Resource> initTemp = createDmlBuffer(impl->device.Get(), initProps.TemporaryResourceSize);
            DML_BUFFER_BINDING tempBinding{};
            tempBinding.Buffer = initTemp.Get();
            tempBinding.SizeInBytes = initProps.TemporaryResourceSize;
            DML_BINDING_DESC tempDesc{};
            tempDesc.Type = DML_BINDING_TYPE_BUFFER;
            tempDesc.Desc = &tempBinding;
            initTable->BindTemporaryResource(&tempDesc);
            // Same lifetime requirement as initHeap above: must outlive the
            // matching executeAndWait() at the end of compileGraph().
            impl->pendingScratch.push_back(initTemp);
        }

        if (opNode.persistentSize > 0)
        {
            DML_BUFFER_BINDING persistBinding{};
            persistBinding.Buffer = opNode.persistent.Get();
            persistBinding.SizeInBytes = opNode.persistentSize;
            DML_BINDING_DESC persistDesc{};
            persistDesc.Type = DML_BINDING_TYPE_BUFFER;
            persistDesc.Desc = &persistBinding;
            initTable->BindOutputs(1, &persistDesc);
        }

        // RecordDispatch reads descriptors through whatever heap is currently bound
        // on the command list — it does not take the heap from the binding table.
        ID3D12DescriptorHeap *initHeaps[] = {initHeap.Get()};
        impl->cmdList->SetDescriptorHeaps(1, initHeaps);
        impl->recorder->RecordDispatch(impl->cmdList.Get(), initializer.Get(), initTable.Get());
        uavBarrier(impl->cmdList.Get());
    }
} // namespace

void *DirectMlBackend::compileGraph(const GraphIR &irIn)
{
    auto compiled = new DmlCompiledGraph();
    compiled->ir = irIn;
    const GraphIR &g = compiled->ir;
    size_t n = g.nodes.size();
    compiled->nodeOutput.resize(n);
    compiled->nodeByteSize.resize(n);

    auto dyn = [&](size_t idx) { return DmlBindingSource{true, idx, nullptr, compiled->nodeByteSize[idx]}; };
    auto fix = [](ComPtr<ID3D12Resource> r, UINT64 sz) { return DmlBindingSource{false, 0, r, sz}; };

    // Compiles+initializes `op`, allocates its output buffer, and records it as a
    // new DmlOpNode. `inputs` lists every bound input in the same order the
    // operator's *_OPERATOR_DESC declared its (non-null) tensor pointers.
    auto pushOp = [&](ComPtr<IDMLCompiledOperator> op, std::vector<DmlBindingSource> inputs,
                       UINT64 outputByteSize) -> ComPtr<ID3D12Resource>
    {
        DmlOpNode opNode;
        opNode.compiledOp = op;
        opNode.inputs = std::move(inputs);
        opNode.output = createDmlBuffer(impl->device.Get(), outputByteSize);
        opNode.outputByteSize = outputByteSize;

        DML_BINDING_PROPERTIES props = opNode.compiledOp->GetBindingProperties();
        opNode.persistentSize = props.PersistentResourceSize;
        opNode.temporarySize = props.TemporaryResourceSize;
        if (opNode.persistentSize > 0)
            opNode.persistent = createDmlBuffer(impl->device.Get(), opNode.persistentSize);
        if (opNode.temporarySize > 0)
            opNode.temporary = createDmlBuffer(impl->device.Get(), opNode.temporarySize);

        UINT descriptorCount = props.RequiredDescriptorCount > 0 ? props.RequiredDescriptorCount : 1;
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = descriptorCount;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        throwIfFailed(impl->device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&opNode.heap)),
                      "ID3D12Device::CreateDescriptorHeap");

        DML_BINDING_TABLE_DESC tableDesc{};
        tableDesc.Dispatchable = opNode.compiledOp.Get();
        tableDesc.CPUDescriptorHandle = opNode.heap->GetCPUDescriptorHandleForHeapStart();
        tableDesc.GPUDescriptorHandle = opNode.heap->GetGPUDescriptorHandleForHeapStart();
        tableDesc.SizeInDescriptors = descriptorCount;
        throwIfFailed(impl->dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&opNode.bindingTable)),
                      "IDMLDevice::CreateBindingTable");

        initializeOperator(impl, opNode);

        ComPtr<ID3D12Resource> outBuf = opNode.output;
        compiled->opNodes.push_back(std::move(opNode));
        return outBuf;
    };

    for (size_t i = 0; i < n; i++)
    {
        const Node &node = g.nodes[i];
        compiled->nodeByteSize[i] =
            dmlAlignedByteSize((UINT64)elementByteSize(node.dataType) * (UINT64)numElements(node.shape));
        DataType dt = node.dataType;
        const std::vector<int64_t> &outShape = node.shape;

        switch (node.kind)
        {
        case OpKind::Input:
        case OpKind::Reshape:
            continue; // resolved dynamically at dispatch time, see resolveBuffer()

        case OpKind::Constant:
        {
            ComPtr<ID3D12Resource> buf = createDmlBuffer(impl->device.Get(), compiled->nodeByteSize[i]);
            size_t sz = node.constantBytes.size();
            ComPtr<ID3D12Resource> upload = createBuffer(impl->device.Get(), sz, D3D12_HEAP_TYPE_UPLOAD,
                                                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            void *mapped = nullptr;
            throwIfFailed(upload->Map(0, nullptr, &mapped), "ID3D12Resource::Map");
            memcpy(mapped, node.constantBytes.data(), sz);
            upload->Unmap(0, nullptr);
            transition(impl->cmdList.Get(), buf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_DEST);
            impl->cmdList->CopyBufferRegion(buf.Get(), 0, upload.Get(), 0, sz);
            transition(impl->cmdList.Get(), buf.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            impl->pendingScratch.push_back(upload);
            compiled->nodeOutput[i] = buf;
            continue;
        }

        case OpKind::Add:
        case OpKind::Mul:
        {
            TensorDescBuilder a, b, out;
            a.setBroadcast(g.nodes[node.inputs[0]].shape, outShape, dt);
            b.setBroadcast(g.nodes[node.inputs[1]].shape, outShape, dt);
            out.setPacked(outShape, dt);
            DML_OPERATOR_DESC opDesc{};
            if (node.kind == OpKind::Add)
            {
                DML_ELEMENT_WISE_ADD_OPERATOR_DESC desc{};
                desc.ATensor = &a.tensor;
                desc.BTensor = &b.tensor;
                desc.OutputTensor = &out.tensor;
                opDesc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
                opDesc.Desc = &desc;
                compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0]), dyn(node.inputs[1])},
                                                  compiled->nodeByteSize[i]);
            }
            else
            {
                DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC desc{};
                desc.ATensor = &a.tensor;
                desc.BTensor = &b.tensor;
                desc.OutputTensor = &out.tensor;
                opDesc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
                opDesc.Desc = &desc;
                compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0]), dyn(node.inputs[1])},
                                                  compiled->nodeByteSize[i]);
            }
            continue;
        }

        case OpKind::Relu:
        {
            TensorDescBuilder x, out;
            x.setPacked(outShape, dt);
            out.setPacked(outShape, dt);
            DML_ACTIVATION_RELU_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.OutputTensor = &out.tensor;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_ACTIVATION_RELU, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            continue;
        }

        case OpKind::Sigmoid:
        {
            TensorDescBuilder x, out;
            x.setPacked(outShape, dt);
            out.setPacked(outShape, dt);
            DML_ACTIVATION_SIGMOID_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.OutputTensor = &out.tensor;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_ACTIVATION_SIGMOID, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            continue;
        }

        case OpKind::Gelu:
        {
            // Composed as 0.5*x*(1+erf(x/sqrt(2))) — matches the MPSGraph backend's
            // composition exactly (formula fidelity over the native ACTIVATION_GELU,
            // which is feature-level gated and not guaranteed to match bit-for-bit).
            // Erf and Identity's ScaleBias apply Scale*x+Bias *before* the unary
            // function, so the first two steps need no separate Mul/Add op.
            TensorDescBuilder xIn, erfOut;
            xIn.setPacked(outShape, dt);
            erfOut.setPacked(outShape, dt);
            DML_SCALE_BIAS invSqrt2{0.70710678118654752f, 0.0f};
            DML_ELEMENT_WISE_ERF_OPERATOR_DESC erfDesc{};
            erfDesc.InputTensor = &xIn.tensor;
            erfDesc.OutputTensor = &erfOut.tensor;
            erfDesc.ScaleBias = &invSqrt2;
            DML_OPERATOR_DESC erfOpDesc{DML_OPERATOR_ELEMENT_WISE_ERF, &erfDesc};
            ComPtr<ID3D12Resource> erfBuf =
                pushOp(compileOp(impl, erfOpDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);

            TensorDescBuilder onePlusErfIn, onePlusErfOut;
            onePlusErfIn.setPacked(outShape, dt);
            onePlusErfOut.setPacked(outShape, dt);
            // DML_ACTIVATION_IDENTITY has no ScaleBias field — ELEMENT_WISE_IDENTITY does.
            DML_SCALE_BIAS onePlusErfScaleBias{1.0f, 1.0f};
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC onePlusErfEwDesc{};
            onePlusErfEwDesc.InputTensor = &onePlusErfIn.tensor;
            onePlusErfEwDesc.OutputTensor = &onePlusErfOut.tensor;
            onePlusErfEwDesc.ScaleBias = &onePlusErfScaleBias;
            DML_OPERATOR_DESC onePlusErfOpDesc{DML_OPERATOR_ELEMENT_WISE_IDENTITY, &onePlusErfEwDesc};
            ComPtr<ID3D12Resource> onePlusErfBuf =
                pushOp(compileOp(impl, onePlusErfOpDesc), {fix(erfBuf, compiled->nodeByteSize[i])},
                       compiled->nodeByteSize[i]);

            TensorDescBuilder halfXIn, halfXOut;
            halfXIn.setPacked(outShape, dt);
            halfXOut.setPacked(outShape, dt);
            DML_SCALE_BIAS halfScaleBias{0.5f, 0.0f};
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC halfXDesc{};
            halfXDesc.InputTensor = &halfXIn.tensor;
            halfXDesc.OutputTensor = &halfXOut.tensor;
            halfXDesc.ScaleBias = &halfScaleBias;
            DML_OPERATOR_DESC halfXOpDesc{DML_OPERATOR_ELEMENT_WISE_IDENTITY, &halfXDesc};
            ComPtr<ID3D12Resource> halfXBuf =
                pushOp(compileOp(impl, halfXOpDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);

            TensorDescBuilder mulA, mulB, mulOut;
            mulA.setPacked(outShape, dt);
            mulB.setPacked(outShape, dt);
            mulOut.setPacked(outShape, dt);
            DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulDesc{};
            mulDesc.ATensor = &mulA.tensor;
            mulDesc.BTensor = &mulB.tensor;
            mulDesc.OutputTensor = &mulOut.tensor;
            DML_OPERATOR_DESC mulOpDesc{DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulDesc};
            compiled->nodeOutput[i] =
                pushOp(compileOp(impl, mulOpDesc),
                       {fix(halfXBuf, compiled->nodeByteSize[i]), fix(onePlusErfBuf, compiled->nodeByteSize[i])},
                       compiled->nodeByteSize[i]);
            continue;
        }

        default:
            break; // remaining op kinds handled below
        }

        switch (node.kind)
        {
        case OpKind::Softmax:
        {
            TensorDescBuilder x, out;
            x.setPacked(outShape, dt);
            out.setPacked(outShape, dt);
            UINT axis = (UINT)(node.axis < 0 ? node.axis + (int32_t)outShape.size() : node.axis);
            UINT axes[] = {axis};
            DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.OutputTensor = &out.tensor;
            desc.AxisCount = 1;
            desc.Axes = axes;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_ACTIVATION_SOFTMAX1, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::LayerNorm:
        {
            TensorDescBuilder x, scale, bias, out;
            x.setPacked(outShape, dt);
            scale.setBroadcast(g.nodes[node.inputs[1]].shape, outShape, dt);
            bias.setBroadcast(g.nodes[node.inputs[2]].shape, outShape, dt);
            out.setPacked(outShape, dt);
            UINT axis = (UINT)outShape.size() - 1;
            UINT axes[] = {axis};
            DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.ScaleTensor = &scale.tensor;
            desc.BiasTensor = &bias.tensor;
            desc.OutputTensor = &out.tensor;
            desc.AxisCount = 1;
            desc.Axes = axes;
            desc.NormalizeVariance = TRUE;
            desc.Epsilon = node.floatAttr0;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &desc};
            compiled->nodeOutput[i] =
                pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0]), dyn(node.inputs[1]), dyn(node.inputs[2])},
                       compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::BatchNorm:
        {
            int64_t C = outShape[1];
            TensorDescBuilder x, mean, variance, scale, bias, out;
            x.setPacked(outShape, dt);
            mean.setChannelBroadcast(C, outShape, dt);
            variance.setChannelBroadcast(C, outShape, dt);
            scale.setChannelBroadcast(C, outShape, dt);
            bias.setChannelBroadcast(C, outShape, dt);
            out.setPacked(outShape, dt);
            DML_BATCH_NORMALIZATION_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.MeanTensor = &mean.tensor;
            desc.VarianceTensor = &variance.tensor;
            desc.ScaleTensor = &scale.tensor;
            desc.BiasTensor = &bias.tensor;
            desc.OutputTensor = &out.tensor;
            desc.Spatial = TRUE;
            desc.Epsilon = node.floatAttr0;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_BATCH_NORMALIZATION, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc),
                                              {dyn(node.inputs[0]), dyn(node.inputs[1]), dyn(node.inputs[2]),
                                               dyn(node.inputs[3]), dyn(node.inputs[4])},
                                              compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::InstanceNorm:
        {
            int64_t C = outShape[1];
            TensorDescBuilder x, scale, bias, out;
            x.setPacked(outShape, dt);
            scale.setChannelBroadcast(C, outShape, dt);
            bias.setChannelBroadcast(C, outShape, dt);
            out.setPacked(outShape, dt);
            UINT axes[] = {2, 3}; // NCHW spatial axes
            DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.ScaleTensor = &scale.tensor;
            desc.BiasTensor = &bias.tensor;
            desc.OutputTensor = &out.tensor;
            desc.AxisCount = 2;
            desc.Axes = axes;
            desc.NormalizeVariance = TRUE;
            desc.Epsilon = node.floatAttr0;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &desc};
            compiled->nodeOutput[i] =
                pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0]), dyn(node.inputs[1]), dyn(node.inputs[2])},
                       compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::MatMul:
        case OpKind::Gemm:
        {
            TensorDescBuilder a, b, out;
            a.setPacked(g.nodes[node.inputs[0]].shape, dt);
            b.setPacked(g.nodes[node.inputs[1]].shape, dt);
            out.setPacked(outShape, dt);
            DML_GEMM_OPERATOR_DESC desc{};
            desc.ATensor = &a.tensor;
            desc.BTensor = &b.tensor;
            desc.OutputTensor = &out.tensor;
            desc.TransA = DML_MATRIX_TRANSFORM_NONE;
            desc.TransB = DML_MATRIX_TRANSFORM_NONE;
            std::vector<DmlBindingSource> inputs{dyn(node.inputs[0]), dyn(node.inputs[1])};
            TensorDescBuilder c;
            if (node.kind == OpKind::Gemm)
            {
                c.setBroadcast(g.nodes[node.inputs[2]].shape, outShape, dt);
                desc.CTensor = &c.tensor;
                desc.Alpha = node.floatAttr0;
                desc.Beta = node.floatAttr1;
                inputs.push_back(dyn(node.inputs[2]));
            }
            else
            {
                desc.CTensor = nullptr;
                desc.Alpha = 1.0f;
                desc.Beta = 0.0f;
                inputs.push_back(DmlBindingSource{false, 0, nullptr, 0, true});
            }
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_GEMM, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), inputs, compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::Transpose:
        {
            const std::vector<int64_t> &inShape = g.nodes[node.inputs[0]].shape;
            auto inStrides = rowMajorStrides(inShape);
            const auto &perm = node.intAttr0;
            std::vector<int64_t> permutedStrides(perm.size());
            for (size_t d = 0; d < perm.size(); d++)
                permutedStrides[d] = inStrides[perm[d]];

            TensorDescBuilder x, out;
            x.setStrided(outShape, permutedStrides, dt);
            out.setPacked(outShape, dt);
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.OutputTensor = &out.tensor;
            desc.ScaleBias = nullptr;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_ELEMENT_WISE_IDENTITY, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::Concat:
        {
            std::vector<TensorDescBuilder> ins(node.inputs.size());
            std::vector<DML_TENSOR_DESC> inTensorDescs(node.inputs.size());
            std::vector<DmlBindingSource> inputs;
            for (size_t k = 0; k < node.inputs.size(); k++)
            {
                ins[k].setPacked(g.nodes[node.inputs[k]].shape, dt);
                inTensorDescs[k] = ins[k].tensor;
                inputs.push_back(dyn(node.inputs[k]));
            }
            TensorDescBuilder out;
            out.setPacked(outShape, dt);
            UINT axis = (UINT)(node.axis < 0 ? node.axis + (int32_t)outShape.size() : node.axis);
            DML_JOIN_OPERATOR_DESC desc{};
            desc.InputCount = (UINT)inTensorDescs.size();
            desc.InputTensors = inTensorDescs.data();
            desc.OutputTensor = &out.tensor;
            desc.Axis = axis;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_JOIN, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), inputs, compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::Slice:
        {
            const std::vector<int64_t> &inShape = g.nodes[node.inputs[0]].shape;
            std::vector<UINT> offsets = toUintVec(node.intAttr0);
            std::vector<UINT> sizes = toUintVec(node.intAttr1);
            std::vector<INT> strides(offsets.size(), 1);

            TensorDescBuilder x, out;
            x.setPacked(inShape, dt);
            out.setPacked(outShape, dt);
            DML_SLICE1_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.OutputTensor = &out.tensor;
            desc.DimensionCount = (UINT)offsets.size();
            desc.InputWindowOffsets = offsets.data();
            desc.InputWindowSizes = sizes.data();
            desc.InputWindowStrides = strides.data();
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_SLICE1, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::Gather:
        {
            // DML_OPERATOR_GATHER's CreateOperator validates `Axis >= IndicesRank -
            // IndexDimensions` (confirmed empirically via the D3D12 debug layer —
            // the message text itself has the comparison backwards). For Axis==0
            // that's only satisfiable with IndexDimensions==IndicesRank, which
            // degenerates into GatherND (each Indices entry becomes a full
            // multi-dimensional coordinate, not the per-axis scalar this op's IR
            // models). Fix: present Input/Indices/Output to DML with the gather
            // axis permuted to the *last* dimension (via Strides only — a
            // transposed view, no data movement, same trick used to materialize
            // Transpose elsewhere in this file), so Axis becomes (rank-1) and a
            // non-degenerate IndexDimensions=1 satisfies the check.
            const std::vector<int64_t> &dataShape = g.nodes[node.inputs[0]].shape;
            UINT axis = (UINT)(node.axis < 0 ? node.axis + (int32_t)dataShape.size() : node.axis);
            size_t rank = dataShape.size();

            // Permutation moving `axis` to the end: every other dim keeps its
            // relative order, e.g. rank 4 axis 1 -> [0, 2, 3, 1].
            std::vector<size_t> perm;
            for (size_t d = 0; d < rank; d++)
                if (d != axis)
                    perm.push_back(d);
            perm.push_back(axis);

            auto permute = [&](const std::vector<int64_t> &v)
            {
                std::vector<int64_t> r(rank);
                for (size_t d = 0; d < rank; d++)
                    r[d] = v[perm[d]];
                return r;
            };

            int64_t numIndices = numElements(g.nodes[node.inputs[1]].shape);

            std::vector<int64_t> dataStridesNative = rowMajorStrides(dataShape);
            std::vector<int64_t> outStridesNative = rowMajorStrides(outShape);

            std::vector<int64_t> indicesSizes(rank), indicesStrides(rank);
            for (size_t d = 0; d < rank; d++)
            {
                indicesSizes[d] = perm[d] == axis ? numIndices : outShape[perm[d]];
                indicesStrides[d] = perm[d] == axis ? 1 : 0;
            }

            TensorDescBuilder data, indices, out;
            data.setStrided(permute(dataShape), permute(dataStridesNative), dt);
            indices.setStrided(indicesSizes, indicesStrides, g.nodes[node.inputs[1]].dataType);
            out.setStrided(permute(outShape), permute(outStridesNative), dt);
            DML_GATHER_OPERATOR_DESC desc{};
            desc.InputTensor = &data.tensor;
            desc.IndicesTensor = &indices.tensor;
            desc.OutputTensor = &out.tensor;
            desc.Axis = (UINT)(rank - 1);
            desc.IndexDimensions = 1;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_GATHER, &desc};
            compiled->nodeOutput[i] = pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0]), dyn(node.inputs[1])},
                                              compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::QuantizeLinear:
        case OpKind::DequantizeLinear:
        {
            float scaleVal = node.floatAttr0;
            DataType xDataType = g.nodes[node.inputs[0]].dataType;
            ComPtr<ID3D12Resource> scaleBuf = createDmlBuffer(impl->device.Get(), sizeof(float));
            {
                ComPtr<ID3D12Resource> upload = createBuffer(impl->device.Get(), sizeof(float), D3D12_HEAP_TYPE_UPLOAD,
                                                               D3D12_RESOURCE_FLAG_NONE,
                                                               D3D12_RESOURCE_STATE_GENERIC_READ);
                void *mapped = nullptr;
                throwIfFailed(upload->Map(0, nullptr, &mapped), "ID3D12Resource::Map");
                memcpy(mapped, &scaleVal, sizeof(float));
                upload->Unmap(0, nullptr);
                transition(impl->cmdList.Get(), scaleBuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_DEST);
                impl->cmdList->CopyBufferRegion(scaleBuf.Get(), 0, upload.Get(), 0, sizeof(float));
                transition(impl->cmdList.Get(), scaleBuf.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                impl->pendingScratch.push_back(upload);
            }
            // Zero-point is stored as a float in the IR (matches CPU's `scale*(x-zeroPoint)`
            // formula) but DML wants it in the *quantized* dtype (Int8) — cast, don't
            // reinterpret, or every dequantized value silently shifts.
            int8_t zpVal = (int8_t)std::lround(node.floatAttr1);
            // zpDesc's broadcast TensorDescBuilder below declares a DWORD-aligned
            // TotalTensorSizeInBytes (4) even though only 1 byte is logically
            // meaningful (every broadcast read is of element 0) — the bound buffer
            // must be allocated at least that large, or BindInputs silently no-ops.
            UINT64 zpBufSize = dmlAlignedByteSize(sizeof(int8_t));
            ComPtr<ID3D12Resource> zpBuf = createDmlBuffer(impl->device.Get(), zpBufSize);
            {
                ComPtr<ID3D12Resource> upload = createBuffer(impl->device.Get(), sizeof(int8_t), D3D12_HEAP_TYPE_UPLOAD,
                                                               D3D12_RESOURCE_FLAG_NONE,
                                                               D3D12_RESOURCE_STATE_GENERIC_READ);
                void *mapped = nullptr;
                throwIfFailed(upload->Map(0, nullptr, &mapped), "ID3D12Resource::Map");
                memcpy(mapped, &zpVal, sizeof(int8_t));
                upload->Unmap(0, nullptr);
                transition(impl->cmdList.Get(), zpBuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_DEST);
                impl->cmdList->CopyBufferRegion(zpBuf.Get(), 0, upload.Get(), 0, sizeof(int8_t));
                transition(impl->cmdList.Get(), zpBuf.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                impl->pendingScratch.push_back(upload);
            }

            TensorDescBuilder x, scaleDesc, zpDesc, out;
            x.setPacked(g.nodes[node.inputs[0]].shape, xDataType);
            // Per-tensor scale/zero-point: DML requires Sizes to match across all
            // of an elementwise op's tensors (broadcasting expressed via zero
            // strides, not differing literal Sizes) — same convention as
            // Add/Mul's setBroadcast above, not a plain {1}-shaped setPacked.
            scaleDesc.setBroadcast({1}, outShape, DataType::Float32);
            zpDesc.setBroadcast({1}, outShape, DataType::Int8);
            out.setPacked(outShape, dt);

            if (node.kind == OpKind::QuantizeLinear)
            {
                DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC desc{};
                desc.InputTensor = &x.tensor;
                desc.ScaleTensor = &scaleDesc.tensor;
                desc.ZeroPointTensor = &zpDesc.tensor;
                desc.OutputTensor = &out.tensor;
                DML_OPERATOR_DESC opDesc{DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR, &desc};
                compiled->nodeOutput[i] =
                    pushOp(compileOp(impl, opDesc),
                           {dyn(node.inputs[0]), fix(scaleBuf, sizeof(float)), fix(zpBuf, zpBufSize)},
                           compiled->nodeByteSize[i]);
            }
            else
            {
                DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC desc{};
                desc.InputTensor = &x.tensor;
                desc.ScaleTensor = &scaleDesc.tensor;
                desc.ZeroPointTensor = &zpDesc.tensor;
                desc.OutputTensor = &out.tensor;
                DML_OPERATOR_DESC opDesc{DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &desc};
                compiled->nodeOutput[i] =
                    pushOp(compileOp(impl, opDesc),
                           {dyn(node.inputs[0]), fix(scaleBuf, sizeof(float)), fix(zpBuf, zpBufSize)},
                           compiled->nodeByteSize[i]);
            }
            break;
        }

        case OpKind::Conv2d:
        {
            const Conv2dDescriptor &p = node.convParams;
            std::vector<UINT> strides = {(UINT)p.strideY, (UINT)p.strideX};
            std::vector<UINT> dilations = {(UINT)p.dilationY, (UINT)p.dilationX};
            std::vector<UINT> startPadding = {(UINT)p.paddingTop, (UINT)p.paddingLeft};
            std::vector<UINT> endPadding = {(UINT)p.paddingBottom, (UINT)p.paddingRight};
            std::vector<UINT> outputPadding = {0, 0};

            TensorDescBuilder x, w, out;
            x.setPacked(g.nodes[node.inputs[0]].shape, dt);
            w.setPacked(g.nodes[node.inputs[1]].shape, dt);
            out.setPacked(outShape, dt);
            DML_CONVOLUTION_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.FilterTensor = &w.tensor;
            desc.BiasTensor = nullptr; // not fused — caller adds bias separately via add()
            desc.OutputTensor = &out.tensor;
            desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
            desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
            desc.DimensionCount = 2;
            desc.Strides = strides.data();
            desc.Dilations = dilations.data();
            desc.StartPadding = startPadding.data();
            desc.EndPadding = endPadding.data();
            desc.OutputPadding = outputPadding.data();
            desc.GroupCount = (UINT)p.groups;
            desc.FusedActivation = nullptr;
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_CONVOLUTION, &desc};
            // BiasTensor is _Maybenull_, but the operator's fixed (Input, Filter,
            // Bias) binding schema still needs a slot for it — same as GEMM's
            // CTensor above. Bound as DML_BINDING_TYPE_NONE rather than omitted.
            compiled->nodeOutput[i] = pushOp(
                compileOp(impl, opDesc),
                {dyn(node.inputs[0]), dyn(node.inputs[1]), DmlBindingSource{false, 0, nullptr, 0, true}},
                compiled->nodeByteSize[i]);
            break;
        }

        case OpKind::MaxPool2d:
        case OpKind::AvgPool2d:
        {
            const Pool2dDescriptor &p = node.poolParams;
            std::vector<UINT> windowSize = {(UINT)p.kernelHeight, (UINT)p.kernelWidth};
            std::vector<UINT> strides = {(UINT)p.strideY, (UINT)p.strideX};
            std::vector<UINT> startPadding = {(UINT)p.paddingTop, (UINT)p.paddingLeft};
            std::vector<UINT> endPadding = {(UINT)p.paddingBottom, (UINT)p.paddingRight};

            TensorDescBuilder x, out;
            x.setPacked(g.nodes[node.inputs[0]].shape, dt);
            out.setPacked(outShape, dt);
            DML_OPERATOR_DESC opDesc{};
            if (node.kind == OpKind::MaxPool2d)
            {
                DML_MAX_POOLING_OPERATOR_DESC desc{};
                desc.InputTensor = &x.tensor;
                desc.OutputTensor = &out.tensor;
                desc.DimensionCount = 2;
                desc.Strides = strides.data();
                desc.WindowSize = windowSize.data();
                desc.StartPadding = startPadding.data();
                desc.EndPadding = endPadding.data();
                opDesc.Type = DML_OPERATOR_MAX_POOLING;
                opDesc.Desc = &desc;
                compiled->nodeOutput[i] =
                    pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            }
            else
            {
                DML_AVERAGE_POOLING_OPERATOR_DESC desc{};
                desc.InputTensor = &x.tensor;
                desc.OutputTensor = &out.tensor;
                desc.DimensionCount = 2;
                desc.Strides = strides.data();
                desc.WindowSize = windowSize.data();
                desc.StartPadding = startPadding.data();
                desc.EndPadding = endPadding.data();
                desc.IncludePadding = FALSE; // matches CPU: divide by non-padded count only
                opDesc.Type = DML_OPERATOR_AVERAGE_POOLING;
                opDesc.Desc = &desc;
                compiled->nodeOutput[i] =
                    pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            }
            break;
        }

        case OpKind::Resize:
        {
            const ResizeDescriptor &p = node.resizeParams;
            const std::vector<int64_t> &inShape = g.nodes[node.inputs[0]].shape;
            int64_t inH = inShape[2], inW = inShape[3];
            int64_t outH = p.outputHeight, outW = p.outputWidth;

            // Scale = outSize/inSize (DML's documented forward convention: "scales > 1
            // scale up the image"); offsets per Microsoft's own worked example
            // (InputPixelOffsets=0.5, OutputPixelOffsets=-0.5 reproduces the standard
            // half-pixel/OpenCV/TF2 resize). Verified against the public DML docs
            // rather than assumed, since getting the direction backward would
            // silently produce a wrong (but plausible-looking) resize.
            auto axisParams = [](int64_t inSize, int64_t outSize, bool alignCorners, bool centerResult)
            {
                double scale, inOff, outOff;
                if (alignCorners)
                {
                    scale = outSize > 1 ? (double)(outSize - 1) / (double)(inSize - 1) : 0.0;
                    inOff = 0.0;
                    outOff = 0.0;
                }
                else
                {
                    scale = (double)outSize / (double)inSize;
                    inOff = centerResult ? 0.5 : 0.0;
                    outOff = centerResult ? -0.5 : 0.0;
                }
                return std::make_tuple((float)scale, (float)inOff, (float)outOff);
            };
            auto [scaleH, inOffH, outOffH] = axisParams(inH, outH, p.alignCorners, p.centerResult);
            auto [scaleW, inOffW, outOffW] = axisParams(inW, outW, p.alignCorners, p.centerResult);

            // DML_AXIS_DIRECTION_INCREASING/DECREASING select pure ceil/floor for
            // nearest-neighbor, not "round to nearest with this tie-break" as the
            // name suggests (confirmed empirically: ResizeNearestAlignCorners came
            // back ceil'd). The CPU backend's !nearestRoundsDown path is
            // std::round() (ties away from zero), which for these non-negative
            // pixel coordinates equals floor(x + 0.5) — so always floor here, and
            // fold a 0.5 input-pixel offset in for the round case. InputPixelOffset
            // is *subtracted* from the scaled coordinate (confirmed empirically —
            // a `+0.5` collapsed every index to 0, a `-0.5` produces the right
            // [0,0,1,1]-style mapping), so pass -0.5 to add 0.5 to the coordinate.
            // nearestRoundsDown already matches DECREASING==floor exactly, no
            // offset needed.
            if (p.mode == ResizeMode::Nearest && !p.nearestRoundsDown)
            {
                inOffH -= 0.5f;
                inOffW -= 0.5f;
            }

            std::vector<FLOAT> scales = {1.0f, 1.0f, scaleH, scaleW};
            std::vector<FLOAT> inputOffsets = {0.0f, 0.0f, inOffH, inOffW};
            std::vector<FLOAT> outputOffsets = {0.0f, 0.0f, outOffH, outOffW};

            TensorDescBuilder x, out;
            x.setPacked(inShape, dt);
            out.setPacked(outShape, dt);
            DML_RESAMPLE2_OPERATOR_DESC desc{};
            desc.InputTensor = &x.tensor;
            desc.OutputTensor = &out.tensor;
            desc.InterpolationMode =
                p.mode == ResizeMode::Nearest ? DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR : DML_INTERPOLATION_MODE_LINEAR;
            desc.RoundingDirection = DML_AXIS_DIRECTION_DECREASING;
            desc.DimensionCount = 4;
            desc.Scales = scales.data();
            desc.InputPixelOffsets = inputOffsets.data();
            desc.OutputPixelOffsets = outputOffsets.data();
            DML_OPERATOR_DESC opDesc{DML_OPERATOR_RESAMPLE2, &desc};
            compiled->nodeOutput[i] =
                pushOp(compileOp(impl, opDesc), {dyn(node.inputs[0])}, compiled->nodeByteSize[i]);
            break;
        }

        default:
            throw std::runtime_error("campello_nn: DirectML backend: unhandled OpKind");
        }
    }

    for (auto &[name, nodeIdx] : g.outputs)
        compiled->outputNodeForName[name] = nodeIdx;

    impl->executeAndWait();
    return compiled;
}

void DirectMlBackend::destroyGraph(void *native)
{
    delete (DmlCompiledGraph *)native;
}

namespace
{
    // Resolves the concrete buffer backing IR node `idx` for this specific
    // dispatch() call. Input nodes are bound by the caller fresh every call;
    // Reshape is a zero-cost alias resolved by walking to its source (recursively,
    // in case of Reshape-of-Reshape); every other node (Constant, or any real
    // compiled op) has had a fixed buffer since compileGraph() ran.
    ID3D12Resource *resolveBuffer(const GraphIR &g, DmlCompiledGraph *compiled, size_t idx,
                                   const std::unordered_map<std::string, void *> &inputs)
    {
        const Node &node = g.nodes[idx];
        if (node.kind == OpKind::Input)
        {
            auto it = inputs.find(node.name);
            if (it == inputs.end())
                throw std::runtime_error("campello_nn: DirectML backend missing input '" + node.name + "'");
            return ((DmlTensor *)it->second)->buffer.Get();
        }
        if (node.kind == OpKind::Reshape)
            return resolveBuffer(g, compiled, node.inputs[0], inputs);
        return compiled->nodeOutput[idx].Get();
    }
} // namespace

void *DirectMlBackend::dispatch(
    void *compiledGraph,
    const std::unordered_map<std::string, void *> &inputs,
    const std::unordered_map<std::string, void *> &outputs)
{
    auto compiled = (DmlCompiledGraph *)compiledGraph;
    const GraphIR &g = compiled->ir;

    for (auto &opNode : compiled->opNodes)
    {
        std::vector<DML_BUFFER_BINDING> inputBindings(opNode.inputs.size());
        std::vector<DML_BINDING_DESC> inputDescs(opNode.inputs.size());
        for (size_t k = 0; k < opNode.inputs.size(); k++)
        {
            const DmlBindingSource &src = opNode.inputs[k];
            inputDescs[k] = {};
            if (src.none)
            {
                inputDescs[k].Type = DML_BINDING_TYPE_NONE;
                continue;
            }
            inputBindings[k] = {};
            inputBindings[k].Buffer = src.dynamic ? resolveBuffer(g, compiled, src.irNodeIndex, inputs) : src.fixed.Get();
            inputBindings[k].SizeInBytes = src.byteSize;
            inputDescs[k].Type = DML_BINDING_TYPE_BUFFER;
            inputDescs[k].Desc = &inputBindings[k];
        }
        opNode.bindingTable->BindInputs((UINT)inputDescs.size(), inputDescs.data());

        DML_BUFFER_BINDING outputBinding{};
        outputBinding.Buffer = opNode.output.Get();
        outputBinding.SizeInBytes = opNode.outputByteSize;
        DML_BINDING_DESC outputDesc{};
        outputDesc.Type = DML_BINDING_TYPE_BUFFER;
        outputDesc.Desc = &outputBinding;
        opNode.bindingTable->BindOutputs(1, &outputDesc);

        if (opNode.temporarySize > 0)
        {
            DML_BUFFER_BINDING tempBinding{};
            tempBinding.Buffer = opNode.temporary.Get();
            tempBinding.SizeInBytes = opNode.temporarySize;
            DML_BINDING_DESC tempDesc{};
            tempDesc.Type = DML_BINDING_TYPE_BUFFER;
            tempDesc.Desc = &tempBinding;
            opNode.bindingTable->BindTemporaryResource(&tempDesc);
        }
        if (opNode.persistentSize > 0)
        {
            DML_BUFFER_BINDING persistBinding{};
            persistBinding.Buffer = opNode.persistent.Get();
            persistBinding.SizeInBytes = opNode.persistentSize;
            DML_BINDING_DESC persistDesc{};
            persistDesc.Type = DML_BINDING_TYPE_BUFFER;
            persistDesc.Desc = &persistBinding;
            opNode.bindingTable->BindPersistentResource(&persistDesc);
        }

        // Same requirement as initializeOperator() above: the command list's bound
        // heap (not the binding table) is what RecordDispatch resolves descriptors
        // against, and each opNode owns its own heap.
        ID3D12DescriptorHeap *opHeaps[] = {opNode.heap.Get()};
        impl->cmdList->SetDescriptorHeaps(1, opHeaps);
        impl->recorder->RecordDispatch(impl->cmdList.Get(), opNode.compiledOp.Get(), opNode.bindingTable.Get());
        uavBarrier(impl->cmdList.Get());
    }

    // Every requested graph output gets copied from its (fixed, dedicated) source
    // buffer into the caller-provided output Tensor's buffer — these are always
    // different ID3D12Resources, since node buffers are allocated once at compile
    // time but the caller's Tensor is only known per dispatch() call.
    for (auto &[name, nativePtr] : outputs)
    {
        auto it = compiled->outputNodeForName.find(name);
        if (it == compiled->outputNodeForName.end())
            throw std::runtime_error("campello_nn: DirectML backend has no graph output named '" + name + "'");
        ID3D12Resource *src = resolveBuffer(g, compiled, it->second, inputs);
        auto dst = (DmlTensor *)nativePtr;

        transition(impl->cmdList.Get(), src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(impl->cmdList.Get(), dst->buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        impl->cmdList->CopyBufferRegion(dst->buffer.Get(), 0, src, 0, dst->byteSize);
        transition(impl->cmdList.Get(), dst->buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(impl->cmdList.Get(), src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    impl->executeAndWait();
    return new DmlFence{true};
}

bool DirectMlBackend::waitFence(void *fenceNative, uint64_t)
{
    return ((DmlFence *)fenceNative)->signaled;
}

bool DirectMlBackend::isFenceSignaled(void *fenceNative)
{
    return ((DmlFence *)fenceNative)->signaled;
}

void DirectMlBackend::destroyFence(void *fenceNative)
{
    delete (DmlFence *)fenceNative;
}
