#include <campello_nn/graph_cache.hpp>
#include <campello_nn/graph_builder.hpp>
#include <fstream>
#include <stdexcept>
#include "ir.hpp"
#include "ir_serialization.hpp"

using namespace systems::leal::campello_nn;

// Deserializes the IR twice — once here (for the input/output descriptor maps)
// and once inside GraphBuilder::deserialize() (for compilation). Cache loading
// isn't a hot path, so the simpler two-pass version is preferred over plumbing
// the already-deserialized GraphIR through a public-header-safe API.
GraphCacheResult systems::leal::campello_nn::loadGraphFromMemory(std::shared_ptr<Context> context, const uint8_t *data, size_t size)
{
    GraphIR ir = deserializeGraphIR(data, size);

    GraphCacheResult result;
    for (const Node &node : ir.nodes)
        if (node.kind == OpKind::Input)
            result.inputs[node.name] = TensorDescriptor{node.dataType, node.shape, false, true};
    for (const auto &[name, nodeIndex] : ir.outputs)
    {
        const Node &node = ir.nodes[nodeIndex];
        result.outputs[name] = TensorDescriptor{node.dataType, node.shape, true, false};
    }

    result.graph = GraphBuilder::deserialize(context, data, size);
    return result;
}

GraphCacheResult systems::leal::campello_nn::loadGraphFromFile(std::shared_ptr<Context> context, const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("campello_nn: loadGraphFromFile() cannot open '" + path + "'");
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return loadGraphFromMemory(context, bytes.data(), bytes.size());
}

void systems::leal::campello_nn::saveGraphToFile(const std::vector<uint8_t> &serializedGraph, const std::string &path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("campello_nn: saveGraphToFile() cannot open '" + path + "' for writing");
    f.write((const char *)serializedGraph.data(), (std::streamsize)serializedGraph.size());
}
