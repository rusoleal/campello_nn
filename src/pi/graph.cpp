#include <campello_nn/graph.hpp>
#include "resource_data.hpp"

using namespace systems::leal::campello_nn;

Graph::Graph(void *pd) : native(pd) {}

Graph::~Graph()
{
    auto data = (GraphData *)native;
    data->backend->destroyGraph(data->native);
    delete data;
}
