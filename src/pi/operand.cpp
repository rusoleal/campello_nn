#include <campello_nn/operand.hpp>

using namespace systems::leal::campello_nn;

Operand::Operand() : builder(nullptr), nodeId(0) {}

Operand::Operand(void *builder, size_t nodeId) : builder(builder), nodeId(nodeId) {}
