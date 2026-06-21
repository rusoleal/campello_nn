"""Regenerates tests/fixtures/conv_add_relu.onnx.

A real, valid ONNX file (validated with onnx.checker.check_model) covering
Conv -> Add -> Relu, used by tests/universal/test_onnx_importer.cpp. Kept here
so the fixture can be regenerated or extended without hand-editing protobuf bytes.

    python3 -m venv venv && source venv/bin/activate
    pip install onnx numpy
    python3 generate_conv_add_relu_onnx.py
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 3, 3])

# Diagonal-pick kernel, matching the CpuOps.Conv2d/MpsOps.Conv2d hand-computed tests.
w_data = np.array([1, 0, 0, 1], dtype=np.float32)
w_init = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 2, 2], w_data.tobytes(), raw=True)

# Exact-shape bias (matches conv_out [1,1,3,3]) -- campello_nn's add() requires an
# exact shape match, no NumPy-style broadcasting yet (see onnx_importer.cpp).
bias_data = np.full((1, 1, 3, 3), 100, dtype=np.float32)
bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [1, 1, 3, 3], bias_data.tobytes(), raw=True)

conv_node = helper.make_node(
    "Conv", inputs=["x", "w"], outputs=["conv_out"],
    strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1,
)
add_node = helper.make_node("Add", inputs=["conv_out", "bias"], outputs=["add_out"])
relu_node = helper.make_node("Relu", inputs=["add_out"], outputs=["out"])

graph = helper.make_graph(
    [conv_node, add_node, relu_node],
    "test_graph",
    [x],
    [out],
    initializer=[w_init, bias_init],
)

model = helper.make_model(graph, producer_name="campello_nn_test", opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "conv_add_relu.onnx")
onnx.save(model, out_path)
print("saved", out_path, "size =", os.path.getsize(out_path))
