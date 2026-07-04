#!/usr/bin/env python3
"""
Mechanical Float16 porting helper for src/gpu/shaders/*.

This script performs the dtype-agnostic replacements needed to make every
shader body compile as both FP32 and FP16 via the preamble-defined T/T2/T4
aliases. It does NOT handle algorithmic changes (e.g. accumulation precision);
those are reviewed and fixed manually after running this script.

Run from the project root:
    python3 scripts/port_shaders_fp16.py
"""

import re
from pathlib import Path

SHADERS_DIR = Path("src/gpu/shaders")

OPS = [
    "relu", "add", "mul", "sigmoid", "gelu",
    "matmul", "gemm",
    "layernorm", "rmsnorm", "softmax",
    "transpose", "slice", "concat", "gather",
    "conv2d", "im2col", "conv_gemm", "conv_fused", "conv_fused_bn",
    "pool2d", "resize", "batchnorm", "instancenorm",
    "quantize_linear", "dequantize_linear", "broadcast_binary",
]


def port_glsl(text: str) -> str:
    # GLSL buffer declarations: { float data[]; } -> { T data[]; }
    text = re.sub(r"\{\s*float\s+data\[\]\s*;\s*\}", "{ T data[]; }", text)
    # Also handle vector buffer declarations if any exist.
    text = re.sub(r"\{\s*vec2\s+data\[\]\s*;\s*\}", "{ T2 data[]; }", text)
    text = re.sub(r"\{\s*vec4\s+data\[\]\s*;\s*\}", "{ T4 data[]; }", text)
    return text


def port_metal(text: str) -> str:
    # Metal pointer declarations: (const) device float *name -> (const) device T *name
    # Be careful not to touch float2/float4.
    text = re.sub(
        r"(\b(?:const\s+)?device\s+)float(\s*\*\s*[a-zA-Z_][a-zA-Z0-9_]*)",
        r"\1T\2",
        text,
    )
    return text


def port_hlsl(text: str) -> str:
    # HLSL structured buffers.
    text = re.sub(r"StructuredBuffer<float>", "StructuredBuffer<T>", text)
    text = re.sub(r"RWStructuredBuffer<float>", "RWStructuredBuffer<T>", text)
    return text


def port_file(path: Path) -> None:
    text = path.read_text()
    original = text
    if path.suffix == ".comp":
        text = port_glsl(text)
    elif path.suffix == ".metal":
        text = port_metal(text)
    elif path.suffix == ".hlsl":
        text = port_hlsl(text)
    else:
        return
    if text != original:
        path.write_text(text)
        print(f"ported {path}")
    else:
        print(f"no changes {path}")


def main() -> None:
    for op in OPS:
        for ext in (".comp", ".metal", ".hlsl"):
            path = SHADERS_DIR / f"{op}{ext}"
            if path.exists():
                port_file(path)
            else:
                print(f"missing {path}")


if __name__ == "__main__":
    main()
