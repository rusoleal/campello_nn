#!/usr/bin/env python3
"""
Compile campello_nn GpuGeneric shaders and emit embedded C++ byte-array headers.

Each shader source in src/gpu/shaders/<op>.{comp,metal,hlsl} is a "body" file:
it does not contain #version / #include <metal_stdlib> / etc. Instead this script
injects a dtype-specific preamble and compiles the result twice:
  * once for Float32  -> <op>_spv.hpp / <op>_metallib.hpp
  * once for Float16 -> <op>_f16_spv.hpp / <op>_f16_metallib.hpp

Requires:
  - glslangValidator in PATH (for SPIR-V)
  - xcrun metal/metallib (for Metal .metallib)

Run from the project root:
    python3 scripts/compile_gpu_shaders.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SHADERS_DIR = Path("src/gpu/shaders")

OPS = [
    "relu",
    "add",
    "matmul",
    "mul",
    "sigmoid",
    "gelu",
    "layernorm",
    "rmsnorm",
    "transpose",
    "slice",
    "concat",
    "gather",
    "gemm",
    "softmax",
    "conv2d",
    "im2col",
    "conv_gemm",
    "conv_fused",
    "conv_fused_bn",
    "pool2d",
    "resize",
    "batchnorm",
    "instancenorm",
    "quantize_linear",
    "dequantize_linear",
    "broadcast_binary",
]


GLSL_PREAMBLE_FP32 = """#version 450
#extension GL_GOOGLE_include_directive : require
#define T float
#define T2 vec2
#define T4 vec4
#include "dtype_common.glsl.inc"
"""

GLSL_PREAMBLE_FP16 = """#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define T float16_t
#define T2 f16vec2
#define T4 f16vec4
#include "dtype_common.glsl.inc"
"""

METAL_PREAMBLE_FP32 = """#include <metal_stdlib>
using namespace metal;
#define T float
#define T2 float2
#define T4 float4
#include "dtype_common.metal.inc"
"""

METAL_PREAMBLE_FP16 = """#include <metal_stdlib>
using namespace metal;
#define T half
#define T2 half2
#define T4 half4
#include "dtype_common.metal.inc"
"""

HLSL_PREAMBLE_FP32 = """#define T float
#define T2 float2
#define T4 float4
#include "dtype_common.hlsl.inc"
"""

HLSL_PREAMBLE_FP16 = """#define T half
#define T2 half2
#define T4 half4
#include "dtype_common.hlsl.inc"
"""


def check_tool(name: str) -> None:
    try:
        subprocess.run([name, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print(f"error: {name} not found in PATH", file=sys.stderr)
        sys.exit(1)


def bytes_to_c_array(data: bytes, var_name: str, source_name: str) -> str:
    lines = [f"// Generated from {source_name} — see TODO.md for regeneration commands.",
             "// clang-format off",
             f"inline constexpr unsigned char {var_name}_bytes[] = {{"]
    row = []
    for i, b in enumerate(data):
        row.append(f"0x{b:02x}")
        if len(row) == 16:
            lines.append("    " + ",".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ",".join(row) + ",")
    lines.append(f"}};")
    lines.append(f"inline constexpr unsigned long {var_name}_bytes_len = {len(data)};")
    lines.append("// clang-format on")
    return "\n".join(lines)


def assemble_source(body_path: Path, preamble: str) -> str:
    return preamble + "\n" + body_path.read_text()


def compile_spv(op: str, tmp_dir: Path) -> None:
    src = SHADERS_DIR / f"{op}.comp"
    if not src.exists():
        print(f"skip {op}: {src} not found")
        return

    for suffix, preamble, var_suffix in [
        ("", GLSL_PREAMBLE_FP32, "_spv"),
        ("_f16", GLSL_PREAMBLE_FP16, "_f16_spv"),
    ]:
        tmp_src = tmp_dir / f"{op}{suffix}.comp"
        spv = tmp_dir / f"{op}{suffix}.spv"
        header = SHADERS_DIR / f"{op}{var_suffix}.hpp"

        tmp_src.write_text(assemble_source(src, preamble))
        print(f"compiling {tmp_src} -> {spv}")
        subprocess.run(
            ["glslangValidator", "-V", "-S", "comp", f"-I{SHADERS_DIR}",
             str(tmp_src), "-o", str(spv)],
            check=True,
        )

        data = spv.read_bytes()
        header.write_text(f"#pragma once\n\n{bytes_to_c_array(data, f'{op}{var_suffix}', f'{op}{suffix}.spv')}\n")
        print(f"wrote {header} ({len(data)} bytes)")


def compile_metallib(op: str, tmp_dir: Path) -> None:
    src = SHADERS_DIR / f"{op}.metal"
    if not src.exists():
        print(f"skip {op}: {src} not found")
        return

    for suffix, preamble, var_suffix in [
        ("", METAL_PREAMBLE_FP32, "_metallib"),
        ("_f16", METAL_PREAMBLE_FP16, "_f16_metallib"),
    ]:
        tmp_src = tmp_dir / f"{op}{suffix}.metal"
        air = tmp_dir / f"{op}{suffix}.air"
        metallib = tmp_dir / f"{op}{suffix}.metallib"
        header = SHADERS_DIR / f"{op}{var_suffix}.hpp"

        tmp_src.write_text(assemble_source(src, preamble))
        print(f"compiling {tmp_src} -> {metallib}")
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", f"-I{SHADERS_DIR}",
             str(tmp_src), "-o", str(air)],
            check=True,
        )
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(metallib)],
            check=True,
        )

        data = metallib.read_bytes()
        header.write_text(f"#pragma once\n\n{bytes_to_c_array(data, f'{op}{var_suffix}', f'{op}{suffix}.metallib')}\n")
        print(f"wrote {header} ({len(data)} bytes)")


def compile_hlsl(op: str, tmp_dir: Path) -> None:
    # HLSL/DXIL compilation is not yet wired; keep the source bodies in sync but
    # do not emit headers until a DXIL toolchain is available (see TODO.md).
    src = SHADERS_DIR / f"{op}.hlsl"
    if not src.exists():
        return
    # Validate that the body can at least be assembled with the preamble.
    for suffix, preamble in [("", HLSL_PREAMBLE_FP32), ("_f16", HLSL_PREAMBLE_FP16)]:
        tmp_src = tmp_dir / f"{op}{suffix}.hlsl"
        tmp_src.write_text(assemble_source(src, preamble))
    print(f"hlsl {op}: body assembled (DXIL generation deferred)")


def main(argv: list[str]) -> None:
    check_tool("glslangValidator")
    check_tool("xcrun")

    if not SHADERS_DIR.is_dir():
        print(f"error: {SHADERS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    ops = argv[1:] if len(argv) > 1 else OPS
    with tempfile.TemporaryDirectory(prefix="campello_nn_shaders_") as tmp:
        tmp_dir = Path(tmp)
        for op in ops:
            compile_spv(op, tmp_dir)
            compile_metallib(op, tmp_dir)
            compile_hlsl(op, tmp_dir)

    print("done")


if __name__ == "__main__":
    main(sys.argv)
