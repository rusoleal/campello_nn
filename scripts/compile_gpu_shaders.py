#!/usr/bin/env python3
"""
Compile campello_nn GpuGeneric shaders and emit embedded C++ byte-array headers.

Requires:
  - glslangValidator in PATH (for SPIR-V)
  - xcrun metal/metallib (for Metal .metallib)

Run from the project root:
    python3 scripts/compile_gpu_shaders.py
"""

import os
import subprocess
import sys
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
    # Remove trailing comma on last row for cleanliness, or keep it (both valid).
    # Keep it consistent with existing generated headers, which end the array row with a comma.
    lines.append(f"}};")
    lines.append(f"inline constexpr unsigned long {var_name}_bytes_len = {len(data)};")
    lines.append("// clang-format on")
    return "\n".join(lines)


def compile_spv(op: str) -> None:
    src = SHADERS_DIR / f"{op}.comp"
    spv = SHADERS_DIR / f"{op}.spv"
    header = SHADERS_DIR / f"{op}_spv.hpp"

    if not src.exists():
        print(f"skip {op}: {src} not found")
        return

    print(f"compiling {src} -> {spv}")
    subprocess.run(
        ["glslangValidator", "-V", "-S", "comp", str(src), "-o", str(spv)],
        check=True,
    )

    data = spv.read_bytes()
    header.write_text(f"#pragma once\n\n{bytes_to_c_array(data, f'{op}_spv', f'{op}.spv')}\n")
    print(f"wrote {header} ({len(data)} bytes)")


def compile_metallib(op: str) -> None:
    src = SHADERS_DIR / f"{op}.metal"
    air = SHADERS_DIR / f"{op}.air"
    metallib = SHADERS_DIR / f"{op}.metallib"
    header = SHADERS_DIR / f"{op}_metallib.hpp"

    if not src.exists():
        print(f"skip {op}: {src} not found")
        return

    print(f"compiling {src} -> {metallib}")
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(src), "-o", str(air)],
        check=True,
    )
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(metallib)],
        check=True,
    )

    data = metallib.read_bytes()
    header.write_text(f"#pragma once\n\n{bytes_to_c_array(data, f'{op}_metallib', f'{op}.metallib')}\n")
    print(f"wrote {header} ({len(data)} bytes)")

    air.unlink(missing_ok=True)


def main(argv: list[str]) -> None:
    check_tool("glslangValidator")
    check_tool("xcrun")

    if not SHADERS_DIR.is_dir():
        print(f"error: {SHADERS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    ops = argv[1:] if len(argv) > 1 else OPS
    for op in ops:
        compile_spv(op)
        compile_metallib(op)

    print("done")


if __name__ == "__main__":
    main(sys.argv)
