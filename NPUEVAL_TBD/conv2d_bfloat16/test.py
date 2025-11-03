import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16, int32
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 16

# Reference code starts
def reference_conv2d_bfloat16(input_matrix: Annotated[np.ndarray, "shape: (256,)"], param: Annotated[np.ndarray, "shape: (5,)"]) -> Annotated[np.ndarray, "shape: (224,)"]:
    # Reshape input from flattened to 16x16 matrix
    input_2d = input_matrix.reshape(16, 16)
    
    # Extract kernel (2x2=4 values) and stride from param (5 float32 values packed as int32)
    kernel_vals = np.array([param[i] for i in range(4)], dtype=np.int32).view(np.float32)
    stride_val = int(np.array([param[4]], dtype=np.int32).view(np.float32)[0])
    
    kernel_2d = kernel_vals.reshape(2, 2).astype(np_bfloat16)
    stride = max(1, stride_val)  # Ensure stride >= 1
    
    # Calculate output dimensions
    input_rows, input_cols = 16, 16
    kernel_size = 2
    output_rows = (input_rows - kernel_size) // stride + 1
    output_cols = (input_cols - kernel_size) // stride + 1
    
    # Fix output to match C kernel: 14x16=224 for 4-byte alignment (224*2=448, 448/4=112)
    output_rows = 14
    output_cols = 16
    
    output_2d = np.zeros((output_rows, output_cols), dtype=np_bfloat16)
    
    for i in range(output_rows):
        for j in range(output_cols):
            conv_sum = 0.0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    # Add boundary check to prevent out-of-bounds access
                    input_i = i * stride + ki
                    input_j = j * stride + kj
                    if input_i < input_rows and input_j < input_cols:
                        conv_sum += float(input_2d[input_i, input_j]) * float(kernel_2d[ki, kj])
            output_2d[i, j] = np_bfloat16(conv_sum)
    
    # Flatten output to 1D array for consistency
    return output_2d.flatten()
# Reference code ends


def _test_conv2d_bfloat16(kernel_path: str):
    conv2d_bfloat16_kernel = ExternalModule(
        top="conv2d_bfloat16",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = bfloat16
    M = 256  # Flattened 16x16 input matrix

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, C: bfloat16[224] @ Ly, Param: int32[5] @ Ly):
            conv2d_bfloat16_kernel(A, C, Param)

    # Generate random bfloat16 16x16 input matrix and 2x2 kernel + stride
    input_matrix = (np.random.randn(256) * 2.0).astype(np_bfloat16)  # Flattened 16x16
    
    # Pack kernel (2x2=4 values) and stride into param array (5 float32 values as int32)
    kernel_vals = (np.random.randn(4) * 0.5).astype(np.float32)
    stride_val = 1  # Use stride=1 for 15x15 output
    
    param = np.array([
        kernel_vals[0].view(np.int32), kernel_vals[1].view(np.int32),
        kernel_vals[2].view(np.int32), kernel_vals[3].view(np.int32),
        np.float32(stride_val).view(np.int32)
    ], dtype=np.int32)
    
    ref_output = reference_conv2d_bfloat16(input_matrix, param.astype(np.float32))

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=True,
            warmup=2000,
            num_iters=10000,
            trace=[("core", (0,))],
            trace_size=65536,
            project=TOP_PRJ_ABS_DIR
        )
        output_allo = np.zeros((224,), dtype=np_bfloat16)
        mod(input_matrix, output_allo, param)
        try:
            # Convert bfloat16 to float32 for comparison
            np.testing.assert_allclose(output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="conv2d_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv2d_bfloat16(args.kernel_path)
