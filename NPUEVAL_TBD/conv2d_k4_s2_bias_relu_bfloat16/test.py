import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16, int32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 1

# Reference code starts
def reference_conv2d_k4_s2_bias_relu_bfloat16(x0: Annotated[np.ndarray, "shape: (1,)"], x1: Annotated[np.ndarray, "shape: (1,)"], param: Annotated[np.ndarray, "shape: (1,)"], param: Annotated[np.ndarray, "shape: (1,)"], param1: Annotated[np.ndarray, "shape: (1,)"] -> Annotated[np.ndarray, "shape: (-3,)"]):
    return out.astype(np.float32)
# Reference code ends


def _test_conv2d_k4_s2_bias_relu_bfloat16(kernel_path: str):
    conv2d_k4_s2_bias_relu_bfloat16_kernel = ExternalModule(
        top="conv2d_k4_s2_bias_relu_bfloat16",
        impl_path=kernel_path,
        input_idx=[0, 1, 3, 4],
        output_idx=[2],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: bfloat16[M] @ Ly, C: bfloat16[-3] @ Ly, Param0: int32[1], Param1: int32[1]):
            conv2d_k4_s2_bias_relu_bfloat16_kernel(A, B, C, Param0, Param1)

    input_tensor1 = np.random.randint(-100, 100, (1,), dtype=np.int8)
    input_tensor2 = np.random.randint(-100, 100, (16,), dtype=np.int8)
    input_tensor3 = np.random.randint(-100, 100, (16,), dtype=np.int8)
    input_tensor4 = np.random.randint(-100, 100, (1,), dtype=np.int8)
    param0 = np.array([2], dtype=np.int32)
    param1 = np.array([2], dtype=np.int32)
    ref_output = reference_conv2d_k4_s2_bias_relu_bfloat16(input_tensor1, input_tensor2, input_tensor3, input_tensor4, param0, param1)

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
        output_allo = np.zeros((-3,), dtype=np.float32)
        mod(input_tensor1, input_tensor2, input_tensor3, input_tensor4, output_allo, param0, param1)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="conv2d_k4_s2_bias_relu_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv2d_k4_s2_bias_relu_bfloat16(args.kernel_path)
