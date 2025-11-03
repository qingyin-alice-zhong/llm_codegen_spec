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
tensor_size = 256

# Reference code starts
def reference_dotproduct_bfloat16(x0: Annotated[np.ndarray, "shape: (256,)"], x1: Annotated[np.ndarray, "shape: (256,)"], x2: Annotated[np.ndarray, "shape: (256,)"], offset: Annotated[np.ndarray, "shape: (1,)"] -> Annotated[np.ndarray, "shape: (256,)"]):
    return np.array([np.dot(x0[:x2], x1[:x2])], dtype=bfloat16).astype(np.float32)
# Reference code ends


def _test_dotproduct_bfloat16(kernel_path: str):
    dotproduct_bfloat16_kernel = ExternalModule(
        top="dotproduct_bfloat16",
        impl_path=kernel_path,
        input_idx=[0, 1, 2, 3],
        output_idx=[],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: bfloat16[M] @ Ly, C: bfloat16[M] @ Ly, Off: int32[1]):
            dotproduct_bfloat16_kernel(A, B, C, Off)



    ref_output = reference_dotproduct_bfloat16()

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
        output_allo = np.zeros((tensor_size,), dtype=np.float32)
        mod(output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="dotproduct_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_dotproduct_bfloat16(args.kernel_path)
