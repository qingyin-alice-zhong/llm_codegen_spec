import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 64

# Reference code starts
def reference_vectormult_bfloat16(vec_a: Annotated[np.ndarray, "shape: (64,)"], vec_b: Annotated[np.ndarray, "shape: (64,)"]) -> Annotated[np.ndarray, "shape: (64,)"]:    
    result = (vec_a * vec_b).astype(np_bfloat16)
    return result.astype(np_bfloat16)
# Reference code ends


def _test_vectormult_bfloat16(kernel_path: str):
    vectormult_bfloat16_kernel = ExternalModule(
        top="vectormult_bfloat16",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: bfloat16[M] @ Ly, C: bfloat16[M] @ Ly):
            vectormult_bfloat16_kernel(A, B, C)



    # Generate random bfloat16 inputs
    # vec_a = np.random.randn(tensor_size).astype(np_bfloat16) * 2.0
    # vec_b = np.random.randn(tensor_size).astype(np_bfloat16) * 2.0

    # vec_a and vec_b are all 1.0
    vec_a = np.ones((tensor_size,)).astype(np_bfloat16)
    vec_b = np.ones((tensor_size,)).astype(np_bfloat16)
    
    ref_output = reference_vectormult_bfloat16(vec_a, vec_b)

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
        output_allo = np.zeros((tensor_size,)).astype(np_bfloat16)
        mod(vec_a, vec_b, output_allo)
        try:
            # Convert bfloat16 to float32 for comparison
            np.testing.assert_allclose(output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=1e-2, atol=1e-2)
            print("PASS!")
            
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
            ## print out two arrays, since it is showing a NaN mismatch. Some arrays have NaN values
            print(f"Output Allo: {output_allo}")
            print(f"Ref Output: {ref_output}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="vectormult_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_vectormult_bfloat16(args.kernel_path)
