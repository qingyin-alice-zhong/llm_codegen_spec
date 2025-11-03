import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import uint8
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 1024

# Reference code starts
def reference_bitwiseor_uint8(x0: Annotated[np.ndarray, "shape: (1024,)"], x1: Annotated[np.ndarray, "shape: (1024,)"] -> Annotated[np.ndarray, "shape: (1024,)"]):
    return np.bitwise_or(x0, x1).astype(np.uint8)
# Reference code ends


def _test_bitwiseor_uint8(kernel_path: str):
    bitwiseor_uint8_kernel = ExternalModule(
        top="bitwiseor_uint8",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = uint8
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: uint8[M] @ Ly, B: uint8[M] @ Ly, C: uint8[M] @ Ly):
            bitwiseor_uint8_kernel(A, B, C)

    input_tensor1 = np.random.randint(-100, 100, (1024,), dtype=np.uint8)
    input_tensor2 = np.random.randint(-100, 100, (1024,), dtype=np.uint8)

    ref_output = reference_bitwiseor_uint8(input_tensor1, input_tensor2)

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
        output_allo = np.zeros((tensor_size,), dtype=np.uint8)
        mod(input_tensor1, input_tensor2, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="bitwiseor_uint8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_bitwiseor_uint8(args.kernel_path)
