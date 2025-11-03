# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import argparse
from typing import Annotated
from allo.ir.types import float32, int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

Ly = Layout("R")

tensor_size = 1024

# PyTorch reference code starts
def vector_scalar_mul(x: Annotated[torch.Tensor, "shape: (1024,)"], factor: Annotated[torch.Tensor, "shape: (1,)"]) -> Annotated[torch.Tensor, "shape: (1024,)"]:
    """
    x: input vector tensor
    factor: scalar factor to multiply with
    """
    return x * factor
# PyTorch reference code ends


def _test_vector_scalar_mul(kernel_path: str):
    
    scale = ExternalModule(
        top="vector_scalar_mul_int16_vector",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty_vec = int16
    Ty_scalar = int32
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty_vec[M] @ Ly,  B: Ty_scalar[1], C: Ty_vec[M] @ Ly):
            scale(A, B, C)

    input_tensor = torch.randint(-100, 100, (tensor_size,), dtype=torch.int16)
    factor_tensor = torch.tensor([3], dtype=torch.int32)
    output = vector_scalar_mul(input_tensor, factor_tensor).to(torch.int16)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie",
                       profile=True,
                       warmup=2000,
                       num_iters=10000,
                       )
        output_allo = np.zeros((tensor_size,)).astype(np.int16)
        mod(input_tensor.cpu().numpy(), factor_tensor.cpu().numpy(), output_allo)
        
        # Gracefully handle verification
        try:
            np.testing.assert_allclose(output_allo, output, rtol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    # add an argument to specify the `impl_path` of the ExternalModule
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="kernels/scale.cc")
    args = parser.parse_args()

    _test_vector_scalar_mul(args.kernel_path)