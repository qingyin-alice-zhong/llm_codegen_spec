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
from allo.backend.experimental.external_kernel import ExternalModule

Ly = Layout("R")

# Matrix dimensions matching the kernel
DIM_M = 32
DIM_K = 32

# PyTorch reference code starts
def matvec_multiply(A: Annotated[torch.Tensor, "shape: (32, 32)"], b: Annotated[torch.Tensor, "shape: (32,)"]) -> Annotated[torch.Tensor, "shape: (32,)"]:
    """
    A: matrix tensor of shape (M, K)
    b: vector tensor of shape (K,)
    Returns: vector tensor of shape (M,)
    """
    return torch.matmul(A, b)
# PyTorch reference code ends


def _test_matvec_multiply(kernel_path: str):
    
    # Perform matrix-vector multiplication
    matvec_kernel = ExternalModule(
        top="matvec_vectorized_i16_i32",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty_in = int16
    Ty_out = int32
    M = DIM_M
    K = DIM_K

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def matvec_core(A: Ty_in[M * K] @ Ly, B: Ty_in[K] @ Ly, C: Ty_out[M] @ Ly):
            matvec_kernel(A, B, C)

    # Generate test data - matrix as 2D for reference computation
    input_matrix_A_2d = torch.randint(-10, 10, (DIM_M, DIM_K), dtype=torch.int16)
    input_vector_b = torch.randint(-10, 10, (DIM_K,), dtype=torch.int16)
    
    # Flatten matrix for kernel input (row-major order)
    input_matrix_A_flat = input_matrix_A_2d.flatten()
    
    # Compute reference output using 2D matrix
    reference_output = matvec_multiply(input_matrix_A_2d.to(torch.int32), input_vector_b.to(torch.int32))

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie-mlir",
                       profile=True,
                       warmup=2000,
                       num_iters=10000,
                       )
        
        # Prepare output buffer
        output_allo = np.zeros((DIM_M,)).astype(np.int32)
        
        # Run the computation with flattened matrix
        mod(input_matrix_A_flat.cpu().numpy(), input_vector_b.cpu().numpy(), output_allo)
        
        # Compare results
        try:
            np.testing.assert_allclose(output_allo, reference_output.cpu().numpy(), rtol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="kernels/mv.cc")
    args = parser.parse_args()

    _test_matvec_multiply(args.kernel_path) 