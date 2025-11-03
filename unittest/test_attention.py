# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import argparse
import math
from typing import Annotated
from allo.ir.types import float32, int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

Ly = Layout("R")
LyMat = Layout("RR")

matrix_size = 64

# PyTorch reference code starts
def scaled_dot_product_attention(q: Annotated[torch.Tensor, "shape: (64, 64)"], 
                                 k: Annotated[torch.Tensor, "shape: (64, 64)"], 
                                 v: Annotated[torch.Tensor, "shape: (64, 64)"],
                                 scale: Annotated[torch.Tensor, "shape: (1,)"]) -> Annotated[torch.Tensor, "shape: (64, 64)"]:
    """
    q: query tensor
    k: key tensor
    v: value tensor
    scale: scale factor
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output
# PyTorch reference code ends



# THIS IS NOT ALLOWED IN AIE: AIE only allow 1 inputs or 2 inputs. Here we have 4 inputs. 
def _test_attention(kernel_path: str):
    
    attention_kernel = ExternalModule(
        top="attention_float32_matrix",
        impl_path=kernel_path,
        input_idx=[0, 1, 2, 3],
        output_idx=[4],
    )

    Ty = float32
    M = matrix_size
    N = matrix_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(Q: Ty[M, N] @ LyMat, K: Ty[M, N] @ LyMat, V: Ty[M, N] @ LyMat, 
                 scale: Ty[1] @ Ly, C: Ty[M, N] @ LyMat):
            attention_kernel(Q, K, V, scale, C)

    # Generate input tensors using float32
    q_tensor = torch.randn(matrix_size, matrix_size, dtype=torch.float32) * 0.1
    k_tensor = torch.randn(matrix_size, matrix_size, dtype=torch.float32) * 0.1
    v_tensor = torch.randn(matrix_size, matrix_size, dtype=torch.float32) * 0.1
    scale_tensor = torch.tensor([1.0 / math.sqrt(matrix_size)], dtype=torch.float32)
    
    # Compute reference output using float32
    output = scaled_dot_product_attention(q_tensor, k_tensor, v_tensor, scale_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie-mlir",
                       profile=True,
                       warmup=100,  # Reduced due to larger matrices
                       num_iters=1000,
                       )
        output_allo = np.zeros((matrix_size, matrix_size)).astype(np.float32)
        mod(q_tensor.cpu().numpy(), k_tensor.cpu().numpy(), v_tensor.cpu().numpy(), 
            scale_tensor.cpu().numpy(), output_allo)
        try:
            # Use appropriate tolerance for float32 computation
            np.testing.assert_allclose(output_allo, output, rtol=1e-5, atol=1e-6)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="kernels_func/attention.cc")
    args = parser.parse_args()

    _test_attention(args.kernel_path)