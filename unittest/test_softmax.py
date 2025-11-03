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

tensor_size = 1024

# PyTorch reference code starts
def softmax(x: Annotated[torch.Tensor, "shape: (1024,)"]) -> Annotated[torch.Tensor, "shape: (1024,)"]:
    """
    x: input vector tensor
    """
    return torch.softmax(x, dim=0)
# PyTorch reference code ends

def _test_softmax(kernel_path: str):
    
    softmax_kernel = ExternalModule(
        top="softmax_int16_vector",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = int16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M] @ Ly, C: Ty[M] @ Ly):
            softmax_kernel(A, C)

    # Generate input tensor with values that won't cause overflow when converted to int16
    input_tensor = torch.randint(-50, 50, (tensor_size,), dtype=torch.int16)
    
    # Compute reference output using float32 for accuracy, then convert to int16
    input_float = input_tensor.float()
    output_float = softmax(input_float)
    # Scale softmax output to int16 range (softmax outputs are in [0,1])
    output = (output_float * 32767).to(torch.int16)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie-mlir",
                       profile=True,
                       warmup=2000,
                       num_iters=10000,
                       )
        output_allo = np.zeros((tensor_size,)).astype(np.int16)
        mod(input_tensor.cpu().numpy(), output_allo)
        try:
            # Use a higher tolerance due to quantization effects and softmax numerical precision
            np.testing.assert_allclose(output_allo, output, rtol=1e-1, atol=100)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="kernels_func/softmax.cc")
    args = parser.parse_args()

    _test_softmax(args.kernel_path)

