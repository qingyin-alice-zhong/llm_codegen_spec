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
def eltwise_add(x: Annotated[torch.Tensor, "shape: (1024,)"], y: Annotated[torch.Tensor, "shape: (1024,)"]) -> Annotated[torch.Tensor, "shape: (1024,)"]:
    """
    x: first input vector tensor
    y: second input vector tensor
    """
    return x + y
# PyTorch reference code ends

def _test_eltwise_add(kernel_path: str):
    
    add_kernel = ExternalModule(
        top="eltwise_add_int16_vector",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = int16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly, C: Ty[M] @ Ly):
            add_kernel(A, B, C)

    input_tensor_a = torch.randint(-100, 100, (tensor_size,), dtype=torch.int16)
    input_tensor_b = torch.randint(-100, 100, (tensor_size,), dtype=torch.int16)
    output = eltwise_add(input_tensor_a, input_tensor_b).to(torch.int16)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie-mlir",
                       profile=True,
                       warmup=2000,
                       num_iters=10000,
                       )
        output_allo = np.zeros((tensor_size,)).astype(np.int16)
        mod(input_tensor_a.cpu().numpy(), input_tensor_b.cpu().numpy(), output_allo)
        try:
            np.testing.assert_allclose(output_allo, output, rtol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="kernels/add_new.cc")
    args = parser.parse_args()

    _test_eltwise_add(args.kernel_path)
