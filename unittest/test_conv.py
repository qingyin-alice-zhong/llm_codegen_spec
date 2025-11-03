# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Annotated
from allo.ir.types import float32, int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

Ly = Layout("RR")

# Convolution dimensions  
IN_H = 16     # Input height (smaller for int32)
IN_W = 16     # Input width (smaller for int32)
K_H = 3       # Kernel height
K_W = 3       # Kernel width
OUT_H = IN_H - K_H + 1  # Output height (14)
OUT_W = IN_W - K_W + 1  # Output width (14)

# PyTorch reference code starts
def conv2d_simple(input_tensor: Annotated[torch.Tensor, "shape: (16, 16)"], 
                  kernel: Annotated[torch.Tensor, "shape: (3, 3)"]) -> Annotated[torch.Tensor, "shape: (14, 14)"]:
    """
    Simple 2D convolution without padding
    input_tensor: input feature map (H x W)
    kernel: convolution kernel/filter (K_H x K_W)
    Returns: output feature map (OUT_H x OUT_W)
    """
    # Add batch and channel dimensions for PyTorch conv2d
    input_4d = input_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)       # [1, 1, K_H, K_W]
    
    # Perform convolution
    output_4d = F.conv2d(input_4d, kernel_4d, padding=0)
    
    # Remove batch and channel dimensions
    return output_4d.squeeze(0).squeeze(0)  # [OUT_H, OUT_W]
# PyTorch reference code ends


def _test_conv2d(kernel_path: str):
    
    conv = ExternalModule(
        top="conv2d_int32",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = int32

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(Input: Ty[IN_H, IN_W] @ Ly, Kernel: Ty[K_H, K_W] @ Ly, Output: Ty[OUT_H, OUT_W] @ Ly):
            conv(Input, Kernel, Output)

    # Create random input data
    input_tensor = torch.randint(-10, 10, (IN_H, IN_W), dtype=torch.int32)
    kernel_tensor = torch.randint(-2, 2, (K_H, K_W), dtype=torch.int32)  # Smaller kernel values
    output = conv2d_simple(input_tensor, kernel_tensor).to(torch.int32)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie-mlir",
                       profile=True,
                       warmup=100,
                       num_iters=1000,
                       )
        output_allo = np.zeros((OUT_H, OUT_W)).astype(np.int32)
        mod(input_tensor.cpu().numpy(), kernel_tensor.cpu().numpy(), output_allo)
        
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
    parser.add_argument("--kernel_path", type=str, default="kernels/conv.cc")
    args = parser.parse_args()

    _test_conv2d(args.kernel_path) 