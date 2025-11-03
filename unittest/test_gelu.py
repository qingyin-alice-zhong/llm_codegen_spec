# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Annotated
from allo.ir.types import float32, int4, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.experimental.external_kernel import ExternalModule

Ly = Layout("RR")

# GeLU dimensions
SEQ_LEN = 4      # Sequence length
FFN_HID = 768    # Feed-forward hidden dimension

# PyTorch reference code starts
def gelu_activation(input_tensor: Annotated[torch.Tensor, "shape: (4, 768), dtype: float32"]) -> Annotated[torch.Tensor, "shape: (4, 768), dtype: float32"]:
    """
    GELU activation function
    input_tensor: input tensor from feed-forward network (SEQ x FFN_HID)
    Returns: GELU activated tensor (SEQ x FFN_HID)
    """
    gelu_func = torch.nn.GELU()
    return gelu_func(input_tensor)
# PyTorch reference code ends


def _test_gelu(kernel_path: str):
    
    gelu = ExternalModule(
        top="gelu_float32",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = float32

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(Input: Ty[SEQ_LEN, FFN_HID] @ Ly, Output: Ty[SEQ_LEN, FFN_HID] @ Ly):
            gelu(Input, Output)

    # Create random input data
    input_tensor = torch.randn(SEQ_LEN, FFN_HID, dtype=torch.float32)
    output = gelu_activation(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie-mlir",
                       profile=True,
                       warmup=100,
                       num_iters=768,
                       )
        output_allo = np.zeros((SEQ_LEN, FFN_HID)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        
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
    parser.add_argument("--kernel_path", type=str, default="kernels/gelu.cc")
    args = parser.parse_args()

    _test_gelu(args.kernel_path)

