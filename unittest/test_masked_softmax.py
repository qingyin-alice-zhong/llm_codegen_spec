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
Ly_1 = Layout("R")

# Masked Softmax dimensions
SEQ_LEN_TILED = 16
SEQ_LEN = 64     # Sequence length

# PyTorch reference code starts
def masked_softmax_tiled(attention_score_tile: Annotated[torch.Tensor, "shape: (16, 64), dtype: float32"], 
                        tile_row_start: Annotated[int, "shape: (1), dtype: int32"]) -> Annotated[torch.Tensor, "shape: (16, 64), dtype: float32"]:
    """
    Causal masked softmax for attention scores (tiled version)
    attention_score_tile: raw attention scores tile (TILE_ROWS x SEQ), dtype=float32
    tile_row_start: starting row index of this tile in the full (64, 64) matrix
    Returns: softmax attention weights with causal masking (TILE_ROWS x SEQ), dtype=float32
    
    Note: This processes a horizontal tile of the attention matrix.
    Each row in the tile gets its own softmax, with causal masking based on global position.
    """
    TILE_ROWS, SEQ = attention_score_tile.shape
    
    # Create causal mask for this tile based on global row positions
    mask = torch.zeros(TILE_ROWS, SEQ, dtype=torch.bool)
    for i in range(TILE_ROWS):
        global_row_idx = tile_row_start + i
        # Mask positions where column_idx > global_row_idx (future tokens)
        mask[i, global_row_idx + 1:] = True
    
    # Apply mask by setting masked positions to -inf
    masked_scores = attention_score_tile.masked_fill(mask, float('-inf'))
    
    # Apply softmax along the last dimension (key dimension)
    attn_weights = F.softmax(masked_scores, dim=-1)
    
    return attn_weights
# PyTorch reference code ends


def _test_masked_softmax_tiled(kernel_path: str):
    
    masked_softmax_kernel = ExternalModule(
        top="masked_softmax_float32",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = float32
    Ty_1 = int32

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(Input: Ty[SEQ_LEN_TILED, SEQ_LEN] @ Ly, TileRowStart: Ty_1[1] @ Ly_1, Output: Ty[SEQ_LEN_TILED, SEQ_LEN] @ Ly):
            masked_softmax_kernel(Input, TileRowStart, Output)

    # Create random input data
    input_tensor = torch.randn(SEQ_LEN_TILED, SEQ_LEN, dtype=torch.float32)
    tile_row_start = 0
    tile_row_start_tensor = torch.tensor([tile_row_start], dtype=torch.int32)
    output = masked_softmax_tiled(input_tensor, tile_row_start_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, 
                       target="aie-mlir",
                       profile=True,
                       warmup=100,
                       num_iters=1000,
                       )
        output_allo = np.zeros((SEQ_LEN_TILED, SEQ_LEN)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), tile_row_start_tensor.cpu().numpy(), output_allo)
        
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
    parser.add_argument("--kernel_path", type=str, default="kernels/masked_softmax.cc")
    args = parser.parse_args()

    _test_masked_softmax_tiled(args.kernel_path)