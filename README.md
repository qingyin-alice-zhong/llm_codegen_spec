# llm_codegen_spec
An enhanced version of llm_codegen with spec-kit integration for better kernel code generation on NPU architectures. 
(version for one kernel)

# Example comments to start 5 iterations:
```bash
cd ~/mlir-aie/allo
conda activate allo
cd ..
source utils/env_setup.sh
cd ~/llm_codegen_spec
#pip install google-generativeai openai
#export GOOGLE_API_KEY=AIzaSyCU7vVkguDsu6r46fGLjAOXYFkVprhlrhM
#!/bin/bash
# need to first activate aie env as below
conda activate aie
source /opt/common/setupVitis.sh
source /opt/common/setupXRT.sh
source ~/mlir-aie/utils/env_setup.sh
# Allo env setup
export MLIR_AIE_EXTERNAL_KERNEL_DIR=/home/qz425/mlir-aie/aie_kernels
export RUNTIME_LIB_DIR=/home/qz425/mlir-aie/aie_runtime_lib
export USE_AIE_MLIR_BUILDER=1
# Use latest cmake
export PATH=/opt/cmake-3.31.5-linux-x86_64/bin:/opt/llvm-project-19.x/build/bin:$PATH
export LLVM_BUILD_DIR=/opt/llvm-project-19.x/build
# setup LDPRELOAD to allow mlir-aie gemm able to be compiled
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

python main.py --provider google --model gemini-2.5-flash --backend npu_new --max_iterations 5 --kernel dotproduct_bias_relu_int8
```

# Best output:
 Speedup: 10.38x
