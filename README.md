# llm_codegen_spec
An enhanced version of llm_codegen with spec-kit integration for better kernel code generation on NPU architectures. 

# Example comments to start 5 iterations:
```bash
cd ~/mlir-aie/allo
conda activate allo
cd ..
source utils/env_setup.sh
cd ~/llm_codegen_spec

python main.py --provider google --model gemini-2.5-flash --backend npu_new --max_iterations 5 --kernel dotproduct_bias_relu_int8
```

