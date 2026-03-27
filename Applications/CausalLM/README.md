# ☄️ CausalLM Inference with NNTrainer

- This application provides examples to run causal llm models using nntrainer.
- This example only provides *inference* mode, not *training* mode yet.

## Supported models

- Llama
- Qwen3 (1.7b/4b/7b/14b)
- Qwen3MoE (30b-A3b)
- Gpt-Oss-20b 
- You can try your own model with custom layers! 
- Feel free to contribute! 😊

## How to run

- download and copy the model files from hugingface to `res/{model}` directory.
- The folder should contain
    - config.json
    - generation_config.json
    - tokenizer.json
    - tokenizer_config.json
    - vocab.json
    - nntr_config.json
    - nntrainer weight binfile (matches with the name in nntr_config.json)
    - which are usuallyl included in HF model deployment.
- compile the Application
- If you test CausalLM on your PC, build with `-Denable-transformer=true`
- run the model with the following command

```
$ cd build/Applications/CausalLM
$ ./nntr_causallm {your model config folder}
```

e.g.,

```
$ ./nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3-4b/
```

### Recommended Configuration 

- PC test
```
$ meson build -Denable-fp16=true -Dthread-backend=omp -Denable-transformer=true -Domp-num-threads=4
$ export OMP_THREAD_LIMIT=16 && export OMP_WAIT_POLICY=active && export OMP_PROC_BIND=true && export OMP_PLACES=cores && export OMP_NUM_THREADS=4
```

- Android test
```
$ ./tools/package_android.sh -Domp-num-threads=4 -Dthread-backend=omp
```

## Supported Models

- Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 32B) [[link](https://huggingface.co/Qwen/Qwen3-4B)]
- Qwen3-MoE (30B-A3B) [[link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)]
- GPT-OSS (MoE: 20B, 120B) [[link](https://huggingface.co/openai/gpt-oss-20b)]

For more details, please refer to the [Model Documentation](models/README.md).

## Quantizing Models

NNTrainer provides a quantization utility (`nntr_quantize`) that converts FP32 CausalLM model weights to lower-precision data types, reducing model size for efficient on-device inference.

### Supported Quantization Types

| Data Type | Description |
|-----------|-------------|
| `FP32`    | 32-bit floating point (default for embedding/LM head) |
| `FP16`    | 16-bit floating point |
| `Q4_0`    | 4-bit quantization (default for FC layers) |
| `Q4_K`    | 4-bit K-quant quantization |
| `Q6_K`    | 6-bit K-quant quantization |

> **Note (Q4_0 platform dependency):** `Q4_0` quantization produces platform-specific binary formats — the output generated on x86 is **not compatible** with ARM, and vice versa. You must run `nntr_quantize` on the **same platform architecture** where the quantized model will be used for inference. Cross-platform quantization is not yet supported.


### Prerequisites

The model directory must contain the following files:
- `config.json` – model architecture configuration
- `generation_config.json` – generation parameters
- `nntr_config.json` – NNTrainer-specific configuration
- `.bin` weight file – FP32 model weights

### Building

The quantization utility is built automatically with the CausalLM application:

```bash
meson build && ninja -C build
# The executable is: build/Applications/CausalLM/nntr_quantize
```

### Usage

```
nntr_quantize <model_path> [options]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o <path>` | Output directory | Same as `<model_path>` |
| `--fc_dtype <type>` | Target dtype for FC (fully-connected) layers | `Q4_0` |
| `--embd_dtype <type>` | Target dtype for embedding layer | `FP32` |
| `--lmhead_dtype <type>` | Target dtype for LM head layer | Same as `embd_dtype` |
| `--output_bin <name>` | Output `.bin` filename | Auto-generated |
| `--config <path>` | Use a target `nntr_config.json` for dtype settings | – |

### Examples

```bash
# Quantize FC layers to Q4_0 (default), embedding stays FP32:
nntr_quantize /path/to/qwen3-4b

# Quantize FC layers to Q4_0 and embedding to Q6_K:
nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K

# Quantize to a different output directory:
nntr_quantize /path/to/qwen3-4b -o /output/qwen3-4b-q4

# Use a pre-configured target nntr_config.json:
nntr_quantize /path/to/qwen3-4b --config /path/to/target_nntr_config.json
```

### Output

The utility produces:
1. A quantized `.bin` weight file (filename auto-generated or specified via `--output_bin`)
2. A new `nntr_config_quantized.json` (or `nntr_config.json` if output directory differs from source)

After quantization, run the quantized model:
```bash
# If output is in the same directory:
mv /path/to/model/nntr_config_quantized.json /path/to/model/nntr_config.json
nntr_causallm /path/to/model

# If output is in a different directory:
cp /path/to/model/config.json /path/to/model/generation_config.json /output/dir/
nntr_causallm /output/dir
```