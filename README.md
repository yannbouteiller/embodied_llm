# embodied_llm

LLMs that speak and see

## Prerequisite:

```bash
sudo apt update
sudo apt install build-essential
sudo apt install portaudio19-dev
conda install make cmake
```

After installing the library, uninstall pytorch and reinstall it via conda:
```bash
pip uninstall torch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Because `RealtimeSTT` and `RealtimeTTS` have absurd dependency management, they are not specified in the requirements and you need to install them separately.

If using conda on Ubuntu 24:
```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
```

## Launch server:

```bash
python3 -m llama_cpp.server --model ~/ellm/ggml-model-q5_k.gguf --clip_model_path ~/ellm/mmproj-model-f16.gguf --chat_format llava-1-5 --n_threads 4 --n_gpu_layers -1 --n_ctx 8192
```
