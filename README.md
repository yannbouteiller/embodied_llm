# embodied_llm

LLMs that speak and see

## Installation (Anaconda):

### Install dependencies:

USING PYTHON 3.9 AND NOTHING ELSE IS IMPORTANT:
```bash
conda create -n transformers python=3.9 -y
conda activate transformers
conda install -c conda-forge gxx -y
conda install make cmake -y
conda install -c conda-forge libstdcxx-ng -y
# sudo apt install build-essential gcc make cmake
# conda install -c conda-forge gxx -y
# conda install make cmake -y
# conda install -c conda-forge gcc
```

```bash
conda install pyaudio -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

These libraries have absurd dependency management, install them with no-deps:
```bash
pip install RealTimeSTT --no-deps
pip install RealTimeTTS --no-deps
pip install -r requirements.txt
```

Install remaining dependencies:
```bash
pip install -e .
```

### Download files:

```bash
mkdir ~/ellm
cd ~/ellm
wget https://huggingface.co/mys/ggml_bakllava-1/resolve/main/ggml-model-q5_k.gguf
wget https://huggingface.co/mys/ggml_bakllava-1/resolve/main/mmproj-model-f16.gguf
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alba/medium/en_GB-alba-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json
wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

And adapt the content of `server_config.txt` to your own path.

(Note: you can choose other models and adapt `server_config.txt` accordingly)


## Launch server:

Adapt:

```bash
conda activate transformers

python3 -m llama_cpp.server --config_file ~/Desktop/git/embodied_llm/server_config.txt

# For reference, we use to do:
# python3 -m llama_cpp.server --model ~/ellm/ggml-model-q5_k.gguf --clip_model_path ~/ellm/mmproj-model-f16.gguf --chat_format llava-1-5 --n_threads 4 --n_gpu_layers -1 --n_ctx 8192
```

### Usage:

```bash
sudo find ~/. -name "libcudnn_ops_infer.so.8"
```

You should get an output like:
```
/home/yann/./miniconda3/envs/transformers/lib/python3.9/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8
/home/yann/./miniconda3/pkgs/pytorch-2.3.1-py3.10_cuda11.8_cudnn8.7.0_0/lib/python3.10/site-packages/torch/lib/libcudnn_ops_infer.so.8
/home/yann/./miniconda3/pkgs/cudnn-8.9.2.26-cuda12_0/lib/libcudnn_ops_infer.so.8
/home/yann/./miniconda3/pkgs/pytorch-2.3.1-py3.8_cuda11.8_cudnn8.7.0_0/lib/python3.8/site-packages/torch/lib/libcudnn_ops_infer.so.8
/home/yann/./miniconda3/pkgs/pytorch-2.3.1-py3.9_cuda11.8_cudnn8.7.0_0/lib/python3.9/site-packages/torch/lib/libcudnn_ops_infer.so.8
```

To launch ellm, replace by the equivalent on your system in the following command:
```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yann/./miniconda3/envs/transformers/lib/python3.9/site-packages/torch/lib python embodied_llm/agent/ellm.py
```
