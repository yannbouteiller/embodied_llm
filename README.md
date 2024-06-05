# embodied_llm

LLMs that speak and see

## Installation (Anaconda):

### Install dependencies:

USING PYTHON 3.9 AND NOTHING ELSE IS IMPORTANT:
```bash
conda create -n transformers python=3.9 -y
conda activate transformers
# sudo apt install build-essential gcc make cmake
# conda install -c conda-forge gxx -y
# conda install make cmake -y
# conda install -c conda-forge gcc
conda install -c conda-forge libstdcxx-ng
```

Needed to install pyaudio dependencies:
```bash
conda install pyaudio -y
conda uninstall pyaudio -y
```

These libraries have absurd dependency management, install them first:
```bash
pip install RealTimeSTT
pip install RealTimeTTS
```

Install remaining dependencies:
```bash
pip install -e .
```

Fix pyaudio:
```bash
pip uninstall pyaudio
conda uninstall pyaudio
conda install pyaudio
```

### Download files:

```bash
mkdir ~/ellm
cd ~/ellm
wget https://huggingface.co/mys/ggml_bakllava-1/resolve/main/ggml-model-q5_k.gguf
wget https://huggingface.co/mys/ggml_bakllava-1/resolve/main/mmproj-model-f16.gguf
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alba/medium/en_GB-alba-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json
```

### Usage:

```bash
find ~/. -name "libcudnn_ops_infer.so.8"
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
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yann/./miniconda3/envs/transformers/lib/python3.9/site-packages/nvidia/cudnn/lib python embodied_llm/agent/ellm.py
```


### Other tested stuff:

CUDA:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

CuDNN: `Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x` [here](https://developer.nvidia.com/rdp/cudnn-archive)
```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.7.0.84_1.0-1_amd64.deb
```

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
```

```bash
sudo apt install nvidia-cuda-toolkit
```

## Launch server:

```bash
conda activate transformers
python3 -m llama_cpp.server --model ~/ellm/ggml-model-q5_k.gguf --clip_model_path ~/ellm/mmproj-model-f16.gguf --chat_format llava-1-5 --n_threads 4 --n_gpu_layers -1 --n_ctx 8192
```
