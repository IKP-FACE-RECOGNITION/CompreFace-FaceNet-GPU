ARG BASE_IMAGE
# FROM ${BASE_IMAGE:-nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04}
FROM ${BASE_IMAGE:-nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA=11.8
# ENV CUDA=12.3

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         software-properties-common \
# 		curl \
# 		pkg-config \
# 		unzip \
#     	python3-dev \
#     	python3-distutils \
#     && rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		build-essential \
        software-properties-common \
		curl \
		pkg-config \
		unzip \
		python3 \
        python3-dev \
        python3-distutils \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*


# RUN python3 -m pip install --no-cache-dir --upgrade pip setuptool
## get-pip bootstrap not needed because python3-pip is installed via apt; just upgrade
## (previously duplicated bootstrap commands removed to reduce layers and failure risk)


# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN python3 -m pip --no-cache-dir install --upgrade pip setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Variables for Tensorflow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Variables for MXNET
ENV MXNET_CPU_WORKER_NTHREADS=24
ENV MXNET_ENGINE_TYPE=ThreadedEnginePerDevice MXNET_CUDNN_AUTOTUNE_DEFAULT=0

# No access to GPU devices in the build stage, so skip tests
ENV SKIP_TESTS=1

# The number of processes depends on GPU memory.
# Keep in mind that one uwsgi process with InsightFace consumes about 2.5GB memory
