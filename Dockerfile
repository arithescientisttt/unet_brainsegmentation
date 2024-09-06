# Base image from NVIDIA's CUDA repository with cuDNN support
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  netbase \
  wget \
  git \
  openssh-client \
  ssh \
  vim \
  && rm -rf /var/lib/apt/lists/*

# Set environment variables to fix Python encoding issues
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

# Install Python 3.6 and essential development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.6 \
  python3.6-dev \
  python3-pip \
  python3-setuptools \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Install specific versions of PyTorch and torchvision compatible with CUDA 10.0
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl \
  && pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# Set the working directory inside the container
WORKDIR /workspace

# Copy the Python package dependencies file
COPY requirements.txt ./

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
