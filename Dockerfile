FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglm-dev \
    libavcodec-extra \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    ffmpeg \
    mesa-va-drivers \
    vainfo \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

RUN /opt/conda/bin/conda init bash
ENV PATH=/opt/conda/bin:$PATH

# Настраиваем conda (без auto_activate_base)
RUN conda config --set auto_activate_base false

# Принимаем лицензионное соглашение Anaconda (интерактивно)
RUN yes | conda tos accept --channel https://repo.anaconda.com/pkgs/main || true
RUN yes | conda tos accept --channel https://repo.anaconda.com/pkgs/r || true

RUN conda create -n mvsplat python=3.10 -y && \
    conda run -n mvsplat pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace

# Копируем все файлы проекта
COPY src /workspace/src
COPY requirements.txt /workspace/requirements.txt
COPY constraints.txt /workspace/constraints.txt

RUN conda run -n mvsplat pip install -r /workspace/requirements.txt -c /workspace/constraints.txt
RUN pip uninstall moviepy -y && pip install moviepy==1.0.3

RUN git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization && cd diff-gaussian-rasterization && python setup.py install

RUN echo "conda activate mvsplat" >> ~/.bashrc

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV DISABLE_ITT=1

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN mkdir /workspace/checkpoints

CMD ["/bin/bash"]