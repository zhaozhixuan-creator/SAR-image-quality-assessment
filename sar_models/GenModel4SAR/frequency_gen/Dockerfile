# syntax=docker/dockerfile:1.6

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda

ENV CONDA_DIR=${CONDA_DIR} \
    CONDA_DEFAULT_ENV=openmmlab \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
ENV PATH=${CONDA_DIR}/envs/openmmlab/bin:${CONDA_DIR}/bin:${PATH}
ENV PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        bzip2 \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
        https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true

WORKDIR /workspace

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm -rf /root/.cache/pip /tmp/environment.yml

COPY . /workspace
RUN chmod +x train.sh test.sh tools/dist_train.sh \
    && mkdir -p /workspace/data /workspace/work_dirs

VOLUME ["/workspace/data", "/workspace/work_dirs"]

CMD ["/bin/bash"]
