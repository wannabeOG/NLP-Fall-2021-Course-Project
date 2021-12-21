FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV ILMULTI_CORPUS_ROOT="/root/datasets_folder/"
ARG ILMULTI_CORPUS_ROOT="/root/datasets_folder/"
# MODEL_CHECKPOINTS are @ "/root/model_checkpoints/"
# CODE is @ "/root/src/"
# config file @ "/root/src/fairseq-ilmt/config.yaml"

RUN apt update \
    && apt install -y htop build-essential python3-dev wget vim git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n ml python=3.7

COPY . /root/src/
RUN /bin/bash -c "mkdir -p /root/datasets_folder/ \
    && cd /root/src/ilmulti \
    && source activate ml \
		&& pip install  torch==1.1.0 torchvision==0.3.0 \
		&& pip install numpy==1.16.0 \
    && pip install -r requirements.txt \
		&& python setup.py install \
    # && bash scripts/download-and-setup-models.sh \
    && pip install pyyaml \
    && pip install lmdb \
		&& cd ../fairseq-ilmt \
    && python preprocess_cvit.py config.yaml"