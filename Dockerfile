# FROM python:3.8-buster
FROM ubuntu:18.04

MAINTAINER "Jonas Kratochvil"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    sudo \
    git \
    ca-certificates \
    wget \
    patch \
    vim \
    gcc

RUN git clone https://github.com/jonaskratochvil/bert_punctuation.git /bert_punctuation

WORKDIR /bert_punctuation/

RUN pip3 install -r requirements.txt && \
    pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html && \
    pip3 install --no-cache boto3

# Here change from default to valid KEYS
ARG AWS_ACCESS_KEY_ID=default
ARG AWS_SECRET_ACCESS_KEY=default

RUN python3 download_from_S3.py \
        $AWS_ACCESS_KEY_ID \
        $AWS_SECRET_ACCESS_KEY \
        "parrot-asr-models" \
        "punctuation/model.bin" \
        /bert_punctuation/model.bin
