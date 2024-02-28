# hash:sha256:000d95f6098fddbd500a3739602efd69ae36873fe312e17588f6663bb21bcbb0
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        matplotlib \
        pynwb \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        os \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    pandas

RUN pip3 install -U --no-cache-dir \
    seaborn
