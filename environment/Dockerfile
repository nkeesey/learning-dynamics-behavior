# hash:sha256:7b34fd1536cab8e6e9fed790a7a56092faf7e6ec3c18ba25ad67c2af0a525f1e
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib \
        numpy \
        os \
        pandas \
        pynwb \
        seaborn \
    && conda clean -ya
