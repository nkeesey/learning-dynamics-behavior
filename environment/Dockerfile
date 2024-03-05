# hash:sha256:36bd6458541dc011f54339a0e97a9973a33c51e2a9deefb6b352e296f0733b23
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.8.0 \
        numpy==1.26.4 \
        pandas==2.1.4 \
        s3fs==2023.10.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    pynwb==2.6.0 \
    seaborn==0.13.2
