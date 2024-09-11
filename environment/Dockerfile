# hash:sha256:0e079a43cb5a5494d1ceb911c7e6ddd58aa86dc71a473312830c76b4ccaee77e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.8.0 \
        numpy==1.26.4 \
        pandas==2.1.4 \
        plotly==5.19.0 \
        s3fs==2023.10.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    pynwb==2.6.0 \
    seaborn==0.13.2 \
    statsmodels==0.14.2
