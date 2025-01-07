# hash:sha256:a5c8ce22041dba1a7b516c3f210907185a3fead00972071d5930d2ab5dee14bf
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        seaborn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-dynamic-foraging-basic-analysis \
    multiprocessing \
    pathlib \
    pynwb \
    tqdm
