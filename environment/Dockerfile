# hash:sha256:8c56dda332177725993b90a101fdaf0b95c8bc519107e53955c45b86d05d52ea
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUSTOM_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.8.0 \
        numpy==1.26.4 \
        pandas==2.1.4 \
        scikit-learn==1.5.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-dynamic-foraging-basic-analysis==0.3.4 \
    aind-dynamic-foraging-data-utils==0.1.1 \
    codeocean==0.2.0 \
    seaborn==0.13.2 \
    statsmodels==0.14.2

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
