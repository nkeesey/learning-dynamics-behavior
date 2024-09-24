# hash:sha256:fb52d08f5efe89986e45ae283ee8133308c7ce3a8a7f0041e8ed08ef35eb0ffe
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
        scikit-learn==1.5.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    pynwb==2.6.0 \
    seaborn==0.13.2 \
    statsmodels==0.14.2

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
