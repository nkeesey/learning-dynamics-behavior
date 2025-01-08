# hash:sha256:3981aaf8effdf6183cd4b7d607dacf24aa1606ed89c09c441b7824ed5b0ec303
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        matplotlib==3.9.2 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        python==3.10.16 \
        scikit-learn==1.5.2 \
        scipy==1.14.1 \
        seaborn==0.13.2 \
        statsmodels==0.14.4 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-dynamic-foraging-basic-analysis==0.3.7 \
    aind-dynamic-foraging-data-utils==0.1.3 \
    joblib \
    pynwb==2.8.3

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
