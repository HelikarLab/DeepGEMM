FROM ghcr.io/achillesrasquinha/deeply:latest

LABEL maintainer=achillesrasquinha@gmail.com

ENV DGEMM_PATH=/dgemm \
    DIAMOND_VERSION=2.0.13 \
    BLAST_PLUS_VERSION=2.13.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        git \
        cmake \
        swig \
        libglpk-dev \
        glpk-utils \
    && pip install --upgrade pip \
    && mkdir -p $DGEMM_PATH \
    && wget http://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-${BLAST_PLUS_VERSION}+-x64-linux.tar.gz -O ./blast_plus.tar.gz \
    && tar xvzf ./blast_plus.tar.gz \
    && mv ./ncbi-blast-${BLAST_PLUS_VERSION}+/bin/update_blastdb.pl /usr/local/bin \
    && wget http://github.com/bbuchfink/diamond/releases/download/v${DIAMOND_VERSION}/diamond-linux64.tar.gz -O ./diamond.tar.gz \
    && tar xzf ./diamond.tar.gz \
    && mv ./diamond /usr/local/bin \
    && rm -rf \
        ./blast_plus.tar.gz \
        ./ncbi-blast-${BLAST_PLUS_VERSION}+ \
        ./diamond.tar.gz