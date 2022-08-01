FROM  python:3.9

ARG DEVELOPMENT=false

LABEL maintainer=achillesrasquinha@gmail.com

ENV GEMPY_PATH=/usr/local/src/gempy \
    DIAMOND_VERSION=2.0.13 \
    BLAST_PLUS_VERSION=2.13.0 

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        git \
    && mkdir -p $GEMPY_PATH \
    && wget http://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-${BLAST_PLUS_VERSION}+-x64-linux.tar.gz -O /blast_plus.tar.gz \
    && tar xzf /blast_plus.tar.gz \
    && mv /ncbi-blast-${BLAST_PLUS_VERSION}+/bin/update_blastdb.pl /usr/local/bin \
    && wget http://github.com/bbuchfink/diamond/releases/download/v${DIAMOND_VERSION}/diamond-linux64.tar.gz -O /diamond.tar.gz \
    && tar xzf /diamond.tar.gz \
    && mv /diamond /usr/local/bin \
    && rm -rf \
        /blast_plus.tar.gz \
        /ncbi-blast-${BLAST_PLUS_VERSION}+ \
        /diamond.tar.gz

COPY . $GEMPY_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $GEMPY_PATH

SHELL ["/bin/bash", "-c"]

RUN if [[ "${DEVELOPMENT}" ]]; then \
        pip install -r ./requirements-dev.txt; \
        python setup.py develop; \
    else \
        pip install -r ./requirements.txt; \
        python setup.py install; \
    fi
    
ENTRYPOINT ["/entrypoint.sh"]

CMD ["gempy"]