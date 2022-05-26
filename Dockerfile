<<<<<<< HEAD
FROM  python:3.10

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
=======


FROM  python:3.7-alpine

LABEL maintainer=achillesrasquinha@gmail.com

ENV GEMPY_PATH=/usr/local/src/gempy

RUN apk add --no-cache \
        bash \
        git \
    && mkdir -p $GEMPY_PATH

COPY . $GEMPY_PATH
COPY ./docker/entrypoint.sh /entrypoint
RUN sed -i 's/\r//' /entrypoint \
	&& chmod +x /entrypoint
>>>>>>> template/master

WORKDIR $GEMPY_PATH

RUN pip install -r ./requirements.txt && \
    python setup.py install

<<<<<<< HEAD
ENTRYPOINT ["/entrypoint.sh"]
=======
ENTRYPOINT ["/entrypoint"]
>>>>>>> template/master

CMD ["gempy"]