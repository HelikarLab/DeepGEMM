FROM  python:3.7-alpine

LABEL maintainer=achillesrasquinha@gmail.com

ENV GEMPY_PATH=/usr/local/src/gempy

RUN apk add --no-cache \
        bash \
        git \
    && mkdir -p $GEMPY_PATH

COPY . $GEMPY_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $GEMPY_PATH

RUN pip install -r ./requirements.txt && \
    python setup.py install

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gempy"]