FROM ghcr.io/achillesrasquinha/dgemm:base

ARG DEVELOPMENT=false

COPY . $DGEMM_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $DGEMM_PATH

SHELL ["/bin/bash", "-c"]

RUN if [[ "${DEVELOPMENT}" ]]; then \
        pip install -r ./requirements-dev.txt; \
        python setup.py develop; \
    else \
        pip install -r ./requirements.txt; \
        python setup.py install; \
    fi
    
ENTRYPOINT ["/entrypoint.sh"]

CMD ["dgemm"]