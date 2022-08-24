FROM ghcr.io/achillesrasquinha/gempy:base

ARG DEVELOPMENT=false

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