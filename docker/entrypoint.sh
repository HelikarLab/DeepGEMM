#!/bin/bash

set -euo pipefail

<<<<<<< HEAD
# if [ "${1:0:1}" = "-" ]; then
#     set -- gempy "$@"
# fi
=======
if [ "${1:0:1}" = "-" ]; then
    set -- gempy "$@"
fi
>>>>>>> template/master

exec "$@"