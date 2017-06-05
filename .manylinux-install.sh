#!/usr/bin/env bash

set -e -x

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "${PYBIN}" == *"cp27"* ]] || [[ "${PYBIN}" == *"cp34"* ]] || [[ "${PYBIN}" == *"cp35"* ]] || [[ "${PYBIN}" == *"cp36"* ]]; then
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/cython" /io/src/*.pyx
        "${PYBIN}/pip" install -e /io/
        "${PYBIN}/pip" wheel /io/ -w wheelhouse/
        rm -rf /io/build /io/*.egg-info
    fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/dedupe*.whl; do
    if  [[ "${whl}" != *"dedupe_"* ]]; then
        auditwheel repair "$whl" -w /io/wheelhouse/
    fi
done

