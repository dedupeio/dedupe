#!/usr/bin/env bash

set -e -x

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "${PYBIN}" == *"cp27"* ]] || [[ "${PYBIN}" == *"cp34"* ]] || [[ "${PYBIN}" == *"cp35"* ]]; then
        "${PYBIN}/pip" install "numpy>=1.9.2"
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/cython" /io/src/*.pyx
        "${PYBIN}/pip" install -e /io/
        "${PYBIN}/pip" wheel /io/ -w wheelhouse/
    fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/dedupe*.whl; do
    if  [[ "${whl}" != *"dedupe_hcluster"* ]]; then
        auditwheel repair "$whl" -w /io/wheelhouse/
    fi
done

# Install packages and test
for PYBIN in /opt/python/*/bin; do
    if [[ "${PYBIN}" == *"cp27"* ]] || [[ "${PYBIN}" == *"cp34"* ]] || [[ "${PYBIN}" == *"cp35"* ]]; then
        "${PYBIN}/pip" uninstall -y dedupe
        "${PYBIN}/pip" install dedupe --no-index -f /io/wheelhouse
        "${PYBIN}/pytest" /io/tests --cov dedupe
    fi
done

