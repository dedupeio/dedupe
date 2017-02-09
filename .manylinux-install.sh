#!/usr/bin/env bash

set -e -x

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "${PYBIN}" == *"cp27"* ]] || [[ "${PYBIN}" == *"cp34"* ]] || [[ "${PYBIN}" == *"cp35"* ]]; then
        "${PYBIN}/pip" install "numpy>=1.9.2"
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" install coveralls
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
        cd /io/
        "${PYBIN}/python" tests/canonical.py -vv
        rm canonical_learned_settings
        cd /
    fi
done

# If everything works, upload wheels to PyPi
travis=$( cat /io/.travis_tag )
PYBIN34="/opt/python/cp34-cp34m/bin"
# if [[ $travis ]]; then
"${PYBIN34}/pip" install twine
"${PYBIN34}/twine" upload --config-file /io/.pypirc /io/wheelhouse/dedupe*.whl
# fi
