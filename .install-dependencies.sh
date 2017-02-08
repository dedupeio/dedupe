#!/usr/bin/env bash

set -e -x

pip install --upgrade pip
pip install wheel
pip install "numpy>=1.9.2"
pip install --upgrade -r requirements.txt
pip install coveralls
cython src/*.pyx
pip install -e .