try:
    from setuptools import Extension, setup
except ImportError:
    raise ImportError(
        "setuptools module required, please go to https://pypi.python.org/pypi/setuptools and follow the instructions for installing setuptools"
    )

from Cython.Build import cythonize

setup(
    ext_modules=cythonize([Extension("dedupe.cpredicates", ["dedupe/cpredicates.pyx"])])
)
