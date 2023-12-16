# Dummy file to allow editable installs
from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="benchmarks",
        packages=find_packages(),
        package_data={
            # If any package contains *.txt or *.json files, include them:
            "": ["*.csv"],
            # And include any files found in the 'mypackage/data' directory:
            "benchmarks": ["datasets/*"],
        },
    )
