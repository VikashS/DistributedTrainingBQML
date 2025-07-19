# setup.py
from setuptools import setup, find_packages

setup(
    name="fraud",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g.,
        "pandas",
        "scikit-learn",
        "google-cloud-storage",
        # Add 'gcsfs' if you use pandas to read/write directly to GCS
        "gcsfs",
    ],
)