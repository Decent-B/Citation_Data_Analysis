"""
Setup script for the ranking package.
"""

from setuptools import setup, find_packages

setup(
    name="ranking",
    version="0.1.0",
    description="Academic paper ranking using citation networks and PageRank",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "networkx>=3.0",
    ],
    extras_require={
        "gpu": [
            "cudf",
            "cugraph",
            "cupy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
