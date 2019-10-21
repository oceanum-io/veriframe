#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "attrdict",
    "Click>=6.0",
    "matplotlib",
    "numpy",
    "owslib",
    "pandas",
    "pyyaml",
    "xarray",
    "scipy",
    "tabulate",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

setup(
    author="Oceanum Developers",
    author_email="developers@oceanum.science",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    description="Library for model verification",
    entry_points={"console_scripts": ["onverify=onverify.cli:main"]},
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="onverify",
    name="onverify",
    packages=find_packages(include=["onverify"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.com/oceanum/onverify",
    version="0.1.0",
    zip_safe=False,
)
