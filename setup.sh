#!/usr/bin/env sh

# Initialize learn2learn, we need the latest pytorch lightning functionality
git submodule update --init --recursive
cd learn2learn && python setup.py build_ext --inplace && python setup.py sdist
