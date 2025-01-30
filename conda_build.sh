#!/bin/bash

# Fail on error
set -e

# Move into the `bin` directory
cd src

# Compile using `make`
make

# Ensure binaries are executable
chmod +x bam2prof

# Move compiled binaries to Conda's bin directory
mkdir -p "${PREFIX}/bin"
mv bam2prof "${PREFIX}/bin/"
