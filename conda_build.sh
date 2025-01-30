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

# Move back to the main directory
cd ..

# Ensure the Python scripts are executable and installed
mkdir -p "${PREFIX}/bin"
cp bam2prof.py cluster.py "${PREFIX}/bin/"

# Ensure Python scripts are executable
chmod +x "${PREFIX}/bin/bam2prof.py"
chmod +x "${PREFIX}/bin/cluster.py"
