# AdDeam

AdDeam is a new tool designed to take a list of BAM files as input and generate damage profiles, which can be visualized to group similar profiles together. AdDeam uses a modified version of `bam2prof` (https://github.com/grenaud/bam2prof).

## Modes of Operation
AdDeam comes with two distinct modes:

- **Classic Mode**: Generates a single damage profile for the entire BAM file. This is useful when the BAM file represents a single sample/extraction/individual.
  
- **Meta Mode**: Generates individual damage profiles for each reference (contig, scaffold or chromosome) in the BAM file. This mode is especially beneficial for metagenomic datasets aligned against large reference databases or when you align the ancient DNA fragments back to the assembly to identify which contigs/scaffolds are likely to be contamination.

![Figure_1_Workflow](https://github.com/user-attachments/assets/ecdccd79-74b8-48f8-acdf-3c9ad6f7486e)

## Requirements
To ensure optimal performance and accurate results, the BAM files used with AdDeam must meet the following requirements:

- The BAM file **must include the MD flags**, which provide necessary information for determining nucleotide mismatches.
  
- The BAM file should be **sorted**, as this improves performance, especially for large files, by enabling efficient access to aligned reads.

### Adding Mismatch Annotations (MD Tags)
To ensure that the BAM file includes mismatch annotations relative to the reference (MD tags), use `samtools calmd` as shown in the following template:

    samtools calmd -b in_file.bam reference.fasta > out_file.bam

Please note that calmd also takes file descriptors and can be piped easily to/from "bwa", "bowtie", "samtools view" or "samtools sort".

## Key Features
AdDeam offers a variety of key features that make it a robust solution for damage profile generation and analysis:

- **Fast Damage Estimation**: AdDeam implements an early stopping criterion, allowing it to quickly estimate damage profiles, even for very large BAM files. If the damage profile has already converged, the analysis will halt early, saving computation time.

- **Individual Profiles for Each Reference**: In meta mode, AdDeam generates separate damage profiles for each reference, which is particularly useful when analyzing metagenomes aligned against comprehensive reference databases.

- **Clustering of Damage Profiles**: Using Gaussian Mixture Models (GMMs), AdDeam clusters the damage profiles, offering a fast and robust method to differentiate between undamaged samples and those with varying degrees of damage.

- **Comprehensive Analysis Output**:
    - Probability assignment of each sample to each cluster, facilitating downstream analysis.
    - Representative damage profiles for each cluster.
    - Principal Component Analysis (PCA) to visually separate clusters for easy identification of distinct damage levels.


## Installation

You can install AdDeam using either a Conda package or from source. Follow the instructions below based on your preferred method.

### 1. Install with Conda (via Bioconda) [DOESNT WORK YET]
To install AdDeam using Conda, simply run:

    conda install -c bioconda addeam

### 2. Install from Source

    git clone https://github.com/LouisPwr/AdDeam
    cd AdDeam/
    cd src/ && make
    cd ..
    pip install -r requirements.txt

Note:
bam2prof is a C++ tool used by AdDeam that is fast and robust for generating damage profiles from BAM files. It relies on htslib and samtools.



## To run AdDeam, follow these steps:

1. Generate Damage Profiles
Use the `addeam-bam2prof.py` wrapper to generate damage profiles from a list of BAM files. Specify the output directory for the profiles:

       python addeam-bam2prof.py -o profilesDir listOfBamFiles.txt

2. Cluster and Plot
Once the profiles are generated, cluster and visualize them using `addeam-cluster.py`. Specify the input directory for the profiles and an output directory for the plots:

       python addeam-cluster.py -i profilesDir -o plotsDir

## Output & Interpretation:

![Figure_2 Output Simulations](https://github.com/user-attachments/assets/2eee9e40-e067-4f43-a7bd-aa16d94632e4)

### Damage Profiles Directory
The `addeam-bam2prof.py` command creates a directory containing multiple `*.prof` files.

### Clustering Results with `addeam-cluster.py`
Running `addeam-cluster.py` on a directory of `*.prof` files generates the following output:

#### 1. PDF Damage Reports
- One report per number of clusters (`k`).
- By default, reports are generated for `k = 2`, `k = 3`, and `k = 4`.

#### 2. Excluded References File
- A text file listing damage profile IDs that were excluded due to insufficient aligned reads.
- These references were not included in the clustering process.

#### 3. PCA and GMM Directories

##### PCA Directory
- Contains figures showing only the PCA component.
- Useful for modular usage of plots.

##### GMM Directory
- Contains subdirectories for each `k` (e.g., `GMM/k2`, `GMM/k3`, etc.).
- Each subdirectory includes:
  - Representative damage profiles (`*.prof` files).
  - A TSV file containing probability assignments for each reference to different clusters.
    - Example: `addeamOutput/GMM/k3/cluster_report_k3.tsv` lists reference names with their probability assignments to clusters `1`, `2`, and `3`.
  - Individual GMM plots for modularity.



## Quick Start:
The directory testAdDeam in this repository contains 6 BAM files and a text file. After downloading, the text file needs to be updated with the paths to all BAM files.
The tool can be tested with these BAM files in both meta and classic mode - just specify the desired mode when running `addeam-bam2prof.py`

Running META mode:

`addeam-bam2prof.py -meta -o path/to/profiles/meta/outdir path/to/test.txt`

Running CLASSIC mode:

`addeam-bam2prof.py -classic -o path/to/profiles/classic/outdir path/to/test.txt`

Clustering:

`addeam-cluster.py -i path/to/profiles/meta/outdir -o path/to/clusters/meta/outdir`
