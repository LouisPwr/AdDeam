# AdDeam

AdDeam is a powerful tool designed to take a list of BAM files as input and generate damage profiles, which can be visualized to group similar profiles together. It is particularly useful for analyzing ancient DNA and other degraded DNA samples, where determining and clustering damage patterns is essential.

## Modes of Operation
AdDeam comes with two distinct modes:

- **Classic Mode**: Generates a single damage profile for the entire BAM file. This is useful when the BAM file represents a single organism or when a general overview of the damage is sufficient.
  
- **Meta Mode**: Generates individual damage profiles for each reference in the BAM file. This mode is especially beneficial for metagenomic datasets aligned against large reference databases, allowing for the assessment of damage patterns for each individual reference.

## Requirements
To ensure optimal performance and accurate results, the BAM files used with AdDeam must meet the following requirements:

- The BAM file **must include the MD flags**, which provide necessary information for determining nucleotide mismatches.
  
- The BAM file should be **sorted**, as this improves performance, especially for large files, by enabling efficient access to aligned reads.

## Key Features
AdDeam offers a variety of key features that make it a robust solution for damage profile generation and analysis:

- **Fast Damage Estimation**: AdDeam implements an early stopping criterion, allowing it to quickly estimate damage profiles, even for very large BAM files. If the damage profile has already converged, the analysis will halt early, saving computation time.

- **Individual Profiles for Each Reference**: In meta mode, AdDeam generates separate damage profiles for each reference, which is particularly useful when analyzing metagenomes aligned against comprehensive reference databases.

- **Clustering of Damage Profiles**: Using Gaussian Mixture Models (GMMs), AdDeam clusters the damage profiles, offering a fast and robust method to differentiate between undamaged samples and those with varying degrees of damage.

- **Comprehensive Analysis Output**:
    - Probability assignment of each sample to each cluster, facilitating downstream analysis.
    - Representative damage profiles for each cluster.
    - Principal Component Analysis (PCA) to visually separate clusters for easy identification of distinct damage levels.

## How to Install
(to be continued)
