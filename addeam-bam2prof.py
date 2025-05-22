#!/usr/bin/env python3
# coding: utf-8

import subprocess
import shutil
import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def find_bam2prof():
#     """Find the compiled `bam2prof` binary in development and Conda environments."""
    
#     # Case 1: Inside a Conda-installed package (expected after Conda installation)
#     #conda_bin = Path(sys.prefix) / "bin" / "bam2prof"
#     conda_bin = Path(__file__).parent / "bam2prof"
#     if conda_bin.exists():
#         return conda_bin

#     # Case 2: Development mode (binary inside `src/` directory)
#     dev_bin = Path(__file__).parent / "src" / "bam2prof"
#     if dev_bin.exists():
#         return dev_bin

#     # If neither exists, exit with an error
#     print("Error: bam2prof binary not found!", file=sys.stderr)
#     sys.exit(1)

def find_bam2prof():
    """
    Locate the bam2prof binary.
    
    1. Try to locate it using PATH (e.g. after installation).
    2. Check the directory of the current script.
    3. Check the src/ subdirectory, which is useful for development (e.g., when the repo is cloned).
    """
    # Check if bam2prof is in the system PATH.
    # path_bin = shutil.which("bam2prof")
    # if path_bin:
    #     return Path(path_bin)
    
    # Get the directory of the current script.
    base_dir = Path(__file__).parent.resolve()
    
    # List possible relative locations.
    possible_locations = [
        base_dir / "bam2prof",        # In the same directory as the script
        base_dir / "submodules" / "src" / "bam2prof"   # In the src subdirectory (common for cloned repos)
    ]
    
    for location in possible_locations:
        if location.exists():
            logger.info('%s %s', "bam2prof located at:", location)
            return location

    logging.error("bam2prof binary not found!")
    sys.exit(1)

def run_bam2prof(args_list, hpc=None, b2p=None):
    """
    Run the compiled `bam2prof` binary with the given arguments.
    If an HPC file is provided via the hpc parameter, write the command to that file
    instead of executing it.
    """
    if b2p is not None:
        bam2prof_path = b2p
    else:
        bam2prof_path = find_bam2prof()

    #bam2prof_path.chmod(0o755)

    command = [str(bam2prof_path)] + args_list
    command_str = ' '.join(command)

    if hpc is not None:
        with open(hpc, 'a') as f:
            f.write(command_str + '\n')
        return
    logger.info(f"bam2prof command: '{command_str}'")
    result = subprocess.run(command, capture_output=True, text=True)

    # if result.stdout.strip():
    #     logger.info(result.stdout)
    # if result.stderr.strip():
    #     logger.error(result.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="Python wrapper for bam2prof")

    # General options
    parser.add_argument('bam_files', nargs='?', help='File with paths to BAM files; one per line.')
    parser.add_argument('-minq', type=int, default=0, help='Minimum base quality (default: %(default)s)')
    parser.add_argument('-minl', type=int, default=35, help='Minimum fragment/read length (default: %(default)s)')
    parser.add_argument('-endo', type=int, default=0, help='Endo flag (default: %(default)s)')
    parser.add_argument('-length', type=int, default=5, help='Length (default: %(default)s)')
    parser.add_argument('-err', type=float, default=0, help='Error rate (default: %(default)s)')
    parser.add_argument('-log', action='store_true', help='Logarithmic scale (default: %(default)s)')
    parser.add_argument('-bed', help='BED file for positions')
    parser.add_argument('-mask', help='BED file for mask positions')
    parser.add_argument('-paired', action='store_true', help='Allow paired reads (default: %(default)s)')
    parser.add_argument('-meta', action='store_true', help='One Profile per unique reference (default: %(default)s)')
    parser.add_argument('-classic', action='store_true', help='One Profile per bam file (default: %(default)s)')
    parser.add_argument('-precision', type=float, default=0, help='Set minimum decimal precision for substitution frequency computation (default: %(default)s [= all alignments used]). Increase speed by setting to either (from faster to slower): 0.01, 0.001, ... ) ')
    parser.add_argument('-minAligned', type=float, default=10000000, help='Number of aligned sequences after which substitution patterns are checked if frequencies converge (default: %(default)s)')
    parser.add_argument('-ref-id', help='Specify reference ID')
    parser.add_argument('-single', action='store_true', help='Single strand library (default: %(default)s)')
    parser.add_argument('-double', action='store_true', help='Double strand library (default: %(default)s)')
    parser.add_argument('-both', action='store_true', help='Report both C->T and G->A (default: %(default)s)')
    parser.add_argument('-o', help='Output directory')
    parser.add_argument('-dp', action='store_true', help='Output in damage-patterns format (default: %(default)s)')
    parser.add_argument('-q', type=int, choices=[0, 1], default=1, help='Do not print why reads are skipped (default: %(default)s)')
    parser.add_argument('-threads', type=int, default=1, help='Number of threads. One file per thread (default: %(default)s)')
    parser.add_argument('-hpc', help='Dry Run. Pipe execution commands to file')
    parser.add_argument('-bam2profpath', help='Provide absolute path to bam2prof binary and use it for execution')

    args = parser.parse_args()

    # Prepare the arguments for bam2prof command
    base_args = []
    if args.minq:
        base_args.extend(['-minq', str(args.minq)])
    if args.minl:
        base_args.extend(['-minl', str(args.minl)])
    if args.endo:
        base_args.extend(['-endo', str(args.endo)])
    if args.length:
        base_args.extend(['-length', str(args.length)])
    if args.err:
        base_args.extend(['-err', str(args.err)])
    if args.log:
        base_args.append('-log')
    if args.bed:
        base_args.extend(['-bed', args.bed])
    if args.mask:
        base_args.extend(['-mask', args.mask])
    if args.precision:
        base_args.extend(['-precision', str(args.precision)])
    if args.minAligned:
        base_args.extend(['-minAligned', str(args.minAligned)])
    if args.paired:
        base_args.append('-paired')
    if args.ref_id:
        base_args.extend(['-ref-id', args.ref_id])
    if args.single:
        base_args.append('-single')
    if args.double:
        base_args.append('-double')
    if args.both:
        base_args.append('-both')
    if args.o:
        base_args.extend(['-o', args.o])
    if args.dp:
        base_args.append('-dp')
    if args.q:  # Ensure that q is provided, even though it has a default
        base_args.extend(['-q', str(args.q)])  # Append '-q' and its value (0 or 1) as a string

    hpc_file = args.hpc
    bam2prof_path = args.bam2profpath

    if not args.bam_files:
        logger.error("Error: BAM file list is required.")
        return
    
    with open(args.bam_files) as f:
        bam_files = f.read().splitlines()

    # Sanity check: Output directory must be specified
    if not args.o:
        logger.error("Error: Output Directory must be specified with < -o >")
        return

    # Check if output directory exists and is not empty
    if os.path.exists(args.o) and os.listdir(args.o):
        logger.warning(f"Warning: Output directory '{args.o}' is not empty.")

    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        if args.classic:
            logger.info("Classic Mode: Will produce one matrix per bam file.")
            base_args.append('-classic')
        elif args.meta:
            logger.info("Metagenome Mode: Will produce a matrix per reference.")
            base_args.append('-meta')
        else:
            logger.error("Error: -meta or -classic needs to be specified")
            return

        missing = [bam for bam in bam_files if not os.path.exists(bam)]
        if missing:
            for bam in missing:
                logger.error(f"Required BAM not found: {bam}")
            sys.exit(1)

        futures = {
            executor.submit(run_bam2prof, base_args + [bam], hpc=hpc_file, b2p=bam2prof_path): bam
            for bam in bam_files
        }
        for future in as_completed(futures):
            bam_file = futures[future]
            try:
                _ = future.result()
            except Exception as exc:
                logger.error(f"{bam_file} generated an exception: {exc}")

if __name__ == "__main__":
    main()

