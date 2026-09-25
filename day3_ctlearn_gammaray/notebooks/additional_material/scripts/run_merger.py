#!/usr/bin/env python

import argparse
import logging
import glob
import os
import re
import time
import subprocess as sp
import numpy as np

def main():

    parser = argparse.ArgumentParser(
        description=("Script to run ctapipe-merge tool with DL1 hdf5 files"))
    parser.add_argument('--input_dir', '-i',
                        help='input directory',
                        default="./")
    parser.add_argument('--pattern', '-p',
                        help='pattern to mask unwanted files',
                        default="*.h5")
    parser.add_argument('--num_outputfiles', '-n',
                        help='number of output files',
                        default=2,
                        type=int)
    parser.add_argument('--output_dir', '-o',
                        help='output directory',
                        default="./")
    parser.add_argument('--log_level', '-l',
                        help='Log level passed to the ctapipe tool, useful valid options: DEBUG, INFO, WARN, ERROR, CRITICAL',
                        default="INFO")

    args = parser.parse_args()

    # Input handling
    abs_file_dir = os.path.abspath(args.input_dir)
    input = np.sort(glob.glob(os.path.join(abs_file_dir, args.pattern)))

    run_numbers, types = [], []
    for file in input:
        filename = file.split("/")[-1]
        run_numbers.append([int(s) for s in re.findall(r'\d+', filename)][-4])
        types.append(filename.split("_")[0])
    run_numbers = np.sort(run_numbers)
    if len(np.unique(types)) == 1:
        type = np.unique(types)[0]
    else:
        raise ValueError(f"Only merge files of the same particle type. Your inputs:'{types}'")

    for runs in np.array_split(run_numbers, args.num_outputfiles):
        output_file = f"{args.output_dir}/{type}_theta_16.087_az_108.090_runs{runs[0]}-{runs[-1]}.r1.dl1.h5"
        print(f"Outputfile: '{output_file}'")
        cmd = [
             f"ctapipe-merge",
             "--overwrite",
             "--MergeTool.skip_broken_files=True",
             f"--output={output_file}",
             f"--log-level={args.log_level}",
            ]
        for run in runs:
            cmd.append(f"{abs_file_dir}/{type}_theta_16.087_az_108.090_run{run}.r1.dl1.h5")
        sp.run(cmd)

    return

if __name__ == "__main__":
    main()

