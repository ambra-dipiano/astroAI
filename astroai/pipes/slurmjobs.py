# *******************************************************************************
# Copyright (C) 2023 Ambra Di Piano
#
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *******************************************************************************

import argparse
from os import system, makedirs
from os.path import abspath, join, expandvars, dirname

def main(pipe, filename):
    job_name = f'{pipe}'
    pipe = f'{pipe}'
    # write bash
    outdir = join(dirname(abspath(__file__)), 'slurms')
    sh_outname = join(outdir, f'{job_name}.sh')
    makedirs(outdir, exist_ok=True)

    with open(sh_outname, 'w+') as f:
        f. write("#!/bin/bash\n")
        f.write(f"\mamba activate astroai")
        f.write(f"\n\tpython {join(dirname(abspath(__file__)), pipe)}.py -f {filename}\n")

    # write job
    job_outname = join(outdir, f'{job_name}.ll')
    job_outlog = join(outdir, f'{job_name}.log')
    with open(job_outname, 'w+') as f:
        f.write('#!/bin/bash')
        f.write(f'\n\n#SBATCH --job-name={job_name}')
        f.write(f'\n#SBATCH --output={job_outlog}')
        f.write('\n#SBATCH --account=dipiano')
        f.write('\n#SBATCH --partition=large')
        f.write(f'\n\nexec sh {str(sh_outname)}\n')

    # sbatch job
    system(f'sbatch {job_outname}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipe', type=str, required=True, choices=['gammapy', 'cnn'], help='Pipeline to submit')
    parser.add_argument('-f', '--filename', type=str, required=True, help='Configuration YAML file')
    args = parser.parse_args()

    main(f'run_{args.pipe}', args.filename)


