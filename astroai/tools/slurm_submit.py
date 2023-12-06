# *******************************************************************************
# Copyright (C) 2023 INAF
#
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *******************************************************************************

import argparse
from os import system
from os.path import abspath, join, expandvars, dirname

def main(script, filename):
    job_name = f'{script}_train'
    script = f'{script}'
    # write bash
    sh_outname = join(dirname(abspath(__file__)), 'slurms', f'{job_name}.sh')
    with open(sh_outname, 'w+') as f:
        f. write("#!/bin/bash\n")
        f.write(f"\nsource {join(expandvars('$HOME'), 'venvs/astroai/bin/activate')}")
        f.write(f"\n\tpython {join(dirname(abspath(__file__)), script)}.py -f {filename}\n")

    # write job
    job_outname = join(dirname(abspath(__file__)), 'slurms', f'{job_name}.ll')
    job_outlog = join(dirname(abspath(__file__)), 'slurms', f'{job_name}.log')
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
    parser.add_argument('-s', '--script', type=str, required=True, choices=['cnn'], help='Script to submit')
    parser.add_argument('-f', '--filename', type=str, required=True, help='Configuration YAML file')
    args = parser.parse_args()

    main(args.script, args.filename)


