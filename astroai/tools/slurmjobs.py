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
from os.path import abspath, join, dirname

def main(script, filename, env):
    job_name = f'{script}'
    script = f'{script}'
    slurmpath = join(dirname(abspath(__file__)), 'slurms')
    makedirs(slurmpath, exist_ok=True)

    # write bash
    sh_outname = join(slurmpath, f'{job_name}.sh')
    with open(sh_outname, 'w+') as f:
        f. write("#!/bin/bash\n")
        if env == 'venv':
            f.write(f"\nsource /home/dipiano/venvs/astroai/bin/activate")
        elif env == 'conda':
            f.write(f"\nconda activate astroai")
        elif env == 'mamba':
            f.write(f"\nmamba activate astroai")
        f.write(f"\n\tpython {join(dirname(abspath(__file__)), script)}.py -f {filename}\n")

    # write job
    job_outname = join(slurmpath, f'{job_name}.ll')
    job_outlog = join(slurmpath, f'{job_name}.log')
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
    parser.add_argument('-s', '--script', type=str, required=True, choices=['preprocess_ds', 'normalise_ds', 'load_and_normalise_maps', 'preclean_ds'], help='Script to submit')
    parser.add_argument('-f', '--filename', type=str, required=True, help='Configuration YAML file')
    parser.add_argument('-e', '--env', type=str, default='venv', choices=['venv', 'conda', 'mamba'], help='Virtual environtmet package')
    args = parser.parse_args()

    main(args.script, args.filename, args.env)


