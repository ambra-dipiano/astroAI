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
from datetime import datetime

def main(architecture, filename, mode, env):
    job_name = f'{architecture}_{mode}_{datetime.now().strftime("%Y%m%dT%H%M%S")}'
    # write bash
    makedirs(join(dirname(abspath(__file__)), 'slurms'), exist_ok=True)
    sh_outname = join(dirname(abspath(__file__)), 'slurms', f'{job_name}.sh')
    with open(sh_outname, 'w+') as f:
        f. write("#!/bin/bash\n")
        if env == 'venv':
            f.write(f"\nsource /home/dipiano/venvs/astroai/bin/activate")
        elif env == 'conda':
            f.write(f"\nconda activate astroai")
        elif env == 'mamba':
            f.write(f"\nmamba activate astroai")
        f.write(f"\n\tpython {join(dirname(abspath(__file__)), architecture)}.py -f {filename}\n")

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
    parser.add_argument('-a', '--architecture', type=str, required=True, choices=['cnn'], help='Architecture of the model')
    parser.add_argument('-f', '--filename', type=str, required=True, help='Configuration YAML file')
    parser.add_argument('-m', '--mode', type=str, required=False, choices=['clean', 'classify', 'detect', 'loc', 'regression'], help='Scope of the model')
    parser.add_argument('-e', '--env', type=str, default='venv', choices=['venv', 'conda', 'mamba'], help='Virtual environtmet package')
    args = parser.parse_args()

    main(args.architecture, args.filename, args.mode, args.env)


