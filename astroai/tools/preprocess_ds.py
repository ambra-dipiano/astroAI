# *****************************************************************************
# Copyright (C) 2023 INAF
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import argparse
from os.path import join
from astroai.tools.utils import load_yaml_conf, process_dataset

def main(configuration):
    conf = load_yaml_conf(configuration)
    conf = conf['preprocess']
    if 'detect' in conf['mode'] or 'class' in conf['mode']:
        src_dataset_path = join(conf['directory'], 'crab', 'sim')
        bkg_dataset_path = join(conf['directory'], 'background', 'sim')
    elif 'clean' in conf['mode']:
        src_dataset_path = join(conf['directory'], 'crab_only', 'sim')
        bkg_dataset_path = join(conf['directory'], 'crab', 'sim')
    else:
        raise ValueError(f"Mode {conf['mode']} not valid")
    trange = [conf['time_start'], conf['time_stop']]
    # create image dataset
    ds = process_dataset(src_dataset_path, bkg_dataset_path, trange=trange, smoothing=conf['smoothing'], binning=conf['binning'], sample=conf['sample'], save=True, output=conf['directory'], min_max_norm=conf['min_max_norm'], mode=conf['mode'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    main(args.configuration)