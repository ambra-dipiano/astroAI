# *****************************************************************************
# Copyright (C) 2023 Ambra Di Piano
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import argparse
from os import system
from os.path import join, dirname
from astroai.tools.utils import load_yaml_conf, process_dataset, process_regressor_dataset

def main(configuration):
    conf = load_yaml_conf(configuration)
    conf = conf['preprocess']
    trange = [conf['time_start'], conf['time_stop']]

    # CLASSIFICATION
    if 'detect' in conf['mode'] or 'class' in conf['mode']:
        ds1_dataset_path = join(conf['directory'], 'crab', 'sim')
        ds2_dataset_path = join(conf['directory'], 'background', 'sim')
        ds = process_dataset(ds1_dataset_path, ds2_dataset_path, trange=trange, smoothing=conf['smoothing'], binning=conf['binning'], sample=conf['sample'], save=True, output=conf['directory'], norm_value=conf['norm_value'], stretch=conf['stretch'], saveas=conf['saveas'])

    # CLEANING
    elif 'clean' in conf['mode']:
        ds1_dataset_path = join(conf['directory'], 'noisy')
        ds2_dataset_path = join(conf['directory'], 'clean')
        ds = process_dataset(ds1_dataset_path, ds2_dataset_path, trange=trange, smoothing=conf['smoothing'], binning=conf['binning'], sample=conf['sample'], save=True, output=join(conf['directory'], '..'), norm_value=conf['norm_value'], stretch=conf['stretch'], saveas=conf['saveas'])

    # REGRESSION
    elif 'loc' in conf['mode'] or 'regress' in conf['mode']:
        ds_dataset_path = join(conf['directory'])
        ds = process_regressor_dataset(ds_dataset_path, infotable=join(conf['directory'], conf['infotable']), smoothing=conf['smoothing'], binning=conf['binning'], sample=conf['sample'], save=True, output=dirname(conf['directory']), stretch=conf['stretch'], norm_value=conf['norm_value'], saveas=conf['saveas'], exposure=conf['time_stop'])
    else:
        raise ValueError(f"Mode {conf['mode']} not valid")

  

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    main(args.configuration)