# *****************************************************************************
# Copyright (C) 2023 INAF
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import argparse
from os.path import join
from astroai.tools.utils import load_yaml_conf, get_and_normalise_dataset

def main(configuration):
    conf = load_yaml_conf(configuration)
    conf = conf['preprocess']
    ds_path = join(conf['directory'])
    # create image dataset
    ds = get_and_normalise_dataset(ds_path=ds_path, sample=conf['sample'], save=True, output=conf['directory'], norm_value=[conf['preprocess']['min_norm'], conf['preprocess']['max_norm']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    main(args.configuration)