# *******************************************************************************
# Copyright (C) 2024 Ambra Di Piano
#
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *******************************************************************************

import argparse
import pandas as pd
from os import makedirs
from os.path import join, dirname
from astroai.tools.utils import load_yaml_conf
import tensorflow as tf

def run_cnn_pipeline(dl3, binning, cleaner, regressor):
    heatmap = 'preprocess map from dl3'
    prediction = cleaner.predict(heatmap)
    candidate = regressor.predict(heatmap) * binning
    return prediction, candidate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    parser.add_argument('-z', '--zenith', type=str, required=True, choices=['20', 'ALL'], help="zenith angle of choice")
    args = parser.parse_args()

    # get configuration and infodata
    conf = load_yaml_conf(args.configuration)
    infodata = pd.read_csv(join(dirname(conf['simulation']['directory']), conf['simulation']['datfile']), sep=' ', header=0).sort_values(by=['seed'])

    # write results
    makedirs(conf['execute']['outdir'], exist_ok=True)
    results = open(join(conf['execute']['outdir'], conf['execute']['outfile']), 'w+')
    results.write('seed loc_ra loc_dec counts_on counts_off excess excess_err sigma\n')

    # cicle every seed in samples
    for i in range(conf['samples']):
        # get seed
        seed = i + 1 + conf['start_seed']
        conf['simulation']['id'] = seed

        # get observation info
        row = infodata[infodata['seed']==seed]
        dl3 = join(conf['simulation']['directory'], f'crab_{seed:05d}.fits')
        conf['simulation']['point_ra'] = row['point_ra'].values[0]
        conf['simulation']['point_dec'] = row['point_dec'].values[0]

        # load models
        cleaner = tf.keras.models.load_model(f'../models/crta_models/cleaner_z{args.zenith}.keras')
        regressor = tf.keras.models.load_model(f'../models/crta_models/regressor_z{args.zenith}.keras')

        # run pipeline
        prediction, candidate = run_cnn_pipeline(conf=conf, dl3=dl3, cleaner=cleaner, rergessor=regressor)
        results.write(f"{seed} {candidate['ra']} {candidate['dec']}\n")

    results.close()