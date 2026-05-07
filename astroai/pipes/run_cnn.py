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
import numpy as np
from os import makedirs
from os.path import join, dirname, isfile
from astropy.table import Table
from astroai.tools.utils import load_yaml_conf, extract_heatmap_from_table, normalise_heatmap, normalise_dataset, stretch_smooth, stretch_min_max
import tensorflow as tf

def preprocess_dl3_heatmap(dl3, conf):
    # SECTION 1 - read events from dl3 and extract dl4 heatmap 
    heatmap = Table.read(dl3, hdu=1).to_pandas()
    heatmap = extract_heatmap_from_table(data=heatmap, trange=[conf['preprocess']['time_start'], conf['preprocess']['time_stop']], smoothing=conf['preprocess']['smoothing'], nbins=conf['preprocess']['binning'], filter=True)

    # SECTION 2 - normalise map according to preprocessing config
    norm_value = conf['preprocess']['norm_value']
    if norm_value == 1 and conf['preprocess']['stretch']:
        heatmap = stretch_smooth(heatmap, conf['preprocess']['smoothing'])
    elif norm_value == 1 and not conf['preprocess']['stretch']:
        heatmap = normalise_heatmap(heatmap)
    elif type(norm_value) == float and conf['preprocess']['stretch']:
        heatmap = stretch_min_max(heatmap, vmax=norm_value)
    elif type(norm_value) == float and not conf['preprocess']['stretch']:
        heatmap = normalise_dataset(heatmap, max_value=norm_value)

    # SECTION 3 - reshape as keras input tensor
    binning = conf['preprocess']['binning']
    if heatmap.shape != (binning, binning):
        heatmap = heatmap.reshape(binning, binning)
    heatmap = np.array(heatmap).reshape(1, binning, binning, 1)
    return heatmap

def run_cnn_pipeline(dl3, conf, cleaner, regressor):
    # Step 1 - Preprocess data (aka make map)
    heatmap = preprocess_dl3_heatmap(dl3=dl3, conf=conf)
    # Step 2 - Apply CNN-cleaner (aka prepare clean map)
    prediction = cleaner.predict(heatmap)
    # Step 3 - Apply CNN-regressor (aka blindsearch)
    candidate = regressor.predict(prediction) * conf['preprocess']['binning']
    return prediction, candidate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    # get configuration and infodata
    conf = load_yaml_conf(args.configuration)
    if 'cnn_inference' not in conf:
        raise KeyError("Missing 'cnn_inference' section in configuration file")
    if 'cleaner_model' not in conf['cnn_inference']:
        raise KeyError("Missing 'cnn_inference.cleaner_model' in configuration file")
    if 'regressor_model' not in conf['cnn_inference']:
        raise KeyError("Missing 'cnn_inference.regressor_model' in configuration file")
    infodata = pd.read_csv(join(dirname(conf['simulation']['directory']), conf['simulation']['datfile']), sep=' ', header=0).sort_values(by=['seed'])

    # write results
    makedirs(conf['execute']['outdir'], exist_ok=True)
    results = open(join(conf['execute']['outdir'], conf['execute']['outfile']), 'w+')
    results.write('seed loc_ra loc_dec counts_on counts_off excess excess_err sigma irf\n')

    # load models from inference configuration
    cleaner_model = conf['cnn_inference']['cleaner_model']
    regressor_model = conf['cnn_inference']['regressor_model']
    if not isfile(cleaner_model):
        cleaner_model = join(dirname(__file__), '../models/crta_models', cleaner_model)
    if not isfile(regressor_model):
        regressor_model = join(dirname(__file__), '../models/crta_models', regressor_model)
    if not isfile(cleaner_model):
        raise FileNotFoundError(f"Cleaner model not found: {cleaner_model}")
    if not isfile(regressor_model):
        raise FileNotFoundError(f"Regressor model not found: {regressor_model}")
    cleaner = tf.keras.models.load_model(cleaner_model)
    regressor = tf.keras.models.load_model(regressor_model)

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

        # run pipeline
        prediction, candidate = run_cnn_pipeline(dl3=dl3, conf=conf, cleaner=cleaner, regressor=regressor)
        results.write(f"{seed} {candidate['ra']} {candidate['dec']}\n")

    results.close()