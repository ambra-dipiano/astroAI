# *******************************************************************************
# Copyright (C) 2024 Ambra Di Piano
#
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *******************************************************************************

import argparse
import os
from time import perf_counter
import pandas as pd
import numpy as np
from os import makedirs
from os.path import join, dirname, isfile
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astroai.tools.utils import load_yaml_conf, set_wcs, create_circular_mask, extract_heatmap_from_table, normalise_heatmap, normalise_dataset, stretch_smooth, stretch_min_max

# force cpu run and reduce tensorflow runtime logs
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

def compute_counts_map(dl3, conf):
    # SECTION 1 - read events from dl3 and extract dl4 heatmap
    heatmap = Table.read(dl3, hdu=1).to_pandas()
    heatmap = extract_heatmap_from_table(data=heatmap, trange=[conf['preprocess']['time_start'], conf['preprocess']['time_stop']], smoothing=conf['preprocess']['smoothing'], nbins=conf['preprocess']['binning'], filter=True)
    return heatmap

def prepare_heatmap(heatmap, conf):
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
    timing = {}

    # Step 1a - counts map (heatmap) computation
    t0 = perf_counter()
    heatmap = compute_counts_map(dl3=dl3, conf=conf)
    timing['t_counts_map'] = perf_counter() - t0

    # Step 1b - preparation of counts map for cnn
    t0 = perf_counter()
    heatmap = prepare_heatmap(heatmap=heatmap, conf=conf)
    timing['t_prepare'] = perf_counter() - t0

    # keep aggregate preprocess timing for compatibility
    timing['t_preprocess'] = timing['t_counts_map'] + timing['t_prepare']

    # Step 2 - Apply CNN-cleaner (aka prepare clean map)
    t0 = perf_counter()
    prediction = cleaner.predict(heatmap)
    timing['t_cleaner'] = perf_counter() - t0

    # Step 3 - Apply CNN-regressor (aka blindsearch)
    t0 = perf_counter()
    candidate = regressor.predict(prediction) * conf['preprocess']['binning']
    timing['t_regressor'] = perf_counter() - t0
    return heatmap, prediction, candidate, timing

def get_cleaner_metrics(heatmap, prediction, row, conf):
    # compute cleaner metrics on full map and on source region
    noisy_map = heatmap[0, :, :, 0]
    clean_map = prediction[0, :, :, 0]
    residual_map = noisy_map - clean_map

    # full map metrics
    sum_cleaned = np.sum(clean_map)
    sum_residual = np.sum(residual_map)

    # source region metrics
    binning = conf['preprocess']['binning']
    pixelsize = (2 * row['fov'].values[0]) / binning
    point_ref = (binning / 2) + (pixelsize / 2)
    w = set_wcs(point_ra=row['point_ra'].values[0], point_dec=row['point_dec'].values[0], point_ref=point_ref, pixelsize=pixelsize)
    source = SkyCoord(row['source_ra'].values[0], row['source_dec'].values[0], unit='deg', frame='icrs')
    x, y = w.world_to_pixel(source)
    radius_deg = conf['photometry']['onoff_radius'] if 'photometry' in conf and 'onoff_radius' in conf['photometry'] else 0.2
    radius_pix = radius_deg / pixelsize
    h, w = clean_map.shape
    mask = create_circular_mask(h, w, center=(y, x), radius=radius_pix)

    sum_on_cleaned = np.sum(clean_map * mask)
    sum_on_residual = np.sum(residual_map * mask)
    return sum_cleaned, sum_residual, sum_on_cleaned, sum_on_residual

def get_candidate_from_regressor(candidate, row, binning):
    # decode candidate from regressor output
    x_pix, y_pix = candidate[0][0], candidate[0][1]
    if np.isnan(x_pix) or np.isnan(y_pix):
        return np.nan, np.nan, np.nan, np.nan

    # convert pixel candidate to sky coordinates
    pixelsize = (2 * row['fov'].values[0]) / binning
    point_ref = (binning / 2) + (pixelsize / 2)
    w = set_wcs(point_ra=row['point_ra'].values[0], point_dec=row['point_dec'].values[0], point_ref=point_ref, pixelsize=pixelsize)
    sky = w.pixel_to_world(x_pix, y_pix)
    return sky.ra.deg, sky.dec.deg, x_pix, y_pix

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
    benchmark = conf['benchmark'] if 'benchmark' in conf else {'enabled': False}
    benchmark_enabled = benchmark['enabled']
    infodata = pd.read_csv(join(dirname(conf['simulation']['directory']), conf['simulation']['datfile']), sep=' ', header=0).sort_values(by=['seed'])

    # write results
    makedirs(conf['execute']['outdir'], exist_ok=True)
    results = open(join(conf['execute']['outdir'], conf['execute']['outfile']), 'w+')
    if benchmark_enabled:
        results.write('seed loc_ra loc_dec loc_x loc_y clean_sum residual_sum on_clean_sum on_residual_sum t_model_load t_counts_map t_prepare t_preprocess t_cleaner t_regressor t_decode t_cleaner_metrics t_total\n')
    else:
        results.write('seed loc_ra loc_dec loc_x loc_y clean_sum residual_sum on_clean_sum on_residual_sum\n')

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
    t0 = perf_counter()
    cleaner = tf.keras.models.load_model(cleaner_model)
    regressor = tf.keras.models.load_model(regressor_model)
    t_model_load = perf_counter() - t0

    # cicle every seed in samples
    for i in range(conf['samples']):
        # get seed
        seed = i + 1 + conf['start_seed']
        conf['simulation']['id'] = seed

        # get observation info
        row = infodata[infodata['seed']==seed]
        if row.empty:
            raise ValueError(f"Seed {seed} not found in info table")
        dl3 = join(conf['simulation']['directory'], f'crab_{seed:05d}.fits')
        conf['simulation']['point_ra'] = row['point_ra'].values[0]
        conf['simulation']['point_dec'] = row['point_dec'].values[0]

        # run pipeline
        t_start = perf_counter()
        heatmap, prediction, candidate, timing = run_cnn_pipeline(dl3=dl3, conf=conf, cleaner=cleaner, regressor=regressor)
        t0 = perf_counter()
        loc_ra, loc_dec, loc_x, loc_y = get_candidate_from_regressor(candidate=candidate, row=row, binning=conf['preprocess']['binning'])
        timing['t_decode'] = perf_counter() - t0
        t0 = perf_counter()
        clean_sum, residual_sum, on_clean_sum, on_residual_sum = get_cleaner_metrics(heatmap=heatmap, prediction=prediction, row=row, conf=conf)
        timing['t_cleaner_metrics'] = perf_counter() - t0
        timing['t_total'] = perf_counter() - t_start
        if benchmark_enabled:
            results.write(f"{seed} {loc_ra} {loc_dec} {loc_x} {loc_y} {clean_sum} {residual_sum} {on_clean_sum} {on_residual_sum} {t_model_load} {timing['t_counts_map']} {timing['t_prepare']} {timing['t_preprocess']} {timing['t_cleaner']} {timing['t_regressor']} {timing['t_decode']} {timing['t_cleaner_metrics']} {timing['t_total']}\n")
        else:
            results.write(f"{seed} {loc_ra} {loc_dec} {loc_x} {loc_y} {clean_sum} {residual_sum} {on_clean_sum} {on_residual_sum}\n")

    results.close()