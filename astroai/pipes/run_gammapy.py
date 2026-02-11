# *******************************************************************************
# Copyright (C) 2024 Ambra Di Piano
#
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *******************************************************************************

import warnings
import argparse
import pandas as pd
import numpy as np
from os import makedirs
from os.path import join, dirname
from astroai.tools.utils import load_yaml_conf, get_irf_name, select_random_irf
from astroai.tools.ganalysis import GAnalysis

with warnings.catch_warnings():
    warnings.filterwarnings('error')

def get_snr(excess, bkg):
    snr = excess/np.sqrt(excess+bkg)
    return snr

def run_gammapy_pipeline(conf, dl3_file, target_name, target_dict):
    ganalysis = GAnalysis()
    ganalysis.set_conf(conf)
    ganalysis.set_eventfilename(dl3_file)
    # get reducedirf or make it if missing
    try:
        ganalysis.set_reducedirfs(conf['execute']['reducedirfdir'], seed=conf['simulation']['id'])
    except AssertionError as e:
        ganalysis.execute_dl3_dl4_reduction()
    # read dataset
    dataset = ganalysis.read_dataset()
    stats, candidate = ganalysis.run_gammapy_analysis_pipeline(dataset, target_name, target_dict)
    return stats, candidate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    # get configuration and infodata
    conf = load_yaml_conf(args.configuration)
    infodata = pd.read_csv(join(dirname(conf['simulation']['directory']), conf['simulation']['datfile']), sep=' ', header=0).sort_values(by=['seed'])

    # write results
    makedirs(conf['execute']['outdir'], exist_ok=True)
    results = open(join(conf['execute']['outdir'], conf['execute']['outfile']), 'w+')
    results.write('seed loc_ra loc_dec offset counts_on counts_off alpha excess excess_err sigma snr aeff\n')

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
        if '/data/cta' not in conf['simulation']['caldb_path']:
            conf['simulation']['caldb_path'] += '/data/cta'
        if conf['simulation']['irf'] == 'random':
            conf['simulation']['irf'] = select_random_irf(caldb_path=conf['simulation']['caldb_path'], prod=conf['simulation']['caldb'])
        else:
            conf['simulation']['irf'] = get_irf_name(irf=row['irf'].values[0], caldb_path=join(conf['simulation']['caldb_path'], conf['simulation']['caldb']))

        # setup coordinates
        true = {'ra': row['source_ra'].values[0], 'dec': row['source_dec'].values[0], 'rad': conf['photometry']['onoff_radius']}
        candidate_init = {'ra': None, 'dec': None, 'rad': conf['photometry']['onoff_radius']}

        # run pipeline
        stats, candidate = run_gammapy_pipeline(conf=conf, dl3_file=dl3, target_name=f"crab_{seed:05d}", target_dict=candidate_init)

        try:
            snr = get_snr(excess=stats['excess'], bkg=stats['counts_off'])
        except:
            snr = np.nan

        results.write(f"{seed} {candidate['ra']} {candidate['dec']} {stats['offset']} {stats['counts']} {stats['counts_off']} {stats['alpha']} {stats['excess']} {stats['excess_error']} {stats['sigma']} {snr} {stats['aeff_mean']}\n")

    results.close()

