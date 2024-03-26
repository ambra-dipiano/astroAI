# *****************************************************************************
# Copyright (C) 2023 Ambra Di Piano
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import pickle
import argparse
import numpy as np
import tensorflow as tf
from os.path import join, abspath, dirname
from astroai.tools.utils import load_yaml_conf

def clean_regressor_dataset(ds_dataset_path, model, save=True):
    # get cleaner model
    print(join(dirname(abspath(__file__)).replace('tools', 'models/cnn_cleaner'), f'{model}.keras'))
    model = tf.keras.models.load_model(join(dirname(abspath(__file__)).replace('tools', 'models/cnn_cleaner'), f'{model}.keras'))
    print(f"CNN-CLEANER: {model}")
    print(f"DS: {ds_dataset_path}")
    # load noisy DS
    if '.pickle' in ds_dataset_path:
        with open(ds_dataset_path,'rb') as f: ds = pickle.load(f)
    elif '.npy' in ds_dataset_path:
        ds = np.load(ds_dataset_path, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
    print(f"COLS: {ds.keys()}")
    print(f"SIZE: {len(ds['DS'])}")
    # replace noisy with cleaned DS
    ds['DS'] = model.predict(ds['DS'])
    if save:
        if '.npy' in ds_dataset_path:
            filename = ds_dataset_path.replace('.npy', '_CLEAN.npy')
            np.save(filename, ds, allow_pickle=True, fix_imports=True)
        elif '.pickle' in ds_dataset_path:
            filename = ds_dataset_path.replace('.pickle', '_CLEAN.pickle')
            with open(filename,'wb') as f: pickle.dump(ds, f, protocol=4)
        else: 
            raise Exception('Dataset must be NPY or PICKLE')
        print(f"Process complete: {filename}")

def main(configuration, model):
    conf = load_yaml_conf(configuration)
    ds_dataset_path = join(conf['preprocess']['directory'], conf['preprocess']['saveas'])
    clean_regressor_dataset(ds_dataset_path, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    parser.add_argument('-m', '--model', type=str, required=True, help='CNN-cleaner model to use for background subtraction')
    args = parser.parse_args()

    main(args.configuration, args.model)