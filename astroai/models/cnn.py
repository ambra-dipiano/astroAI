# *****************************************************************************
# Copyright (C) 2023 INAF
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from os.path import join, isfile
from astroai.tools.utils import load_yaml_conf, split_dataset
TF_CPP_MIN_LOG_LEVEL="1"

def create_cnn_binary_classifier(binning):
    model = tf.keras.models.Sequential()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(2, (5, 5), activation='relu', input_shape=(binning, binning, 1), name='conv2d_1'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), name='maxpool2d_1'))
    model.add(tf.keras.layers.Conv2D(2, (5, 5), activation='relu', name='conv2d_2'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), name='maxpool2d_2'))
    model.add(tf.keras.layers.Conv2D(2, (5, 5), activation='relu', name='conv2d_3'))
    model.add(tf.keras.layers.Dropout(0.2, name='drop_1'))
    model.add(tf.keras.layers.Flatten(name='flat_1'))
    model.add(tf.keras.layers.Dense(10, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dropout(0.2, name='drop_2'))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid', name='dense_2'))
    model.summary()
    return model

def compile_and_fit_binary_classifier(model, train_ds, train_lb, test_ds, test_lb, batch_sz=32, epochs=25, learning=0.001, shuffle=True, logdate=True):
    logdir = join("logs", "cnn-classify") 
    if logdate:
        logdir += datetime.now().strftime("%Y%m%dT%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning), 
              loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(train_ds, train_lb, batch_size=batch_sz, epochs=epochs, validation_data=(test_ds, test_lb), 
            callbacks=[tensorboard_callback], shuffle=shuffle)
    return history

def main(configuration, mode):
    conf = load_yaml_conf(configuration)
    exposure = conf['preprocess']['time_stop'] - conf['preprocess']['time_start']
    sample = conf['preprocess']['sample']
    smoothing = conf['preprocess']['smoothing']
    filename = join(conf['preprocess']['directory'], f"{mode}_{exposure}s_{smoothing}sgm_{sample}sz.npy") 
    if isfile(filename):
        ds = np.load(filename, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
    else:
        raise FileNotFoundError(filename)

    # binary classification network
    if ('detect' or 'class') in mode:
        # split dataset
        train_data, train_labels, test_data, test_labels = split_dataset(ds, split=conf['detection']['split'], reshape=conf['detection']['reshape'], binning=conf['preprocess']['binning'])
        # create model
        model = create_cnn_binary_classifier(binning=conf['preprocess']['binning'])
        # compile and fit
        history = compile_and_fit_binary_classifier(model=model, train_ds=train_data, train_lb=train_labels, test_ds=test_data, test_lb=test_labels, batch_sz=conf['detection']['batch_sz'], epochs=conf['detection']['epochs'], shuffle=conf['detection']['shuffle'], learning=conf['detection']['learning'])

    else:
        raise ValueError('Not implemented yet')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['classify', 'cleaning', 'detection', 'localisation'], help="scope of the CNN and thus related network")
    args = parser.parse_args()

    main(args.configuration, args.mode)