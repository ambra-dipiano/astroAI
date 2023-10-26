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
from os.path import join
from astroai.tools.utils import load_yaml_conf, split_dataset
TF_CPP_MIN_LOG_LEVEL="1"

def create_cnn_01():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(2, (25, 25), activation='relu', input_shape=(250, 250, 1), name='conv2d_1'))
    model.add(tf.keras.layers.MaxPooling2D((20, 20), name='maxpool2d_1'))
    model.add(tf.keras.layers.Conv2D(2, (10, 10), activation='relu', name='conv2d_2'))
    model.add(tf.keras.layers.MaxPooling2D((20, 20), name='maxpool2d_2'))
    model.add(tf.keras.layers.Conv2D(2, (5, 5), activation='relu', name='conv2d_3'))
    model.add(tf.keras.layers.Dropout(0.2), name='drop_1')
    # add dense layer
    model.add(tf.keras.layers.Flatten(), name='flat_1')
    model.add(tf.keras.layers.Dense(10, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dropout(0.2, name='drop_2'))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid', name='dense_2'))
    return model

def compile_and_fit(model, train_ds, test_ds, batch_sz=32, epochs=25, learning=0.001, shuffle=True, logdate=True):
    logdir = join("logs", "cnn-v01-detect") 
    if logdate:
        logdir += datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning), 
              loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(train_ds, batch_size=batch_sz, epochs=epochs, validation_data=(test_ds), 
            callbacks=[tensorboard_callback], shuffle=shuffle)
    return history

def main(configuration):
    conf = load_yaml_conf(configuration)
    exposure = conf['preprocess']['time_stop'] - conf['preprocess']['time_start']
    src_sz, bkg_sz = conf['preprocess']['src_sample'], conf['preprocess']['bkg_sample']
    filename = join(conf['preprocess']['directory'], f"datasets_{exposure}s_{src_sz}src_{bkg_sz}bkg.npy") 
    #filename example: /data01/homes/dipiano/E4/datasets_50s_1000src_1000bkg.npy
    ds = np.load(filename, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
    train_data, train_labels, test_data, test_labels = split_dataset(ds, split=conf['detection']['split'], reshape=conf['detection']['reshape'], binning=conf['preprocess']['binning'])

    # create TF dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

    # create model
    model = create_cnn_01()
    model.summary()

    # compile and fit
    history = compile_and_fit(model=model, train_ds=train_ds, test_ds=test_ds, batch_sz=conf['detection']['batch_sz'], epochs=conf['detection']['epochs'], shuffle=conf['detection']['shuffle'], learning=conf['detection']['learning'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    main(args.configuration)