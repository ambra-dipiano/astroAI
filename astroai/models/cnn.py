# *****************************************************************************
# Copyright (C) 2023 INAF
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import argparse
import tensorflow as tf
from os.path import join, isfile
from astroai.tools.utils import *
TF_CPP_MIN_LOG_LEVEL="1"


def create_bkg_cleaner(binning):
    input_shape = tf.keras.Input(shape=(binning, binning, 1))
    # encoder
    x = tf.keras.layers.Conv2D(5, (5, 5), activation='relu', padding='same')(input_shape)
    x = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(x)
    x = tf.keras.layers.Conv2D(5, (5, 5), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((5, 5), padding='same')(x)

    # decoder #1
    #x = tf.keras.layers.Conv2D(25, (3, 3), activation='relu', padding='same')(encoded)
    #x = tf.keras.layers.UpSampling2D((2, 2))(x)
    #x = tf.keras.layers.Conv2D(25, (3, 3), activation='relu', padding='same')(x)
    #x = tf.keras.layers.UpSampling2D((2, 2))(x)
    #x = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # decoder #2
    x = tf.keras.layers.Conv2DTranspose(5, (5, 5), strides=5, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(5, (5, 5), strides=5, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(1, (5, 5), activation="sigmoid", padding="same")(x)

    # autoencoder
    autoencoder = tf.keras.Model(input_shape, x)
    autoencoder.summary()
    return autoencoder

def compile_and_fit_bkg_cleaner(model, train_noisy, train_clean, test_noisy, test_clean, logdir, cpdir, batch_sz=32, epochs=25, learning=0.001, shuffle=True, savename=None):
    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpdir, save_weights_only=True, verbose=1, 
                                                              save_freq=5*int(len(train_noisy) / batch_sz))
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning),  
                  loss=tf.keras.losses.binary_crossentropy) # metrics=['accuracy']
    # fit
    history = model.fit(train_noisy, train_clean, epochs=epochs, batch_size=batch_sz,
                        validation_data=(test_noisy, test_clean), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])
    
    if savename is not None:
        model.save(f"{savename}.keras")
    return history

def create_binary_classifier(binning):
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

def compile_and_fit_binary_classifier(model, train_ds, train_lb, test_ds, test_lb, logdir, cpdir, batch_sz=32, epochs=25, learning=0.001, shuffle=True, savename=None):
    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpdir, save_weights_only=True, verbose=1,
                                                              save_freq=5*int(len(train_ds) / batch_sz))
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning),  
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    # fit
    history = model.fit(train_ds, train_lb, batch_size=batch_sz, epochs=epochs, 
                        validation_data=(test_ds, test_lb), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])

    if savename is not None:
        model.save(f"{savename}.keras")
    return history

def main(configuration, mode):
    conf = load_yaml_conf(configuration)
    filename = join(conf['cnn']['directory'], conf['cnn']['dataset'])
    if isfile(filename):
        ds = load_dataset_npy(filename)
    else:
        raise FileNotFoundError(filename)

    logdir = tensorboard_logdir(mode=mode, suffix=conf['cnn']['suffix'], logdate=True)
    cpdir = checkpoint_dir(mode=mode, suffix=conf['cnn']['suffix'], logdate=True)

    # binary classification network
    if ('detect' in mode or 'class' in mode):
        # split dataset
        train_data, train_labels, test_data, test_labels = split_dataset(ds, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
        # create model
        model = create_binary_classifier(binning=conf['preprocess']['binning'])
        # compile and fit
        history = compile_and_fit_binary_classifier(model=model, train_ds=train_data, train_lb=train_labels, test_ds=test_data, test_lb=test_labels, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])

    # background cleaner autoencoder
    elif 'clean' in mode:
        # split dataset
        train_noisy,train_clean, test_noisy, test_clean = split_noisy_dataset(ds, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
        # create model
        model = create_bkg_cleaner(binning=conf['preprocess']['binning'])
        # compile and fit
        history = compile_and_fit_bkg_cleaner(model=model, train_clean=train_clean, train_noisy=train_noisy, test_clean=test_clean, test_noisy=test_noisy, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])


    else:
        raise ValueError('Not implemented yet')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['classify', 'clean', 'localise'], help="scope of the CNN and thus related network")
    args = parser.parse_args()

    print(f"\n\n{'!'*3} CNN {args.mode.upper()} {'!'*3}\n\n")
    main(args.configuration, args.mode)