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

# ---- CNN BACKGROUND CLEANER -----

def create_bkg_cleaner(binning, decoder=1):
    input_shape = tf.keras.Input(shape=(binning, binning, 1))
    # encoder
    x = tf.keras.layers.Conv2D(5, (5, 5), activation='relu', padding='same')(input_shape)
    x = tf.keras.layers.AveragePooling2D((5, 5), padding='same')(x)
    x = tf.keras.layers.Conv2D(5, (5, 5), activation='relu', padding='same')(x)
    x = tf.keras.layers.AveragePooling2D((5, 5), padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.2)

    # decoder #1
    if decoder == 1:
        x = tf.keras.layers.Conv2D(5, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((5, 5))(x)
        x = tf.keras.layers.Conv2D(5, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((5, 5))(x)
        x = tf.keras.layers.Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

    # decoder #2
    elif decoder == 2:
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
    if savename is not None:
        print(f'SAVING MODEL AS {savename}.keras')
        model.save(f"{savename}.keras")

    # fit
    history = model.fit(train_noisy, train_clean, epochs=epochs, batch_size=batch_sz,
                        validation_data=(test_noisy, test_clean), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])
    if savename is not None:
        print(f'SAVING HISTORY AS {savename}_history.npy')
        np.save(f'{savename}_history.npy', history.history)
    return history

def cnn_bkg_cleaner(ds, conf, logdir, cpdir):
    # split dataset
    train_noisy,train_clean, test_noisy, test_clean = split_noisy_dataset(ds, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
    # create model
    model = create_bkg_cleaner(binning=conf['preprocess']['binning'])
    # compile and fit
    print('\n\n\n', conf['cnn']['saveas'], '\n\n\n')
    history = compile_and_fit_bkg_cleaner(model=model, train_clean=train_clean, train_noisy=train_noisy, test_clean=test_clean, test_noisy=test_noisy, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])
    return

# ----- CNN BINARY CLASSIFIER -----

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
    if savename is not None:
        print(f'SAVING MODEL AS {savename}.keras')
        model.save(f"{savename}.keras")

    # fit
    history = model.fit(train_ds, train_lb, batch_size=batch_sz, epochs=epochs, 
                        validation_data=(test_ds, test_lb), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])
    if savename is not None:
        print(f'SAVING HISTORY AS {savename}_history.npy')
        np.save(f'{savename}_history.npy', history.history)
    return history

def cnn_binary_classifier(ds, conf, logdir, cpdir):
    # split dataset
    train_data, train_labels, test_data, test_labels = split_dataset(ds, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
    # create model
    model = create_binary_classifier(binning=conf['preprocess']['binning'])
    # compile and fit
    history = compile_and_fit_binary_classifier(model=model, train_ds=train_data, train_lb=train_labels, test_ds=test_data, test_lb=test_labels, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])
    return history

# ----- CNN COORDINATES REGRESSOR -----

def create_loc_regressor(binning, number_of_conv=4):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(6, kernel_size=(12, 12), activation='relu', input_shape=(binning,binning,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    for i in range(number_of_conv):
        model.add(tf.keras.layers.Conv2D(12, (6, 6), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.summary()
    return model

def compile_and_fit_loc_regressor(model, train_ds, train_lb, test_ds, test_lb, logdir, cpdir, batch_sz=32, epochs=25, learning=0.001, shuffle=True, savename=None):
    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpdir, save_weights_only=True, verbose=1,
                                                              save_freq=5*int(len(train_ds) / batch_sz))
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning),  
                  loss=tf.keras.losses.mae, metrics=['accuracy'])
    if savename is not None:
        print(f'SAVING MODEL AS {savename}.keras')
        model.save(f"{savename}.keras")

    # fit
    history = model.fit(train_ds, train_lb, batch_size=batch_sz, epochs=epochs, 
                        validation_data=(test_ds, test_lb), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])
    if savename is not None:
        print(f'SAVING HISTORY AS {savename}_history.npy')
        np.save(f'{savename}_history.npy', history.history)
    return history

def cnn_loc_regressor(ds, conf, logdir, cpdir):
    # split dataset
    infotable = join(conf['cnn']['directory'], conf['cnn']['infotable'])
    train_data, train_labels, test_data, test_labels = split_regression_dataset(ds, infotable, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
    # create model
    model = create_binary_classifier(binning=conf['preprocess']['binning'])
    # compile and fit
    history = compile_and_fit_binary_classifier(model=model, train_ds=train_data, train_lb=train_labels, test_ds=test_data, test_lb=test_labels, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])
    return history

# ----- CNN MAIN ROUTINE -----

def main(configuration):
    conf = load_yaml_conf(configuration)
    filename = join(conf['cnn']['directory'], conf['cnn']['dataset'])
    mode = savename=conf['cnn']['mode']
    if isfile(filename):
        ds = load_dataset_npy(filename)
    else:
        raise FileNotFoundError(filename)

    logdir = tensorboard_logdir(savename=conf['cnn']['saveas'], suffix=conf['cnn']['suffix'], logdate=True)
    cpdir = checkpoint_dir(savename=conf['cnn']['saveas'], suffix=conf['cnn']['suffix'], logdate=True)

    # binary classification network
    if ('detect' in mode or 'class' in mode):
        history = cnn_binary_classifier(ds=ds, conf=conf, logdir=logdir, cpdir=cpdir)

    # background cleaner autoencoder
    elif 'clean' in mode:
        history = cnn_bkg_cleaner(ds=ds, conf=conf, logdir=logdir, cpdir=cpdir)

    # hotspots identification regressor
    elif 'loc' in mode:
        history = cnn_loc_regressor(ds=ds, conf=conf, logdir=logdir, cpdir=cpdir)

    else:
        raise ValueError('Not implemented yet')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    print(f"\n\n{'!'*3} CNN {args.mode.upper()} {'!'*3}\n\n")
    main(args.configuration, args.mode)