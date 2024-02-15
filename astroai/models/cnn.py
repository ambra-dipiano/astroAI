# *****************************************************************************
# Copyright (C) 2023 Ambra Di Piano
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

# ---- CREATE MODELS -----

# BACKGROUND CLEANING MODEL
def create_bkg_cleaner(binning, encoder=2, decoder=1, conv_filter=2, conv_kern=2, pool_kern=2):
    input_shape = tf.keras.Input(shape=(binning, binning, 1))

    # encoder #1
    if encoder == 1:
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(input_shape)
        x = tf.keras.layers.MaxPool2D((pool_kern, pool_kern), padding='same')(x)
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPool2D((pool_kern, pool_kern), padding='same')(x)

    # encoder #2
    elif encoder == 2:
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(input_shape)
        x = tf.keras.layers.AveragePooling2D((pool_kern, pool_kern), padding='same')(x)
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(x)
        x = tf.keras.layers.AveragePooling2D((pool_kern, pool_kern), padding='same')(x)
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(x)
        x = tf.keras.layers.AveragePooling2D((pool_kern, pool_kern), padding='same')(x)

    # decoder #1
    if decoder == 1:
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((pool_kern, pool_kern))(x)
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((pool_kern, pool_kern))(x)
        x = tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((pool_kern, pool_kern))(x)
        x = tf.keras.layers.Conv2D(1, (conv_kern, conv_kern), activation='sigmoid', padding='same')(x)

    # decoder #2
    elif decoder == 2:
        x = tf.keras.layers.Conv2DTranspose(conv_filter, (conv_kern, conv_kern), strides=5, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(conv_filter, (conv_kern, conv_kern), strides=5, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(1, (conv_kern, conv_kern), activation="sigmoid", padding="same")(x)

    # autoencoder
    autoencoder = tf.keras.Model(input_shape, x)
    autoencoder.summary()
    return autoencoder

# SOURCE LOCALISATION MODEL
def create_loc_regressor(binning, number_of_conv=4, conv_filter=2, conv_kern=2, pool_kern=2, dropout=0.2, dense=10):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(conv_filter, kernel_size=(conv_kern*2, conv_kern*2), activation='relu', input_shape=(binning,binning,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_kern, pool_kern)))
    for i in range(number_of_conv):
        model.add(tf.keras.layers.Conv2D(conv_filter*2, (conv_kern, conv_kern), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_kern, pool_kern)))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.summary()
    return model

# SOURCE DETECTION MODEL
def create_binary_classifier(binning, conv_filter=2, conv_kern=2, pool_kern=2, dropout=0.2, dense=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', input_shape=(binning, binning, 1), name='conv2d_1'))
    model.add(tf.keras.layers.MaxPooling2D((pool_kern, pool_kern), name='maxpool2d_1'))
    model.add(tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', name='conv2d_2'))
    model.add(tf.keras.layers.MaxPooling2D((pool_kern, pool_kern), name='maxpool2d_2'))
    model.add(tf.keras.layers.Conv2D(conv_filter, (conv_kern, conv_kern), activation='relu', name='conv2d_3'))
    model.add(tf.keras.layers.Dropout(dropout, name='drop_1'))
    model.add(tf.keras.layers.Flatten(name='flat_1'))
    model.add(tf.keras.layers.Dense(dense, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dropout(dropout, name='drop_2'))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid', name='dense_2'))
    model.summary()
    return model





# ---- COMPILE MODELS -----

# BACKGROUND CLEANING COMPILER
def compile_and_fit_bkg_cleaner(model, train_noisy, train_clean, test_noisy, test_clean, logdir, cpdir, batch_sz=32, epochs=25, learning=0.001, shuffle=True, savename=None):
    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpdir, save_weights_only=True, verbose=1, save_freq=5*int(len(train_noisy) / batch_sz))
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning),  
                  loss=tf.keras.losses.binary_crossentropy) # metrics=['accuracy']
    # fit
    history = model.fit(train_noisy, train_clean, epochs=epochs, batch_size=batch_sz,
                        validation_data=(test_noisy, test_clean), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])
    
    if savename is not None:
        print(f'SAVING MODEL AS {savename}.keras')
        model.save(f"{savename}.keras")
        print(f'SAVING HISTORY AS {savename}_history.npy')
        np.save(f'{savename}_history.npy', history.history)
    return history

# SOURCE LOCALISATION COMPILER
def compile_and_fit_loc_regressor(model, train_ds, train_lb, test_ds, test_lb, logdir, cpdir, batch_sz=32, epochs=25, learning=0.001, shuffle=True, savename=None):
    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpdir, save_weights_only=True, verbose=1, save_freq=5*int(len(train_ds) / batch_sz))
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning),  
                  loss=tf.keras.losses.mae, metrics=['accuracy'])
    # fit
    history = model.fit(train_ds, train_lb, batch_size=batch_sz, epochs=epochs, 
                        validation_data=(test_ds, test_lb), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])
    
    if savename is not None:
        print(f'SAVING MODEL AS {savename}.keras')
        model.save(f"{savename}.keras")
        print(f'SAVING HISTORY AS {savename}_history.npy')
        np.save(f'{savename}_history.npy', history.history)

# SOURCE DETECTION COMPILER
def compile_and_fit_binary_classifier(model, train_ds, train_lb, test_ds, test_lb, logdir, cpdir, batch_sz=32, epochs=25, learning=0.001, shuffle=True, savename=None):
    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cpdir, save_weights_only=True, verbose=1, save_freq=5*int(len(train_ds) / batch_sz))
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning),  
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    # fit
    history = model.fit(train_ds, train_lb, batch_size=batch_sz, epochs=epochs, 
                        validation_data=(test_ds, test_lb), shuffle=shuffle,
                        callbacks=[tensorboard_callback, checkpoints_callback, earlystop_callback])
    
    if savename is not None:
        print(f'SAVING MODEL AS {savename}.keras')
        model.save(f"{savename}.keras")
        print(f'SAVING HISTORY AS {savename}_history.npy')
        np.save(f'{savename}_history.npy', history.history)






# ---- COMPILE MODELS -----

# BACKGROUND CLEANING ENTRYPOINT
def cnn_bkg_cleaner(ds, conf, logdir, cpdir):
    # split dataset
    train_noisy,train_clean, test_noisy, test_clean = split_noisy_dataset(ds, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
    # create model
    model = create_bkg_cleaner(binning=conf['preprocess']['binning'], conv_filter=conf['cnn']['layers']['conv_filter'], conv_kern=conf['cnn']['layers']['conv_kernel'], pool_kern=conf['cnn']['layers']['sampling_kernel'])
    # compile and fit
    history = compile_and_fit_bkg_cleaner(model=model, train_clean=train_clean, train_noisy=train_noisy, test_clean=test_clean, test_noisy=test_noisy, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])

# SOURCE LOCALISATION ENTRYPOINT
def cnn_loc_regressor(ds, conf, logdir, cpdir):
    # split dataset
    train_data, train_labels, test_data, test_labels = split_regression_dataset(ds, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
    # create model
    model = create_loc_regressor(binning=conf['preprocess']['binning'], number_of_conv=conf['cnn']['layers']['number_convs'], conv_filter=conf['cnn']['layers']['conv_filter'], conv_kern=conf['cnn']['layers']['conv_kernel'], pool_kern=conf['cnn']['layers']['sampling_kernel'], dense=conf['cnn']['layers']['dense'], dropout=conf['cnn']['layers']['dropout'])
    # compile and fit
    history = compile_and_fit_loc_regressor(model=model, train_ds=train_data, train_lb=train_labels, test_ds=test_data, test_lb=test_labels, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])

# SOURCE DETECTION ENTRYPOINT
def cnn_binary_classifier(ds, conf, logdir, cpdir):
    # split dataset
    train_data, train_labels, test_data, test_labels = split_dataset(ds, split=conf['cnn']['split'], reshape=conf['cnn']['reshape'], binning=conf['preprocess']['binning'])
    # create model
    model = create_binary_classifier(binning=conf['preprocess']['binning'], conv_filter=conf['cnn']['layers']['conv_filter'], conv_kern=conf['cnn']['layers']['conv_kernel'], pool_kern=conf['cnn']['layers']['sampling_kernel'], dense=conf['cnn']['layers']['dense'], dropout=conf['cnn']['layers']['dropout'])
    # compile and fit
    history = compile_and_fit_binary_classifier(model=model, train_ds=train_data, train_lb=train_labels, test_ds=test_data, test_lb=test_labels, logdir=logdir, cpdir=cpdir, batch_sz=conf['cnn']['batch_sz'], epochs=conf['cnn']['epochs'], shuffle=conf['cnn']['shuffle'], learning=conf['cnn']['learning'], savename=conf['cnn']['saveas'])





# ----- CNN MAIN ROUTINE -----

def main(configuration):
    conf = load_yaml_conf(configuration)
    filename = join(conf['cnn']['directory'], conf['cnn']['dataset'])
    mode = conf['cnn']['mode']
    if isfile(filename):
        ds = load_dataset_npy(filename)
    else:
        raise FileNotFoundError(filename)

    logdir = tensorboard_logdir(savename=conf['cnn']['saveas'], logdate=True)
    cpdir = checkpoint_dir(savename=conf['cnn']['saveas'], logdate=True)

    # binary classification network
    if ('detect' in mode or 'class' in mode):
        cnn_binary_classifier(ds=ds, conf=conf, logdir=logdir, cpdir=cpdir)

    # background cleaner autoencoder
    elif 'clean' in mode:
        cnn_bkg_cleaner(ds=ds, conf=conf, logdir=logdir, cpdir=cpdir)

    # hotspots identification regressor
    elif 'loc' in mode:
        cnn_loc_regressor(ds=ds, conf=conf, logdir=logdir, cpdir=cpdir)

    else:
        raise ValueError('Not implemented yet')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--configuration', type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    print(f"\n\n{'!'*3} START CNN TRAINING {'!'*3}\n\n")
    main(args.configuration)