# *****************************************************************************
# Copyright (C) 2023 INAF
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import yaml
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile
from datetime import datetime
from astropy.wcs import WCS
from astropy.table import Table
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# set WCS for plotting
def set_wcs(point_ra, point_dec, point_ref, pixelsize):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---CAR', 'DEC--CAR']
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.crpix = [point_ref, point_ref]
    w.wcs.crval = [point_ra, point_dec]
    w.wcs.cdelt = [-pixelsize, pixelsize]
    #w.wcs.lonpole = 0.0
    #w.wcs.latpole = 67.49
    return w

# extract heatmap from DL3 with configurable exposure
def extract_heatmap(data, trange, smoothing, nbins, save=False, save_name=None):
    data = data[(data['TIME'] >= trange[0]) & (data['TIME'] <= trange[1])] 
    ra = data['RA'].to_numpy()
    dec = data['DEC'].to_numpy()
    heatmap, xe, ye = np.histogram2d(ra, dec, bins=nbins)
    if smoothing != 0:
        heatmap = gaussian_filter(heatmap, sigma=smoothing)
    if save and save_name is not None:
        np.save(save_name, heatmap, allow_pickle=True, fix_imports=True)
    return heatmap.T

# normalise single heatmap between 0 and 1
def normalise_heatmap(heatmap, save=False, save_name=None):
    min_value = np.min(heatmap)
    max_value = np.max(heatmap)
    heatmap = (heatmap - min_value) / (max_value - min_value)
    if save and save_name is not None:
        np.save(save_name, heatmap, allow_pickle=True, fix_imports=True)
    return heatmap

# normalise all heatmaps dataset between 0 and 1 with fixed normalisation
def normalise_dataset(ds, max_value, min_value=0, save=False, save_name=None):
    ds = (ds - min_value) / (max_value - min_value)
    if save and save_name is not None:
        np.save(save_name, ds, allow_pickle=True, fix_imports=True)
    return ds

# plot heatmap
def plot_heatmap(heatmap, title='heatmap', show=False, save=False, save_name=None):
    plt.figure()
    plt.title(title)
    plt.imshow(heatmap, vmin=0, vmax=1)
    plt.xlabel('x(det) [pixels]')
    plt.ylabel('y(det) [pixels]')
    plt.colorbar()
    if save and save_name is not None:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.close()
    return

# gather simulations and create heatmaps 
def process_dataset(src_dataset_path, bkg_dataset_path, trange, smoothing, binning, sample, save=False, output=None, min_max_norm=1, mode='ds'):
    datapath = {'SRC': src_dataset_path, 'BKG': bkg_dataset_path}
    exposure = trange[1]-trange[0]

    # datasets files
    datafiles = {'SRC': [], 'BKG': []}
    classes = datafiles.keys()
    for k in classes:
        datafiles[k] = sorted([join(datapath[k], f) for f in listdir(datapath[k]) if '.fits' in f and isfile(join(datapath[k], f))])

        
    # create images dataset 
    datasets = {'SRC': [], 'BKG': []}
    classes = datasets.keys()
    for k in classes:
        print(f"Load {k} data...")
        for f in tqdm(datafiles[k][:sample]):
            # load
            heatmap = Table.read(f, hdu=1).to_pandas()
            # integrate exposure
            heatmap = extract_heatmap(heatmap, trange, smoothing, binning)
            # normalise map
            if min_max_norm == 1:
                if 'detect' in mode:
                    heatmap = stretch_smooth(heatmap, smoothing)
                elif 'clean' in mode:
                    heatmap = normalise_heatmap(heatmap)
                else:
                    pass
            elif min_max_norm == 0:
                continue
            else:
                if 'detect' in mode:
                    heatmap = stretch_min_max(heatmap, vmax=min_max_norm)
                elif 'clean' in mode:
                    heatmap = normalise_dataset(heatmap, max_value=min_max_norm)
                else: 
                    pass
            # add to dataset
            if heatmap.shape != (binning, binning):
                heatmap.reshape(binning, binning)
            datasets[k].append(heatmap)

    # convert to numpy array
    datasets['BKG'] = np.array(datasets['BKG'])
    datasets['SRC'] = np.array(datasets['SRC'])
    
    # save processed dataset
    if save and output is not None:
        np.save(join(output, f'{mode}_{exposure}s_{smoothing}sgm_{sample}sz.npy'), datasets, allow_pickle=True, fix_imports=True)
    return datasets

# gather simulations and create heatmaps
def get_and_normalise_dataset(ds_path, sample, save=False, output=None, min_max_norm=None):
    datapath = {'SRC': join(ds_path, 'crab'), 'BKG': join(ds_path, 'background')}

    # datasets files
    datafiles = {'SRC': [], 'BKG': []}
    classes = datafiles.keys()
    for k in classes:
        datafiles[k] = sorted([join(datapath[k], f) for f in listdir(datapath[k]) if '.npy' in f and isfile(join(datapath[k], f))])
        
    # create images dataset 
    datasets = {'SRC': [], 'BKG': []}
    classes = datasets.keys()
    for k in classes:
        print(f"Load {k} data...")
        for f in tqdm(datafiles[k][:sample]):
            # load
            heatmap = np.load(f, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
            # normalise map
            if min_max_norm is not None:
                normalise_dataset(heatmap, min_value=min_max_norm[0], max_value=min_max_norm[1])
            else:
                heatmap = normalise_heatmap(heatmap)
            # add to dataset
            datasets[k].append(heatmap)

    # convert to numpy array
    datasets['BKG'] = np.array(datasets['BKG'])
    datasets['SRC'] = np.array(datasets['SRC'])
    
    # save processed dataset
    if save and output is not None:
        np.save(join(output, f'ds_normalised.npy'), datasets, allow_pickle=True, fix_imports=True)
    return datasets

# split train and test datasets with labels
def split_dataset(dataset, split=80, reshape=True, binning=250):
    total_size = len(dataset['SRC']) + len(dataset['BKG'])
    # train_size = split % of half total size (sum of src and bkg samples)
    #TODO: instead of halving treat separately src and bkg sample sizes
    train_size = int(((total_size / 100) * split) / 2)
    # train dataset
    train_src = np.copy(dataset['SRC'][:train_size])
    train_bkg = np.copy(dataset['BKG'][:train_size])
    train_data = np.append(train_src, train_bkg, axis=0) 
    train_labels = np.array([1 for f in range(len(train_src))] + [0 for f in range(len(train_bkg))])
    # test dataset
    test_src = np.copy(dataset['SRC'][train_size:])
    test_bkg = np.copy(dataset['BKG'][train_size:])
    test_data = np.append(test_src, test_bkg, axis=0)
    test_labels = np.array([1 for f in range(len(test_src))] + [0 for f in range(len(test_bkg))])
    # reshape
    if reshape:
        train_data = train_data.reshape(train_data.shape[0], binning, binning, 1)
        test_data = test_data.reshape(test_data.shape[0], binning, binning, 1)
        train_labels = train_labels.reshape(train_data.shape[0], 1)
        test_labels = test_labels.reshape(test_data.shape[0], 1)    
    return train_data, train_labels, test_data, test_labels

def split_noisy_dataset(dataset, split=80, reshape=True, binning=250):
    size = len(dataset['SRC'])
    # train_size = split % of half total size (sum of src and bkg samples)
    #TODO: instead of halving treat separately src and bkg sample sizes
    train_size = int((size / 100) * split)
    # train dataset
    train_clean = np.copy(dataset['SRC'][:train_size])
    train_noisy = np.copy(dataset['BKG'][:train_size])
    # test dataset
    test_clean = np.copy(dataset['SRC'][train_size:])
    test_noisy = np.copy(dataset['BKG'][train_size:])
    # reshape
    if reshape:
        train_clean = train_clean.reshape(train_clean.shape[0], binning, binning, 1)
        train_noisy = train_noisy.reshape(train_noisy.shape[0], binning, binning, 1)
        test_clean = test_clean.reshape(test_clean.shape[0], binning, binning, 1)
        test_noisy = test_noisy.reshape(test_noisy.shape[0], binning, binning, 1)    
    return train_clean, train_noisy, test_clean, test_noisy

# load configuration file
def load_yaml_conf(yamlfile):
    with open(yamlfile) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    return configuration

def get_min_max_norm(vmin, vmax):
    if vmin == 0 and vmax == 0:
        min_max_norm = 0
    elif vmin == 1 and vmax == 1:
        min_max_norm = 1
    else:
        min_max_norm = (vmin, vmax)
    return min_max_norm

# min-max stretch but the max is mean+sigma*std and min is mean-sigma*std
def stretch_smooth(heatmap, sigma=3):
    if np.min(heatmap)<0: # if the min is less that zero, first we add min to all pixels so min becomes 0
        heatmap = heatmap + np.abs(np.min(heatmap)) 
    heatmap = heatmap / np.max(heatmap)
    std = np.std(heatmap)
    mean = np.mean(heatmap)
    vmax = mean+(sigma*std)
    vmin = mean-(sigma*std)
    heatmap = (heatmap-vmin)/(vmax-vmin)
    # this streching cuases the values less than `mean-simga*std` to become negative
    # so we clip the values less than 0 
    heatmap[heatmap<0]=0
    return heatmap

# min-max stretch but the max is mean+sigma*std and min is mean-sigma*std
def stretch_min_max(heatmap, vmax, vmin=0):
    if np.min(heatmap)<0: # if the min is less that zero, first we add min to all pixels so min becomes 0
        heatmap = heatmap + np.abs(np.min(heatmap)) 
    heatmap = heatmap / np.max(heatmap)
    heatmap = (heatmap-vmin)/(vmax-vmin)
    # this streching cuases the values less than `mean-simga*std` to become negative
    # so we clip the values less than 0 
    heatmap[heatmap<0]=0
    return heatmap

def tensorboard_logdir(mode, suffix, logdate):
    logdir = join("logs", f"cnn_{mode}")
    if suffix is not None:
        logdir += '_' + suffix
    if logdate:
        logdir += '_' + datetime.now().strftime("%Y%m%dT%H%M%S")
    return logdir

def load_dataset_npy(path):
    ds = np.load(path, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
    return ds