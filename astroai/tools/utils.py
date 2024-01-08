# *****************************************************************************
# Copyright (C) 2023 INAF
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile
from datetime import datetime
from astropy.wcs import WCS
from astropy.table import Table
from astropy.io import fits
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
def extract_heatmap(data, smoothing, nbins, save=False, save_name=None, filter=True, trange=None):
    if filter and trange is not None:
        data = data[(data['TIME'] >= trange[0]) & (data['TIME'] <= trange[1])] 
    ra = data['RA'].to_numpy()
    dec = data['DEC'].to_numpy()
    heatmap, xe, ye = np.histogram2d(ra, dec, bins=nbins)
    if smoothing != 0:
        heatmap = gaussian_filter(heatmap, sigma=smoothing)
    if save and save_name is not None:
        np.save(save_name, heatmap, allow_pickle=True, fix_imports=True)
    return heatmap.T

# smooth heatmap from DL4 
def smooth_heatmap(heatmap, smoothing, save=False, save_name=None):
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
def process_dataset(ds1_dataset_path, ds2_dataset_path, trange, smoothing, binning, sample, save=False, output=None, norm_value=1, mode='ds', suffix=None, dl=3):
    datapath = {'DS1': ds1_dataset_path, 'DS2': ds2_dataset_path}
    exposure = trange[1]-trange[0]

    # datasets files
    datafiles = {'DS1': [], 'DS2': []}
    classes = datafiles.keys()
    for k in classes:
        datafiles[k] = sorted([join(datapath[k], f) for f in listdir(datapath[k]) if '.fits' in f and isfile(join(datapath[k], f))])

        
    # create images dataset 
    datasets = {'DS1': [], 'DS2': []}
    classes = datasets.keys()
    for k in classes:
        print(f"Load {k} data...")
        for f in tqdm(datafiles[k][:sample]):
            # load
            if dl == 3:
                heatmap = Table.read(f, hdu=1).to_pandas()
                # integrate exposure
                heatmap = extract_heatmap(heatmap, trange, smoothing, binning)
            elif dl == 4:
                with fits.open(f) as h:
                    heatmap = smooth_heatmap(h[0].data, smoothing)

            # normalise map
            if norm_value == 1:
                if 'detect' in mode:
                    heatmap = stretch_smooth(heatmap, smoothing)
                elif 'clean' in mode:
                    heatmap = normalise_heatmap(heatmap)
                else:
                    pass
            elif norm_value == 0:
                continue
            else:
                if 'detect' in mode:
                    heatmap = stretch_min_max(heatmap, vmax=norm_value)
                elif 'clean' in mode:
                    heatmap = normalise_dataset(heatmap, max_value=norm_value)
                else: 
                    pass
            # add to dataset
            if heatmap.shape != (binning, binning):
                heatmap.reshape(binning, binning)
            datasets[k].append(heatmap)

    # convert to numpy array
    datasets['DS2'] = np.array(datasets['DS2'])
    datasets['DS1'] = np.array(datasets['DS1'])
    
    # save processed dataset
    if save and output is not None:
        filename = join(output, f'{mode}_{exposure}s_{smoothing}sgm_{sample}sz.npy')
        if suffix is not None:
            filename = filename.replace('.npy', f'_{suffix}.npy')
        np.save(filename, datasets, allow_pickle=True, fix_imports=True)
        print(f"Process complete: {filename}")
    return datasets

# gather simulations and create heatmaps for regressor
def process_regressor_dataset(ds_dataset_path, smoothing, binning, sample, save=False, output=None, norm_single_map=False, stretch=True, norm_value=1, suffix=None, dl=3):
    datapath = {'DS': ds_dataset_path}

    # datasets files
    datafiles = {'DS': []}
    datafiles['DS'] = sorted([join(datapath['DS'], f) for f in listdir(datapath['DS']) if '.fits' in f and isfile(join(datapath['DS'], f))])
        
    # create images dataset 
    datasets = {'DS': []}
    print(f"Load DS data...")
    for f in tqdm(datafiles['DS'][:sample]):
        # load
        if dl == 3:
            heatmap = Table.read(f, hdu=1).to_pandas()
            # integrate exposure
            heatmap = extract_heatmap(heatmap, smoothing, binning)
        elif dl == 4:
            with fits.open(f) as h:
                heatmap = smooth_heatmap(h[0].data, smoothing)

        # normalise map
        if norm_single_map:
            if stretch:
                heatmap = stretch_smooth(heatmap, smoothing)
            else:
                heatmap = normalise_heatmap(heatmap)
        else:
            if stretch:
                heatmap = stretch_min_max(heatmap, vmax=norm_value)
            else:
                heatmap = normalise_dataset(heatmap, max_value=norm_value)
        # add to dataset
        if heatmap.shape != (binning, binning):
            heatmap.reshape(binning, binning)
        datasets['DS'].append(heatmap)

    # convert to numpy array
    datasets['DS'] = np.array(datasets['DS'])
    
    # save processed dataset
    if save and output is not None:
        filename = join(output, f'regression_{smoothing}sgm_{sample}sz.npy')
        if suffix is not None:
            filename = filename.replace('.npy', f'_{suffix}.npy')
        np.save(filename, datasets, allow_pickle=True, fix_imports=True)
        print(f"Process complete: {filename}")
    return datasets

# gather simulations and create heatmaps
def get_and_normalise_dataset(ds_path, sample, save=False, output=None, norm_value=None):
    datapath = {'DS1': join(ds_path, 'crab'), 'DS2': join(ds_path, 'background')}

    # datasets files
    datafiles = {'DS1': [], 'DS2': []}
    classes = datafiles.keys()
    for k in classes:
        datafiles[k] = sorted([join(datapath[k], f) for f in listdir(datapath[k]) if '.npy' in f and isfile(join(datapath[k], f))])
        
    # create images dataset 
    datasets = {'DS1': [], 'DS2': []}
    classes = datasets.keys()
    for k in classes:
        print(f"Load {k} data...")
        for f in tqdm(datafiles[k][:sample]):
            # load
            heatmap = np.load(f, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
            # normalise map
            if norm_value is not None:
                normalise_dataset(heatmap, min_value=norm_value[0], max_value=norm_value[1])
            else:
                heatmap = normalise_heatmap(heatmap)
            # add to dataset
            datasets[k].append(heatmap)

    # convert to numpy array
    datasets['DS2'] = np.array(datasets['DS2'])
    datasets['DS1'] = np.array(datasets['DS1'])
    
    # save processed dataset
    if save and output is not None:
        np.save(join(output, f'ds_normalised.npy'), datasets, allow_pickle=True, fix_imports=True)
    return datasets

# split train and test datasets with labels
def split_dataset(dataset, split=80, reshape=True, binning=250):
    total_size = len(dataset['DS1']) + len(dataset['DS2'])
    # train_size = split % of half total size (sum of src and bkg samples)
    #TODO: instead of halving treat separately src and bkg sample sizes
    train_size = int(((total_size / 100) * split) / 2)
    # train dataset
    train_src = np.copy(dataset['DS1'][:train_size])
    train_bkg = np.copy(dataset['DS2'][:train_size])
    train_data = np.append(train_src, train_bkg, axis=0) 
    train_labels = np.array([1 for f in range(len(train_src))] + [0 for f in range(len(train_bkg))])
    # test dataset
    test_src = np.copy(dataset['DS1'][train_size:])
    test_bkg = np.copy(dataset['DS2'][train_size:])
    test_data = np.append(test_src, test_bkg, axis=0)
    test_labels = np.array([1 for f in range(len(test_src))] + [0 for f in range(len(test_bkg))])
    # reshape
    if reshape:
        train_data = train_data.reshape(train_data.shape[0], binning, binning, 1)
        test_data = test_data.reshape(test_data.shape[0], binning, binning, 1)
        train_labels = train_labels.reshape(train_data.shape[0], 1)
        test_labels = test_labels.reshape(test_data.shape[0], 1)    
    return train_data, train_labels, test_data, test_labels

# split train and test datasets with coordinates labels
def split_regression_dataset(dataset, infotable, split=80, reshape=True, binning=250):
    total_size = len(dataset['DS'])
    train_size = int((total_size / 100) * split)
    infodata = pd.read_csv(infotable, sep=' ', header=0).sort_values(by=['seed'])
    seeds = infodata['seed']
    total_labels = np.array([(infodata[infodata['seed']==seed]['source_ra'].values[0], infodata[infodata['seed']==seed]['source_dec'].values[0]) for seed in seeds])
    # train dataset
    train_data = np.copy(dataset['DS'][:train_size])
    train_labels = np.copy(total_labels[:train_size])
    # test dataset
    test_data = np.copy(dataset['DS'][train_size:])
    test_labels = np.copy(total_labels[train_size:])
    # reshape
    if reshape:
        train_data = train_data.reshape(train_data.shape[0], binning, binning, 1)
        test_data = test_data.reshape(test_data.shape[0], binning, binning, 1)
        train_labels = train_labels.reshape(train_data.shape[0], 1)
        test_labels = test_labels.reshape(test_data.shape[0], 1)    
    return train_data, train_labels, test_data, test_labels

def split_noisy_dataset(dataset, split=80, reshape=True, binning=250):
    size = len(dataset['DS1'])
    # train_size = split % of half total size (sum of src and bkg samples)
    #TODO: instead of halving treat separately src and bkg sample sizes
    train_size = int((size / 100) * split)
    # train dataset
    train_clean = np.copy(dataset['DS1'][:train_size])
    train_noisy = np.copy(dataset['DS2'][:train_size])
    # test dataset
    test_clean = np.copy(dataset['DS1'][train_size:])
    test_noisy = np.copy(dataset['DS2'][train_size:])
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

def get_norm_value(vmin, vmax):
    if vmin == 0 and vmax == 0:
        norm_value = 0
    elif vmin == 1 and vmax == 1:
        norm_value = 1
    else:
        norm_value = (vmin, vmax)
    return norm_value

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

def tensorboard_logdir(savename, suffix=None, logdate=True):
    logdir = join("logs", f"{savename}")
    if logdate:
        logdir += '_' + datetime.now().strftime("%Y%m%dT%H%M%S")
    if suffix is not None:
        logdir += '_' + suffix
    return logdir

def checkpoint_dir(savename, suffix=None, logdate=False):
    logdir = join("checkpoints", f"{savename}")
    if logdate:
        logdir += '_' + datetime.now().strftime("%Y%m%dT%H%M%S")
    if suffix is not None:
        logdir += '_' + suffix
    return logdir

def load_dataset_npy(path):
    ds = np.load(path, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
    return ds