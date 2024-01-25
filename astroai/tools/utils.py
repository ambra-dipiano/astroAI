# *****************************************************************************
# Copyright (C) 2023 Ambra Di Piano
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import yaml
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile, basename
from datetime import datetime
from astropy.wcs import WCS
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
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
    w.wcs.latpole = 30.078643218
    w.wcs.lonpole = 0.0
    return w

# extract heatmap from DL3 with configurable exposure
def extract_heatmap_from_table(data, smoothing, nbins, save=False, save_name=None, filter=True, trange=None, wcs=None):
    if filter and trange is not None:
        data = data[(data['TIME'] >= trange[0]) & (data['TIME'] <= trange[1])] 
    ra, dec = data['RA'].to_numpy(), data['DEC'].to_numpy()
    
    if wcs is not None:
        ra, dec = wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg', frame='icrs'))
    heatmap, xe, ye = np.histogram2d(ra, dec, bins=nbins)
    if smoothing != 0:
        heatmap = smooth_heatmap(heatmap, sigma=smoothing)
    if save and save_name is not None:
        np.save(save_name, heatmap, allow_pickle=True, fix_imports=True)
    return heatmap.T, xe, ye

# extract heatmap from DL3 with configurable exposure
def extract_heatmap(data, smoothing, nbins, save=False, save_name=None, filter=True, trange=None, wcs=None):
    if filter and trange is not None:
        data = data[(data['TIME'] >= trange[0]) & (data['TIME'] <= trange[1])] 
    #ra = data['RA'].to_numpy()
    #dec = data['DEC'].to_numpy()
    ra, dec = data.field('RA'), data.field('DEC')
    
    if wcs is not None:
        ra, dec = wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg', frame='icrs'))
    heatmap, xe, ye = np.histogram2d(ra, dec, bins=nbins)
    if smoothing != 0:
        heatmap = smooth_heatmap(heatmap, sigma=smoothing)
    if save and save_name is not None:
        np.save(save_name, heatmap, allow_pickle=True, fix_imports=True)
    return heatmap.T, xe, ye

# smooth heatmap from DL4 
def smooth_heatmap(heatmap, smoothing, save=False, save_name=None):
    heatmap = gaussian_filter(heatmap, sigma=smoothing)
    if save and save_name is not None:
        np.save(save_name, heatmap, allow_pickle=True, fix_imports=True)
    return heatmap

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
def plot_heatmap(heatmap, title='heatmap', show=False, save=False, save_name=None, wcs=None):
    if wcs is not None:
        plt.subplot(projection=wcs)
    else:
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

# plot heatmap
def plot_heatmap_wcs(heatmap, title='heatmap', xlabel='x', ylabel='y', show=False, save=False, 
                     save_name=None, wcs=None, src=None, pnt=None):
    if wcs is not None:
        ax = plt.subplot(projection=wcs, aspect='equal')
        ax.coords[0].set_format_unit(u.deg)
        ax.coords[1].set_format_unit(u.deg)
        img = ax.imshow(heatmap, vmin=0, vmax=1, origin='lower') # extent=[xe[0], xe[-1], ye[0], ye[-1]]
        #ax.invert_xaxis()
    else:
        ax = plt.subplot(aspect='equal')
        img = ax.imshow(heatmap, vmin=0, vmax=1, origin='lower')
        ax.invert_yaxis()
    if pnt is not None:
        try:
            ra, dec = pnt.ra.deg, pnt.dec.deg
        except:
            ra, dec = pnt[0], pnt[1]
        ax.scatter(np.array(ra), np.array(dec), marker='+', s=50, facecolor='none', edgecolor='r', transform=ax.get_transform(wcs))
    if src is not None:
        try:
            ra, dec = src.ra.deg, src.dec.deg
        except:
            ra, dec = src[0], src[1]
        ax.scatter(np.array(ra), np.array(dec), marker='o', s=50, facecolor='none', edgecolor='r', transform=ax.get_transform(wcs))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(img)
    if save and save_name is not None:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.close()
    return

# gather simulations and create heatmaps 
def process_dataset(ds1_dataset_path, ds2_dataset_path, saveas, trange, smoothing, binning, sample, save=False, output=None, norm_value=1, stretch=False):
    datapath = {'DS1': ds1_dataset_path, 'DS2': ds2_dataset_path}

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
            try:
                with fits.open(f) as h:
                    heatmap = smooth_heatmap(h[0].data, smoothing)
            except:
                heatmap = Table.read(f, hdu=1).to_pandas()
                # integrate exposure
                heatmap = extract_heatmap_from_table(heatmap, trange, smoothing, binning)

            # normalise map
            if norm_value == 1 and stretch:
                heatmap = stretch_smooth(heatmap, smoothing)
            elif norm_value == 1 and not stretch:
                heatmap = normalise_heatmap(heatmap)
            elif type(norm_value) == float and stretch:
                heatmap = stretch_min_max(heatmap, vmax=norm_value)
            elif type(norm_value) == float and not stretch:
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
        filename = join(output, saveas)
        np.save(filename, datasets, allow_pickle=True, fix_imports=True)
        print(f"Process complete: {filename}")
    return datasets

# gather simulations and create heatmaps for regressor
def process_regressor_dataset(ds_dataset_path, infotable, saveas, smoothing, binning, sample, save=False, output=None, stretch=True, norm_value=1, exposure=None):
    datapath = {'DS': ds_dataset_path}

    # datasets files
    datafiles = {'DS': []}
    datafiles['DS'] = sorted([join(datapath['DS'], f) for f in listdir(datapath['DS']) if '.fits' in f and isfile(join(datapath['DS'], f))])

    # get info data
    infodata = pd.read_csv(infotable, sep=' ', header=0).sort_values(by=['seed'])
        
    # create images dataset 
    datasets = {'DS': [], 'LABELS': [], 'SEED': [], 'SOURCE': [], 'FILE': [], 'EXPOSURE': []}
    print(f"Load DS data...")
    for f in tqdm(datafiles['DS'][:sample]):
        # get source coordinates
        seed = int(''.join(filter(str.isdigit, basename(f))))
        row = infodata[infodata['seed']==seed]
        # verify seed and file are correct 
        assert seed == row['seed'].values[0]
        assert row['name'].values[0] in f
        # set wcs
        w = set_wcs(point_ra=row['point_ra'].values[0], point_dec=row['point_dec'].values[0], 
                    point_ref=binning/2+0.5, pixelsize=row['fov'].values[0]/binning)
        x, y = w.world_to_pixel(SkyCoord(row['source_ra'].values[0], row['source_dec'].values[0], unit='deg', frame='icrs'))

        # load
        try:
            heatmap = Table.read(f, hdu=1).to_pandas()
            # integrate exposure
            if exposure is not None:
                if exposure == 'random':
                    exposure = np.random.randint(10, row['duration'].values[0])
                heatmap = extract_heatmap_from_table(heatmap, smoothing, binning, filter=True, trange=(0, exposure), wcs=w)
            else:
                heatmap = extract_heatmap_from_table(heatmap, smoothing, binning, wcs=w)
        except:
            with fits.open(f) as h:
                heatmap = smooth_heatmap(h[0].data, smoothing)

        # normalise map
        # normalise map
        if norm_value == 1 and stretch:
            heatmap = stretch_smooth(heatmap, smoothing)
        elif norm_value == 1 and not stretch:
            heatmap = normalise_heatmap(heatmap)
        elif type(norm_value) == float and stretch:
            heatmap = stretch_min_max(heatmap, vmax=norm_value)
        elif type(norm_value) == float and not stretch:
            heatmap = normalise_dataset(heatmap, max_value=norm_value)
        else:
            pass

        # add to dataset
        if heatmap.shape != (binning, binning):
            heatmap.reshape(binning, binning)

        # append to ds
        datasets['DS'].append(heatmap)
        datasets['LABELS'].append((x,y))
        datasets['SOURCE'].append((row['source_ra'].values[0], row['source_dec'].values[0]))
        datasets['SEED'].append(seed)
        datasets['EXPOSURE'].append(exposure)
        datasets['FILE'].append(f)

    # convert to numpy array
    datasets['DS'] = np.array(datasets['DS'])
    datasets['LABELS'] = np.array(datasets['LABELS'])
    
    # save processed dataset
    if save and output is not None:
        filename = join(output, saveas)
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
def split_regression_dataset(dataset, split=80, reshape=True, binning=250):
    total_size = len(dataset['DS'])
    train_size = int((total_size / 100) * split)
    # train dataset
    train_data = np.copy(dataset['DS'][:train_size])
    train_labels = np.copy(dataset['LABELS'][:train_size])
    train_seeds = np.copy(dataset['SEED'][:train_size])
    # test dataset
    test_data = np.copy(dataset['DS'][train_size:])
    test_labels = np.copy(dataset['LABELS'][train_size:])
    test_seeds = np.copy(dataset['SEED'][train_size:])
    print(train_data.shape, train_labels.shape)
    # reshape
    if reshape:
        train_data = train_data.reshape(train_data.shape[0], binning, binning, 1)
        test_data = test_data.reshape(test_data.shape[0], binning, binning, 1)  
    return train_data, train_labels, train_seeds, test_data, test_labels, test_seeds

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

def tensorboard_logdir(savename, logdate=True):
    logdir = join("logs", f"{savename}")
    if logdate:
        logdir += '_' + datetime.now().strftime("%Y%m%dT%H%M%S")
    return logdir

def checkpoint_dir(savename, logdate=False):
    logdir = join("checkpoints", f"{savename}")
    if logdate:
        logdir += '_' + datetime.now().strftime("%Y%m%dT%H%M%S")
    return logdir

def load_dataset_npy(path):
    ds = np.load(path, allow_pickle=True, encoding='latin1', fix_imports=True).flat[0]
    return ds