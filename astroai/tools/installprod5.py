# *****************************************************************************
# Copyright (C) 2021 INAF
# This software was provided as IKC to the Cherenkov Telescope Array Observatory
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
#
#    Ambra Di Piano <ambra.dipiano@inaf.it>
#    Nicolò Parmiggiani <nicolo.parmiggiani@inaf.it>
#    Andrea Bulgarelli <andrea.bulgarelli@inaf.it>
#    Valentina Fioretti <valentina.fioretti@inaf.it>
#    Leonardo Baroncelli <leonardo.baroncelli@inaf.it>
#    Antonio Addis <antonio.addis@inaf.it>
#    Giovanni De Cesare <giovanni.decesare@inaf.it>
#    Gabriele Panebianco <gabriele.panebianco3@unibo.it>
# *****************************************************************************

import argparse
from os.path import join, abspath, isfile
from os import listdir, system

parser = argparse.ArgumentParser(description='')
parser.add_argument('-p', '--path', type=str, default='.', help="path to the caldb installation")
args = parser.parse_args()

print('Download FITS')
if not isfile("cta-prod5-zenodo-fitsonly-v0.1.zip"):
    system('wget https://zenodo.org/record/5499840/files/cta-prod5-zenodo-fitsonly-v0.1.zip')
system('unzip cta-prod5-zenodo-fitsonly-v0.1.zip')
system('rm -rf figures')

print('Extract TAR')
tars = [abspath(join('fits', f)) for f in listdir('fits') if '.FITS.tar.gz' in f]
for tar in tars:
    print(tar)
    system(f'tar -xvf {tar}')

print('Extract GZ')
gzs = [abspath(f) for f in listdir('.') if '.fits.gz' in f]
for gz in gzs:
    print(gz)
    system(f'gunzip {gz}')

print('Copy FITS in PROD5')
system('mkdir -p caldb/data/cta/prod5')
system('mv *.fits caldb/data/cta/prod5/.')

print('Remove duplicates')
system('rm -rf fits')

print('Delete archive')
system('rm cta-prod5-zenodo-fitsonly-v0.1.zip')
system('rm Website.md')

print('Move LICENSE and README')
system('mv LICENSE caldb/')
system('mv README.md caldb/')

if args.path != '.':
    print('Move CALDB to destination')
    system(f"mv caldb {abspath(args.path)}")
