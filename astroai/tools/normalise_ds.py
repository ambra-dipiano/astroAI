import pickle
import argparse
from astroai.tools.utils import normalise_heatmap
from os.path import expandvars

def main(filename):
    fname = expandvars(filename)
    with open(fname,'rb') as f: ds = pickle.load(f)

    for i, m in enumerate(ds['DS1']):
        m = normalise_heatmap(m)
        ds['DS1'][i] = m
    print('DS1 done')
    for i, m in enumerate(ds['DS2']):
        m = normalise_heatmap(m)
        ds['DS2'][i] = m
    print('DS2 done')    

    with open(fname,'wb') as f: pickle.dump(ds, f, protocol=4)
    print('End')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True, help='Configuration YAML file')
    args = parser.parse_args()

    main(args.filename)

