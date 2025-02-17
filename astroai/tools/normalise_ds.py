import sys
import pickle
import numpy as np
from astroai.tools.utils import normalise_heatmap

#fname = sys[1]
fname = '/home/dipiano/E4/irf_random/crab/cleaner_5sgm_expALL.pickle'
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
