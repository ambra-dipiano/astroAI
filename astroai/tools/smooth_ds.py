import sys
import pickle
from astroai.tools.utils import smooth_heatmap

fname = sys[1]
with open(fname,'rb') as f: ds = pickle.load(f)

for i, m in enumerate(ds['DS1']):
    m = smooth_heatmap(m, 5)
    ds['DS1'][i] = m
print('DS1 done')
for i, m in enumerate(ds['DS2']):
    m = smooth_heatmap(m, 5)
    ds['DS2'][i] = m
print('DS2 done')    

with open(fname,'wb') as f: pickle.dump(ds, f, protocol=4)
print('End')
