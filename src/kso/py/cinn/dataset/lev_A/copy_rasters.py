import os

print(os.getcwd())

import gzip
from shutil import copyfileobj
# from kso.py.tools.find_files import find_files
from ....tools.find_files import find_files

db_path = '/exports/fi1/IRIS/archive/level2/2013'
# db_path = '/exports/fi1/IRIS/archive/level2/2014'
# db_path = '/exports/fi1/IRIS/archive/level2/2015'
# db_path = '/exports/fi1/IRIS/archive/level2/2016'
# db_path = '/exports/fi1/IRIS/archive/level2/2017'
# db_path = '/exports/fi1/IRIS/archive/level2/2018'
pattern = '*raster*'

save_path = '/mnt/roy/IRIS_level2'


[roots, names] = find_files(db_path, pattern)

for root, name in zip(roots, names):

    o_path = os.path.join(root, name)   # old path
    n_path = os.path.join(save_path, name)

    print(o_path)

    with gzip.open(o_path, 'rb') as f_in:
        with open(n_path, 'wb') as f_out:
            copyfileobj(f_in, f_out)