import xarray as xr
import numpy as np
import os
import datetime as dt

def get_filename_metadata(f):
    #gp_2016-06-01_00:00:00_P500.nc
    f = os.path.basename(f)
    t1 = f.split('_')
    year = int(t1[1][0:4])
    month = int(t1[1][5:7])
    day = int(t1[1][8:10])
    hour = int(t1[2][0:2])
    minute = int(t1[2][3:5])
    second = int(t1[2][6:8])

    datetime = dt.datetime(year, month, day, hour, minute, second)
    return datetime


class Eco(object):
    '''
    manipulate and read ECO1280 grids
    '''
    def __init__(self, fpath):
        meta = get_filename_metadata(fpath)
        self.fpath = fpath
        self.datetime = meta

    def open_dataset(self, scale_factor=1, rescale=True, force=False, chunks=None):
        if not hasattr(self, 'data'):
            ds = xr.open_dataset(self.fpath, chunks=chunks)

            ds_rad = ds['gp_newP']
            
            if rescale:
                ds_rad = ds_rad * scale_factor

            ds['gp_newP'] = ds_rad
            self.data = ds
        return self.data
