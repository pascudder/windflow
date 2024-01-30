from scipy.interpolate import RegularGridInterpolator

import numpy as np
from netCDF4 import Dataset

# resample methods:
linear = 'linear'
cubic = 'cubic'
nearest = 'nearest'

eco_x = np.linspace(0.0, 2.0, 3600)
eco_y = np.linspace(0.0, 1.0, 1801)
g5_x = np.linspace(0.0, 2.0, 5760)
g5_y = np.linspace(0.0, 1.0, 2881)

g5_scale_x = 5760 / 3600
g5_scale_y = 2881 / 1801


def time_linear_interp_grids(grid_a, grid_b, time_a, time_b, time_i):
    grd_shape = grid_a.shape

    grid_a = grid_a.flatten()
    grid_b = grid_b.flatten()

    del_time = time_b - time_a

    w_a = 1.0 - (time_i - time_a) / del_time
    w_b = (time_i - time_a) / del_time

    grid_i = w_a * grid_a + w_b * grid_b
    grid_i = np.reshape(grid_i, grd_shape)

    return grid_i


def upsample(scalar_field, ej_a=0, ej_b=1801, ei_a=0, ei_b=3600):
    gj_a = int(g5_scale_y * ej_a)
    gj_b = int(g5_scale_y * ej_b)
    gi_a = int(g5_scale_x * ei_a)
    gi_b = int(g5_scale_x * ei_b)

    # match N-S orientation to GEOS5-NR
    scalar_field = scalar_field[::-1, :]

    scalar_field = scalar_field[ej_a:ej_b, ei_a:ei_b]
    print('input dims: ', scalar_field.shape[0], scalar_field.shape[1])

    intrp = RegularGridInterpolator((eco_y[ej_a:ej_b], eco_x[ei_a:ei_b]), scalar_field, method=linear, bounds_error=False)

    g5_y_s = g5_y[gj_a:gj_b]
    g5_x_s = g5_x[gi_a:gi_b]
    print('output dims: ', g5_y_s.shape[0], g5_x_s.shape[0])

    xg, yg = np.meshgrid(g5_x_s, g5_y_s, indexing='xy')
    yg, xg = yg.flatten(), xg.flatten()
    pts = np.array([yg, xg])
    t_pts = np.transpose(pts)

    return np.reshape(intrp(t_pts), (g5_y_s.shape[0], g5_x_s.shape[0]))


def test(eco_file, g5_file, ej_a=0, ej_b=1801, ei_a=0, ei_b=3600):
    rtgrp = Dataset(eco_file, 'r', format='NETCDF3')
    gp_var = rtgrp['gp_newP']
    eco_gp = gp_var[:, :, 0].copy()
    print('gp_newP range/shape: ', np.nanmin(eco_gp), np.nanmax(eco_gp), eco_gp.shape)

    rsm_img = upsample(eco_gp, ej_a=ej_a, ej_b=ej_b, ei_a=ei_a, ei_b=ei_b)

    return rsm_img, eco_gp




