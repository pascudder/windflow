import xarray as xr
import numpy as np
import cv2
import scipy.interpolate

def fillmiss(x):
    if x.ndim != 2:
        raise ValueError("X have only 2 dimensions.")
    mask = ~np.isnan(x)
    xx, yy = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    data0 = np.ravel(x[mask])
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0

def interp_dim(x, scale):
    x0, xlast = x[0], x[-1]
    newlength = int(len(x) * scale)
    y = np.linspace(x0, xlast, num=newlength, endpoint=False)
    return y

def interp_tensor4d(X, scale, fill=True, how=cv2.INTER_NEAREST):
    nld = int(X.shape[1]*scale[0])
    nlt = int(X.shape[2]*scale[1])
    nln = int(X.shape[3]*scale[2])
    newshape = (X.shape[0], nld, nlt, nln)
    scaled_tensor = np.empty(newshape)
    for j, vol in enumerate(X):
        if fill:
            vol[np.isnan(vol)] = 0
        scaled_vol = np.empty((nld, nlt, nln))
        for k, im in enumerate(vol):
            scaled_vol[k] = cv2.resize(im, (newshape[3], newshape[2]), interpolation=how)
        scaled_tensor[j] = scaled_vol
    return scaled_tensor

def interp_da(da, scale, how=cv2.INTER_LINEAR):
    """
    Assume da is of dimensions ('time','lat', 'lon')
    """
    tensor = da.values

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[1]].values, scale)
    lonnew = interp_dim(da[da.dims[2]].values, scale)

    # lets store our interpolated data
    scaled_tensor = interp_tensor(tensor, scale, fill=True, how=how)

    if latnew.shape[0] != scaled_tensor.shape[1]:
        raise ValueError("New shape is shitty")
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[da[da.dims[0]].values, latnew, lonnew],
                 dims=da.dims)

def interp_da2d(da, scale, fillna=False, how=cv2.INTER_NEAREST):
    """
    Assume da is of dimensions ('time','lat', 'lon')
    """
    # lets store our interpolated data
    newshape = (int(da.shape[0]*scale),int(da.shape[1]*scale))
    im = da.values
    scaled_tensor = np.empty(newshape)
    # fill im with nearest neighbor
    if fillna:
        filled = fillmiss(im)
    else:
        filled = im
    scaled_tensor = cv2.resize(filled, dsize=(0,0), fx=scale, fy=scale,
                              interpolation=how)

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[1]].values, scale)
    lonnew = interp_dim(da[da.dims[0]].values, scale)

    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[lonnew, latnew],
                 dims=da.dims)

def blocks4d(data, width=(64, 128, 128)):
    d = data.depth.shape[0]
    h = data.lat.shape[0]
    w = data.lon.shape[0]

    ds = np.arange(0, d, width[0])
    hs = np.arange(0, h, width[1])
    ws = np.arange(0, w, width[2])
    blocks = []
    for dindex in ds:
        if dindex + width[0] > d:
            dindex = d - width[0]
        for hindex in hs:
            if hindex + width[1] > h:
                hindex = h - width[1]
            for windex in ws:
                if windex + width[2] > w:
                    windex = w - width[2]
                blocks.append(data.sel(depth=data.depth.values[dindex:dindex+width[0]],
                                       lat=data.lat.values[hindex:hindex+width[1]],
                                       lon=data.lon.values[windex:windex+width[2]]))
    return blocks

def fillmiss3d(x):
    if x.ndim != 3:
        raise ValueError("X must have 3 dimensions.")
    mask = ~np.isnan(x)
    xx, yy, zz = np.meshgrid(np.arange(x.shape[2]), np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]), np.ravel(zz[mask]))).T
    data0 = np.ravel(x[mask])
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy), np.ravel(zz)).reshape(x.shape)
    return result0

def interp_da3d(da, scale, how=cv2.INTER_LINEAR):
    """
    Assume da is of dimensions ('time', 'depth', 'lat', 'lon')
    """
    tensor = da.values

    # interpolate depth, lat and lons
    depthnew = interp_dim(da[da.dims[1]].values, scale[0])
    latnew = interp_dim(da[da.dims[2]].values, scale[1])
    lonnew = interp_dim(da[da.dims[3]].values, scale[2])

    scaled_tensor = interp_tensor4d(tensor, scale, fill=True, how=how)

    return xr.DataArray(scaled_tensor, coords=[da[da.dims[0]].values, depthnew, latnew, lonnew],
                        dims=da.dims)
