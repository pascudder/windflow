import numpy as np
import torch
import torch.nn as nn
import xarray as xr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def split_array_3d(arr, tile_size=(64, 128, 128), overlap=(16, 32, 32)):
    '''
    Split a 4D numpy array into patches for inference
    (Channels, Depth, Height, Width)
    '''
    arr = arr[np.newaxis]
    depth, height, width = arr.shape[2:5]
    arrs = dict(patches=[], upper_left=[])
    for i in range(0, depth, tile_size[0] - overlap[0]):
        for j in range(0, height, tile_size[1] - overlap[1]):
            for k in range(0, width, tile_size[2] - overlap[2]):
                i = min(i, depth - tile_size[0])
                j = min(j, height - tile_size[1])
                k = min(k, width - tile_size[2])
                arrs['patches'].append(arr[:,:,i:i+tile_size[0],j:j+tile_size[1],k:k+tile_size[2]])
                arrs['upper_left'].append([[i,j,k]])
    arrs['patches'] = np.concatenate(arrs['patches'])
    arrs['upper_left'] = np.concatenate(arrs['upper_left'])
    return arrs['patches'], arrs['upper_left']

def single_inference_3d(X0, X1, t, flownet, interpnet, multivariate):
    X0_arr_torch = torch.from_numpy(X0)
    X1_arr_torch = torch.from_numpy(X1)

    X0_arr_torch[np.isnan(X0_arr_torch)] = 0.
    X1_arr_torch[np.isnan(X1_arr_torch)] = 0.

    X0_arr_torch = torch.unsqueeze(X0_arr_torch, 0).to(device, dtype=torch.float)
    X1_arr_torch = torch.unsqueeze(X1_arr_torch, 0).to(device, dtype=torch.float)

    f = flownet(X0_arr_torch, X1_arr_torch)
    n_channels = X0_arr_torch.shape[1]

    if multivariate:
        f_01 = f[:,:3*n_channels]
        f_10 = f[:,3*n_channels:]
    else:
        f_01 = f[:,:3]
        f_10 = f[:,3:]

    I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1 = interpnet(X0_arr_torch, X1_arr_torch, f_01, f_10, t)
    
    out = {'f_01': f_01, 'f_10': f_10, 'I_t': I_t,
           'V_t0': V_t0, 'V_t1': V_t1, 'delta_f_t0': delta_f_t0,
           'delta_f_t1': delta_f_t0}

    for k in out.keys():
        out[k] = out[k][0].cpu().detach().numpy()

    return out

def single_inference_split_3d(X0, X1, t, flownet, interpnet,
                              multivariate, block_size=(64, 128, 128), overlap=(16, 32, 32),
                              discard=0):
    X0_split, upper_left_idxs = split_array_3d(X0, block_size, overlap)
    X1_split, _ = split_array_3d(X1, block_size, overlap)

    assert len(X0_split) > 0

    depth, height, width = X0.shape[1:4]
    counter = np.zeros((1, depth-discard*2, height-discard*2, width-discard*2))
    res_sum = {}
    for i, (x0, x1) in enumerate(zip(X0_split, X1_split)):
        iz, ix, iy = upper_left_idxs[i]
        res_i = single_inference_3d(x0, x1, t, flownet, interpnet, multivariate)
        keys = res_i.keys()
        if i == 0:
            res_sum = {k: np.zeros((res_i[k].shape[0], depth-discard*2, height-discard*2, width-discard*2)) for k in keys}

        for var in keys:
            if discard > 0:
                res_i[var] = res_i[var][:,discard:-discard,discard:-discard,discard:-discard]
            res_sum[var][:,iz:iz+block_size[0]-discard*2,ix:ix+block_size[1]-discard*2,iy:iy+block_size[2]-discard*2] += res_i[var]
            counter[:,iz:iz+block_size[0]-discard*2,ix:ix+block_size[1]-discard*2,iy:iy+block_size[2]-discard*2] += 1.

    out = {}
    for var in res_sum.keys():
       out[var] = res_sum[var] / counter

    return out

def _inference_3d(X0, X1, flownet, interpnet, warper,
                  multivariate, T=4, block_size=None):
    X0_arr_torch = torch.from_numpy(X0.values)
    X1_arr_torch = torch.from_numpy(X1.values)

    X0_arr_torch[np.isnan(X0_arr_torch)] = 0.
    X1_arr_torch[np.isnan(X1_arr_torch)] = 0.

    X0_arr_torch = torch.unsqueeze(X0_arr_torch, 0).to(device, dtype=torch.float)
    X1_arr_torch = torch.unsqueeze(X1_arr_torch, 0).to(device, dtype=torch.float)

    f = flownet(X0_arr_torch, X1_arr_torch)
    n_channels = X0_arr_torch.shape[1]

    if multivariate:
        f_01 = f[:,:3*n_channels]
        f_10 = f[:,3*n_channels:]
    else:
        f_01 = f[:,:3]
        f_10 = f[:,3:]

    predicted_frames = []

    for j in range(1,T+1):
        t = 1. * j / (T+1)
        I_t, g0, g1, V_t0, V_t1, delta_f_t0, delta_f_t1 = interpnet(X0_arr_torch, X1_arr_torch, f_01, f_10, t)
        predicted_frames.append(I_t.cpu().detach().numpy())

    torch.cuda.empty_cache()
    return predicted_frames

def inference_3d(X0, X1, flownet, interpnet, warper,
                 multivariate, T=4, block_size=(64, 128, 128)):
    X0_blocks = blocks_3d(X0, width=block_size)
    X1_blocks = blocks_3d(X1, width=block_size)

    interpolated_blocks = []
    for x0, x1 in zip(X0_blocks, X1_blocks):
        predicted_frames = _inference_3d(x0, x1, flownet,
                                         interpnet, warper,
                                         multivariate, T)
        predicted_frames = [x0.values[np.newaxis]] + predicted_frames + [x1.values[np.newaxis]]
        interpolated_blocks += [block_predictions_to_dataarray_3d(predicted_frames, x0)]
    return merge_and_average_dataarrays_3d(interpolated_blocks)

def blocks_3d(data, width):
    """
    Split 3D xarray DataArray into blocks.
    
    Args:
    data (xarray.DataArray): Input 3D data
    width (tuple): Block size (depth, height, width)
    
    Returns:
    list: List of DataArray blocks
    """
    depth, height, width = data.shape[1:]
    d_step, h_step, w_step = width
    
    blocks = []
    for d in range(0, depth, d_step):
        for h in range(0, height, h_step):
            for w in range(0, width, w_step):
                block = data.isel(
                    z=slice(d, min(d+d_step, depth)),
                    y=slice(h, min(h+h_step, height)),
                    x=slice(w, min(w+w_step, width))
                )
                blocks.append(block)
    return blocks

def block_predictions_to_dataarray_3d(predictions, block):
    """
    Convert block predictions to xarray DataArray.
    
    Args:
    predictions (list): List of numpy arrays with predictions
    block (xarray.DataArray): Original data block for coordinate reference
    
    Returns:
    xarray.DataArray: DataArray with predictions
    """
    block_predictions = np.concatenate(predictions, 0)
    block_predictions = np.clip(block_predictions, 0, 1)

    N_pred = block_predictions.shape[0]
    T = np.arange(0, N_pred)
    da = xr.DataArray(
        block_predictions,
        coords=[
            T,
            block.z,
            block.y,
            block.x
        ],
        dims=['t', 'z', 'y', 'x']
    )
    return da

def merge_and_average_dataarrays_3d(dataarrays):
    """
    Merge and average multiple 3D DataArrays.
    
    Args:
    dataarrays (list): List of xarray DataArrays
    
    Returns:
    xarray.DataArray: Merged and averaged DataArray
    """
    ds = xr.merge([xr.Dataset({f'block_{i}': d}) for i, d in enumerate(dataarrays)])
    das = [ds[var] for var in ds.data_vars]
    return xr.concat(das, dim='block').mean('block')