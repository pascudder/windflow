import numpy as np
import torch
import torch.nn.functional as F

def split_array_3d(arr, tile_size=128, overlap=16):
    '''
    Split a 4D numpy array into patches for inference
    (Channels, Depth, Height, Width)
    Args:
        tile_size: depth, width, and height of patches to return
        overlap: number of voxels to overlap between patches
    Returns:
        dict(patches, upper_left): patches and indices of original array
    '''
    arr = arr[np.newaxis]
    depth, width, height = arr.shape[2:5]
    arrs = dict(patches=[], upper_left=[])
    for i in range(0, depth, tile_size - overlap):
        for j in range(0, width, tile_size - overlap):
            for k in range(0, height, tile_size - overlap):
                i = min(i, depth - tile_size)
                j = min(j, width - tile_size)
                k = min(k, height - tile_size)
                arrs['patches'].append(arr[:,:,i:i+tile_size,j:j+tile_size,k:k+tile_size])
                arrs['upper_left'].append([[i,j,k]])
    arrs['patches'] = np.concatenate(arrs['patches'])
    arrs['upper_left'] = np.concatenate(arrs['upper_left'])
    return arrs['patches'], arrs['upper_left']

def coords_grid_3d(batch, d, h, w, device):
    coords = torch.meshgrid(torch.arange(d, device=device),
                            torch.arange(h, device=device),
                            torch.arange(w, device=device))
    coords = torch.stack(coords[::-1], dim=-1).float()
    return coords[None].repeat(batch, 1, 1, 1, 1)

def trilinear_sampler(img, coords):
    d, h, w = img.shape[-3:]
    coords = coords.clamp(-1, 1)
    
    z = coords[..., 2]
    y = coords[..., 1]
    x = coords[..., 0]

    z_floor = torch.floor(z)
    y_floor = torch.floor(y)
    x_floor = torch.floor(x)
    z_ceil = z_floor + 1
    y_ceil = y_floor + 1
    x_ceil = x_floor + 1

    z_floor = torch.clamp(z_floor, 0, d-1)
    y_floor = torch.clamp(y_floor, 0, h-1)
    x_floor = torch.clamp(x_floor, 0, w-1)
    z_ceil = torch.clamp(z_ceil, 0, d-1)
    y_ceil = torch.clamp(y_ceil, 0, h-1)
    x_ceil = torch.clamp(x_ceil, 0, w-1)

    c000 = img[..., z_floor.long(), y_floor.long(), x_floor.long()]
    c001 = img[..., z_floor.long(), y_floor.long(), x_ceil.long()]
    c010 = img[..., z_floor.long(), y_ceil.long(), x_floor.long()]
    c011 = img[..., z_floor.long(), y_ceil.long(), x_ceil.long()]
    c100 = img[..., z_ceil.long(), y_floor.long(), x_floor.long()]
    c101 = img[..., z_ceil.long(), y_floor.long(), x_ceil.long()]
    c110 = img[..., z_ceil.long(), y_ceil.long(), x_floor.long()]
    c111 = img[..., z_ceil.long(), y_ceil.long(), x_ceil.long()]

    xd = x - x_floor
    yd = y - y_floor
    zd = z - z_floor
    xm = 1 - xd
    ym = 1 - yd
    zm = 1 - zd

    c00 = c000*xm + c001*xd
    c01 = c010*xm + c011*xd
    c10 = c100*xm + c101*xd
    c11 = c110*xm + c111*xd

    c0 = c00*ym + c01*yd
    c1 = c10*ym + c11*yd

    c = c0*zm + c1*zd

    return c

def upflow8_3d(flow, mode='trilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3], 8 * flow.shape[4])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)