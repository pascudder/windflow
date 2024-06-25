import numpy as np
import torch
import torch.nn as nn
import scipy.stats

def split_array_3d(arr, tile_size, overlap):
    c, d, h, w = arr.shape
    patches = []
    indices = []
    for i in range(0, d, tile_size[0]-overlap[0]):
        for j in range(0, h, tile_size[1]-overlap[1]):
            for k in range(0, w, tile_size[2]-overlap[2]):
                i = min(i, d - tile_size[0])
                j = min(j, h - tile_size[1])
                k = min(k, w - tile_size[2])
                indices.append([i, j, k])
                patches.append(arr[np.newaxis,:,i:i+tile_size[0], j:j+tile_size[1], k:k+tile_size[2]])
    indices = np.array(indices)
    patches = np.concatenate(patches, axis=0)
    return patches, indices

def reassemble_split_array_3d(arr, upperleft, shape, trim=0):
    assert len(arr) > 0

    depth, height, width = arr.shape[2:5]
    counter = np.zeros(shape)
    out_sum = np.zeros(shape)
    for i, x in enumerate(arr):
        iz, iy, ix = upperleft[i]
        if trim > 0:
            counter[:,iz+trim:iz+depth-trim,iy+trim:iy+height-trim,ix+trim:ix+width-trim] += 1
            out_sum[:,iz+trim:iz+depth-trim,iy+trim:iy+height-trim,ix+trim:ix+width-trim] += x[:,trim:-trim,trim:-trim,trim:-trim]
        else:
            counter[:,iz:iz+depth,iy:iy+height,ix:ix+width] += 1
            out_sum[:,iz:iz+depth,iy:iy+height,ix:ix+width] += x

    out = out_sum / counter
    return out

def reassemble_with_3d_gaussian(arr, upperleft, shape, trim=0):
    assert len(arr) > 0

    depth, height, width = arr.shape[2:5]
    counter = np.zeros(shape)
    out_sum = np.zeros(shape)
    
    nsig = 3
    x = np.linspace(-nsig, nsig, depth+1-trim*2)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern3d = np.outer(np.outer(kern1d, kern1d), kern1d).reshape((depth+1-trim*2, height+1-trim*2, width+1-trim*2))
    pdf = kern3d/kern3d.sum()
    
    for i, x in enumerate(arr):
        iz, iy, ix = upperleft[i]
        if trim > 0:
            counter[:,iz+trim:iz+depth-trim,iy+trim:iy+height-trim,ix+trim:ix+width-trim] += pdf
            out_sum[:,iz+trim:iz+depth-trim,iy+trim:iy+height-trim,ix+trim:ix+width-trim] += x[:,trim:-trim,trim:-trim,trim:-trim] * pdf
        else:
            counter[:,iz:iz+depth,iy:iy+height,ix:ix+width] += pdf
            out_sum[:,iz:iz+depth,iy:iy+height,ix:ix+width] += x * pdf

    out = out_sum / counter
    return out

def inference_3d(model, X0, X1, 
                 tile_size=(64, 128, 128), 
                 overlap=(16, 32, 32), 
                 upsample_flow_factor=None,
                 batch_size=4):
    c, d, h, w = X0.shape
    trim = 0  # overlap[0] // 4
    
    if isinstance(upsample_flow_factor, int):
        upsample_flow = nn.Upsample(scale_factor=upsample_flow_factor, mode='trilinear')
    else:
        upsample_flow = None
        
    f_sum = np.zeros((1, 3, d, h, w))
    f_counter = np.zeros((1, 1, d, h, w))
    
    x0_patches, upperleft = split_array_3d(X0, tile_size, overlap)
    x1_patches, _ = split_array_3d(X1, tile_size, overlap)
    
    x0_patches = torch.from_numpy(x0_patches).float()
    x1_patches = torch.from_numpy(x1_patches).float()
    pred = []
    for batch in range(0, x1_patches.shape[0], batch_size):
        x0_batch = x0_patches[batch:batch+batch_size].cuda()
        x1_batch = x1_patches[batch:batch+batch_size].cuda()
        
        model_out = model(x0_batch, x1_batch, test_mode=True)[0]
            
        if upsample_flow:
            model_out = upsample_flow(model_out)
            
        pred.append(model_out.cpu().detach().numpy())

    pred = np.concatenate(pred, 0)   
    UVW = reassemble_with_3d_gaussian(pred, upperleft, (3, d, h, w), trim=trim)

    return UVW

class FlowRunner3D(object):
    def __init__(self, model_name, 
                 tile_size=(64, 128, 128), 
                 overlap=(16, 32, 32), 
                 batch_size=4, 
                 upsample_input=None,
                 device='cuda:0'):        
        self.model = get_flow_model_3d(model_name, small=False)  # Assuming you have a 3D model getter
        self.model_name = model_name
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.overlap = overlap
        
        if self.model_name.lower() in ['flownets3d', 'pwcnet3d', 'pwc-net3d', 'maskflownet3d']:
            self.upsample_flow_factor = 4
        else:
            self.upsample_flow_factor = None
            
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)

    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.global_step = checkpoint['global_step']
        try:
            self.model.module.load_state_dict(checkpoint['model'])
        except:
            self.model.load_state_dict(checkpoint['model'])
                
        print(f"Loaded checkpoint: {checkpoint_file}")

    def preprocess(self, x):
        x[~np.isfinite(x)] = 0.
        x = image_histogram_equalization(x)
        return x    

    def forward(self, img1, img2):
        mask = (img1 == img1)
        mask[~mask] = np.nan

        img1_norm = self.preprocess(img1)
        img2_norm = self.preprocess(img2)

        flows = inference_3d(self.model, img1_norm[np.newaxis], img2_norm[np.newaxis], 
                             tile_size=self.tile_size, overlap=self.overlap, 
                             upsample_flow_factor=self.upsample_flow_factor,
                             batch_size=self.batch_size)
        return img1, flows * mask[np.newaxis]