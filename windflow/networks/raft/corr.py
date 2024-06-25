import torch
import torch.nn.functional as F
from .utils.utils import trilinear_sampler, coords_grid_3d

class CorrBlock3D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock3D.corr(fmap1, fmap2)

        batch, d1, h1, w1, dim, d2, h2, w2 = corr.shape
        corr = corr.reshape(batch*d1*h1*w1, dim, d2, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool3d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 4, 1)
        batch, d1, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i].float()
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            dz = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dz, dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*d1*h1*w1, 1, 1, 1, 3) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2*r+1, 3)
            coords_lvl = centroid_lvl + delta_lvl

            corr = trilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, d1, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 4, 1, 2, 3).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, dt, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, dt*ht*wd)
        fmap2 = fmap2.view(batch, dim, dt*ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, dt, ht, wd, 1, dt, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())

class AlternateCorrBlock3D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool3d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool3d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 4, 1)
        B, D, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 4, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 4, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, D, H, W, 3).contiguous()
            corr = self.corr_func(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, D, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

    def corr_func(self, fmap1, fmap2, coords, r):
        # This function needs to be implemented for 3D correlation
        # You may need to write a custom CUDA kernel for efficient implementation
        raise NotImplementedError("3D correlation function not implemented")