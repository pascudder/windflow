import os, sys
import glob
import argparse
import time

# T.Rink, MacBook OSX doesn't support MPI
# from mpi4py import MPI

# MPI_COMM = MPI.COMM_WORLD
# MPI_RANK = MPI_COMM.Get_rank()
# MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = 0

# T.Rink, MacBook GPU (AMD Radeon Pro 5300M) doesn't support CUDA
# N_GPUS = 2

# rank_gpu = MPI_RANK % N_GPUS
# os.environ['CUDA_VISIBLE_DEVICES'] = f'{rank_gpu}'
os.environ['CUDA_VISIBLE_DEVICES'] = f'{0.0}'

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from windflow.datasets.g5nr import g5nr
from windflow.inference.inference_flows import FlowRunner
from windflow.datasets.utils import cartesian_to_speed_eco

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="raft", type=str)
# parser.add_argument("--checkpoint", default="models/raft-size_512/checkpoint.pth.tar", type=str)
parser.add_argument("--checkpoint", default="model_weights/windflow.raft.pth.tar", type=str)
parser.add_argument("--data_directory", default="data/ECO1280/", type=str)
parser.add_argument("--output", default="data/ECO_flows/raft/", type=str)
parser.add_argument("--n_files", default=None, type=int)
parser.add_argument("--batch_size", default=2, type=int)

args = parser.parse_args()

# Set model information
model_name = args.model_name

# Set data variables
data_directory = args.data_directory
to_directory = args.output

if (MPI_RANK == 0) and (not os.path.exists(to_directory)):
    os.makedirs(to_directory)

# get file list
#files = g5nr.get_files(data_directory, 'test')
#print("Number of OSSE test files", len(files))

pattern = os.path.join(data_directory, '*.nc4')
files = sorted(glob.glob(pattern))

variables = ['gp', 'uv']
var_files = {v: sorted([f for f in files if f'{v}_' in f]) for v in variables}
var_files = pd.DataFrame(var_files)

if args.n_files is None:
    N_files = len(files)
else:
    N_files = args.n_files

# N_per_rank = N_files // MPI_SIZE
# files = files.iloc[N_per_rank * MPI_RANK: N_per_rank * (MPI_RANK+1) + 1]

# load model
tile_size = 512
overlap = 128

if model_name in ['pwc-net-rmsvd', 'pwc-net-l1']:
    runner = FlowRunner('pwc-net', 
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=args.batch_size)
else:
    runner = FlowRunner(model_name.replace('-guided','').replace('-unflow', ''),
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=args.batch_size,
                        device=torch.device('cpu'))
 
runner.load_checkpoint(args.checkpoint)

stats = []

# iterate and perform inference, compute test statistics 
ds1 = xr.open_mfdataset(files.iloc[0].values, engine='netcdf4')
for i in range(1, files.shape[0]): 
    # open dataset
    
    ds2 = xr.open_mfdataset(files.iloc[i].values, engine='netcdf4')
 
    f = os.path.basename(files.iloc[i-1]['uv'])  # get ds1 file
    to_flow_file = os.path.join(to_directory, f.replace('uv_', '_WindFlow_').replace('.nc', '.zarr'))
    if os.path.exists(to_flow_file):
        ds1 = ds2.copy()
        continue 

    # t = ds1['time'].values

    output_ds = xr.zeros_like(ds1)
    del output_ds['gp_newP']

    U_flows = np.zeros(ds1.ugrd_newP.shape)
    V_flows = np.zeros(ds1.vgrd_newP.shape)

    t0 = time.time()
    
    for i, lev in enumerate(ds1.lev_p):
        qv1_lev = ds1.sel(lev_p=lev)['gp_newP'].values[0]
        qv2_lev = ds2.sel(lev_p=lev)['gp_newP'].values[0]
        _, flows_lev = runner.forward(qv1_lev, qv2_lev)
        U_flows[:, i] = flows_lev[0]
        V_flows[:, i] = flows_lev[1]

    output_ds['ugrd_newP'] = output_ds['ugrd_newP'] + U_flows
    output_ds['vgrd_newP'] = output_ds['vgrd_newP'] + V_flows

    output_ds = cartesian_to_speed_eco(output_ds)
 
    output_ds.attrs['Source'] = 'NEX'
    output_ds.attrs['Title'] = 'Optical Flow Feature Tracking'
    output_ds.attrs['Contact'] = 'vandal@baeri.org'
    output_ds.attrs['History'] = 'G5NR outputs from GEOS-5 by gmao processed by NEX optical flow.'
    output_ds.attrs['Model'] = model_name
    output_ds.attrs['Pytorch_Checkpoint'] = args.checkpoint

    # output_ds.to_netcdf(to_flow_file)
    output_ds.to_zarr(to_flow_file)
    # print(f'Wrote to file {to_flow_file}')
    print(f"Wrote to file: {to_flow_file} -- Processing time {time.time()-t0} (seconds)")

    ds1 = ds2.copy()
