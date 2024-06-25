import os, sys
import glob
import argparse
import time

from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

N_GPUS = 2

rank_gpu = MPI_RANK % N_GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = f'{rank_gpu}'

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from windflow.datasets.g5nr import g5nr
from windflow.inference.inference_flows import FlowRunner3D  # Assuming you've created a 3D version of FlowRunner
from windflow.datasets.utils import cartesian_to_speed_3d  # Assuming you've created a 3D version of this function

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="raft3d", type=str)  # Updated to use 3D RAFT model
parser.add_argument("--checkpoint", default="models/raft3d-size_512/checkpoint.pth.tar", type=str)
parser.add_argument("--data_directory", default="data/G5NR/", type=str)
parser.add_argument("--output", default="data/G5NR_Flows/raft3d/", type=str)
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
files = g5nr.get_files(data_directory, 'test')
print("Number of OSSE test files", len(files))

if args.n_files is None:
    N_files = len(files)
else:
    N_files = args.n_files

N_per_rank = N_files // MPI_SIZE
files = files.iloc[N_per_rank * MPI_RANK: N_per_rank * (MPI_RANK+1) + 1]

# load model
tile_size = 128  # Adjust this based on your 3D model's input size
overlap = 32

runner = FlowRunner3D(model_name,
                      tile_size=tile_size,
                      overlap=overlap,
                      batch_size=args.batch_size)
 
runner.load_checkpoint(args.checkpoint)

stats = []

# iterate and perform inference, compute test statistics 
ds1 = xr.open_mfdataset(files.iloc[0].values, engine='netcdf4')
for i in range(1, files.shape[0]): 
    ds2 = xr.open_mfdataset(files.iloc[i].values, engine='netcdf4')
 
    f = os.path.basename(files.iloc[i-1]['U']) # get ds1 file
    to_flow_file = os.path.join(to_directory, f.replace('_U_', '_WindFlow3D_').replace('.nc', '.zarr'))
    if os.path.exists(to_flow_file):
        ds1 = ds2.copy()
        continue 

    t = ds1['time'].values

    output_ds = xr.zeros_like(ds1)
    del output_ds['QV'], output_ds['tpw']

    U_flows = np.zeros(ds1.U.shape)
    V_flows = np.zeros(ds1.V.shape)
    W_flows = np.zeros(ds1.U.shape)  # Add vertical component

    t0 = time.time()
    
    # Process entire 3D volume at once
    qv1 = ds1['QV'].values[0]
    qv2 = ds2['QV'].values[0]
    _, flows_3d = runner.forward(qv1, qv2)
    U_flows[0] = flows_3d[0]
    V_flows[0] = flows_3d[1]
    W_flows[0] = flows_3d[2]  # Add vertical component

    output_ds['U'] = output_ds['U'] + U_flows
    output_ds['V'] = output_ds['V'] + V_flows
    output_ds['W'] = W_flows  # Add vertical component to output

    output_ds = cartesian_to_speed_3d(output_ds)  # Convert to 3D speed
 
    output_ds.attrs['Source'] = 'NEX'
    output_ds.attrs['Title'] = '3D Optical Flow Feature Tracking'
    output_ds.attrs['Contact'] = 'vandal@baeri.org'
    output_ds.attrs['History'] = 'G5NR outputs from GEOS-5 by gmao processed by NEX 3D optical flow.'
    output_ds.attrs['Model'] = model_name
    output_ds.attrs['Pytorch_Checkpoint'] = args.checkpoint

    output_ds.to_zarr(to_flow_file)
    print(f"Wrote to file: {to_flow_file} -- Processing time {time.time()-t0} (seconds)")

    ds1 = ds2.copy()