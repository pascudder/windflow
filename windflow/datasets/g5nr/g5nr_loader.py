import os
import glob
import numpy as np
import xarray as xr
import logging
from datetime import timedelta
from tqdm import tqdm

import torch
from torch.utils import data

from windflow.datasets import flow_transforms
from windflow.datasets.preprocess import image_histogram_equalization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_timestamp(filename):
    try:
        basename = os.path.basename(filename)
        timestamp_str = basename.split('.')[0]  # Remove file extension
        return int(timestamp_str)
    except ValueError as e:
        logging.error(f"Error parsing timestamp from filename {filename}: {str(e)}")
        return None

class TileFlows(data.Dataset):
    def __init__(self, directory, scale_factor=None, size=None, 
                 augment=False, frames=2, convert_cartesian=False, time_step=10800000):  # time_step in milliseconds
        self.directory = directory
        self.files = np.array(sorted(glob.glob(os.path.join(self.directory, '*.nc4'))))
        
        self.timestamps = np.array([parse_timestamp(f) for f in self.files])
        
        valid_indices = self.timestamps != None
        self.timestamps = self.timestamps[valid_indices]
        self.files = self.files[valid_indices]
        
        self.size = size
        self.scale_factor = scale_factor
        self.augment = augment
        self.frames = frames
        self.convert_cartesian = convert_cartesian
        self.time_step = time_step  # in milliseconds
        
        self.successes = 0
        self.failures = 0
        
        # Pre-compute valid pairs
        self.valid_pairs = self._compute_valid_pairs()

        self.rand_transform = flow_transforms.Compose([
            flow_transforms.ToTensor(images_order='CHW', flows_order='CHW'),
            flow_transforms.RandomCrop((size, size)),
            flow_transforms.RandomHorizontalFlip(),
            flow_transforms.RandomVerticalFlip(),
        ])

    def _compute_valid_pairs(self):
        valid_pairs = []
        for i in range(len(self.timestamps) - 1):
            j = i + 1
            while j < len(self.timestamps):
                if abs(self.timestamps[j] - self.timestamps[i] - self.time_step) < 60000:  # 1-minute tolerance
                    valid_pairs.append((i, j))
                    break
                elif self.timestamps[j] - self.timestamps[i] > self.time_step + 60000:
                    break
                j += 1
        return valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        start_idx, end_idx = self.valid_pairs[idx]
        
        # Check if the time difference is actually 3 hours (with 1-minute tolerance)
        time_diff = self.timestamps[end_idx] - self.timestamps[start_idx]
        if abs(time_diff - self.time_step) < 60000:  # 1-minute tolerance
            self.successes += 1
            is_valid = True
        else:
            self.failures += 1
            is_valid = False

        if is_valid:
             ds = xr.open_mfdataset([self.files[start_idx], self.files[end_idx]])
             h, w = ds.U.isel(time=0).shape
             if self.scale_factor:
                 h, w = int(h * self.scale_factor), int(w * self.scale_factor)
                 new_lats = np.linspace(ds.lat.values[0], ds.lat.values[-1], h)
                 new_lons = np.linspace(ds.lon.values[0], ds.lon.values[-1], w)
                 ds = ds.interp(lat=new_lats, lon=new_lons)
             qv = ds['QV'].values
             u = ds['U'].values
             v = ds['V'].values
             uv = np.concatenate([u[:,np.newaxis], v[:,np.newaxis]], 1)
             uv[~np.isfinite(uv)] = 0. 
             qv[~np.isfinite(qv)] = 0.
             if self.convert_cartesian:
                 lat_rad = np.radians(ds.lat.values)
                 lon_rad = np.radians(ds.lon.values)
                 a = np.cos(lat_rad)**2 * np.sin((lon_rad[1]-lon_rad[0])/2)**2
                 d = 2 * 6378.137 * np.arcsin(a**0.5)
                 size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1) # kms
                 uv = uv / size_per_pixel / 1000 * (self.time_step / 1000)  # Convert time_step to seconds
             qv = image_histogram_equalization(qv)
             images = [q[np.newaxis] for q in qv]
             flows = [_uv for _uv in uv]
             images_tensor, flow_tensor = self.rand_transform(images, flows)
             return images_tensor, flow_tensor
        else:
             return None  # or some placeholder value


    def get_total_files(self):
        return len(self.files)

class G5NRFlows(data.ConcatDataset):
    def __init__(self, directory, scale_factor=None, size=None, augment=False, frames=2, convert_cartesian=True, time_step=10800000,max_samples=None):  # time_step in milliseconds
        self.directory = directory
        tiles = os.listdir(directory)
        tile_paths = [os.path.join(directory, t) for t in tiles]
        self.datasets = [TileFlows(p, scale_factor=scale_factor, size=size, augment=augment, frames=frames, convert_cartesian=convert_cartesian, time_step=time_step) for p in tile_paths]
        if max_samples is not None:
            self.datasets = self.datasets[:max_samples]
        data.ConcatDataset.__init__(self, self.datasets)

    def get_stats(self):
        total_successes = sum(dataset.successes for dataset in self.datasets)
        total_failures = sum(dataset.failures for dataset in self.datasets)
        total_files = sum(dataset.get_total_files() for dataset in self.datasets)
        return total_successes, total_failures, total_files

if __name__ == '__main__':
    dataset = G5NRFlows('/data/sreiner/windflow_patches/train', size=128, time_step=10800000)  # 3 hours in milliseconds
    logging.info(f"Total dataset length: {len(dataset)}")
    
    # Limit the number of samples to process
    num_samples_to_process = 1000  # Adjust this number as needed
    
    # Process limited number of samples with a progress bar
    successes = 0
    failures = 0
    for i in tqdm(range(min(num_samples_to_process, len(dataset))), desc="Processing samples"):
        try:
            result = dataset[i]
            if result[0] is not None:  # Assuming None is returned for invalid pairs
                successes += 1
            else:
                failures += 1
        except Exception as e:
            logging.error(f"Error processing sample {i}: {str(e)}")
            failures += 1
    
    logging.info(f"Processed samples: {num_samples_to_process}")
    logging.info(f"Successes (valid 3-hour intervals): {successes}")
    logging.info(f"Failures (invalid or errored samples): {failures}")
    if successes + failures > 0:
        success_rate = successes / (successes + failures) * 100
        logging.info(f"Success rate: {success_rate:.2f}%")