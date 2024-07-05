import os
import glob
import numpy as np
import xarray as xr
from tqdm import tqdm

import torch
from torch.utils import data

from windflow.datasets import flow_transforms
from windflow.datasets.preprocess import image_histogram_equalization

def parse_timestamp(filename):
    try:
        basename = os.path.basename(filename)
        timestamp_str = basename.split('.')[0]  # Remove file extension
        return int(timestamp_str.lstrip('0'))  # Remove leading zeros and convert to int
    except ValueError:
        return None

class TileFlows(data.Dataset):
    def __init__(self, directory, common_timestamps, scale_factor=None, size=None, 
                 augment=False, frames=2, convert_cartesian=False, time_step=10800000):
        
        self.directory = directory
        print(f"Initializing TileFlows for directory: {directory}")
        
        self.files = np.array(glob.glob(os.path.join(self.directory, '*.nc4')))
        print(f"Found {len(self.files)} .nc4 files in the directory")
        
        filenames = np.array([parse_timestamp(f) for f in self.files])
        
        # Filter out None values resulting from parse errors
        valid_indices = [i for i, ts in enumerate(filenames) if ts is not None]
        self.files = self.files[valid_indices]
        filenames = filenames[valid_indices]
        
        print(f"After filtering invalid timestamps, {len(self.files)} files remain")
        
        filesort = np.argsort(filenames)
        self.files = self.files[filesort]
        self.timestamps = filenames[filesort]

        # Filter files based on common timestamps
        valid_indices = [i for i, ts in enumerate(self.timestamps) if ts in common_timestamps]
        self.timestamps = np.array(self.timestamps)[valid_indices]
        self.files = np.array(self.files)[valid_indices]

        print(f"After filtering for common timestamps, {len(self.files)} files remain")

        if len(self.files) == 0:
            print(f"No valid files found in {self.directory} with common timestamps.")
            return

        self.size = size
        self.scale_factor = scale_factor
        self.augment = augment
        self.frames = frames
        self.convert_cartesian = convert_cartesian
        self.time_step = time_step  # in milliseconds

        self.rand_transform = flow_transforms.Compose([
            flow_transforms.ToTensor(images_order='CHW', flows_order='CHW'),
            flow_transforms.RandomCrop((size, size)),
            flow_transforms.RandomHorizontalFlip(),
            flow_transforms.RandomVerticalFlip(),
        ])

        # Validate files and create a list of valid frame sets
        self.valid_frame_sets = self._validate_files()
        
        print(f"Found {len(self.valid_frame_sets)} valid frame sets in {self.directory}")

    def _validate_files(self):
        valid_frame_sets = []
        expected_diff = 1800000  # 30 minutes in milliseconds
        tolerance = 300000  # 5 minutes tolerance in milliseconds

        for i in range(len(self.timestamps) - self.frames + 1):
            time_diffs = np.diff(self.timestamps[i:i + self.frames])
            
            if all(abs(diff - expected_diff) <= tolerance for diff in time_diffs):
                valid_frame_sets.append(i)
            else:
                print(f"\nInvalid time difference in files:")
                for j, file in enumerate(self.files[i:i + self.frames]):
                    print(f"  {j+1}: {file}")
                print(f"Timestamps: {self.timestamps[i:i + self.frames]}")
                print(f"Time differences: {time_diffs}")
                print(f"Expected difference: {expected_diff} ± {tolerance}")
        
        print(f"\nValidated {len(self.timestamps) - self.frames + 1} potential frame sets, found {len(valid_frame_sets)} valid ones")
        return valid_frame_sets

    def __len__(self):
        return len(self.valid_frame_sets)

    def __getitem__(self, idx):
        if len(self.valid_frame_sets) == 0:
            raise IndexError("Dataset contains no valid frame sets")
        
        max_retries = len(self.valid_frame_sets)
        for _ in range(max_retries):
            start_idx = self.valid_frame_sets[idx]
            
            try:
                ds = xr.open_mfdataset(self.files[start_idx:start_idx + self.frames])
                
                if self.scale_factor:
                    h, w = ds.U.isel(time=0).shape
                    h, w = int(h * self.scale_factor), int(w * self.scale_factor)
                    new_lats = np.linspace(ds.lat.values[0], ds.lat.values[-1], h)
                    new_lons = np.linspace(ds.lon.values[0], ds.lon.values[-1], w)
                    ds = ds.interp(lat=new_lats, lon=new_lons)

                qv = ds['QV'].values
                u = ds['U'].values
                v = ds['V'].values

                uv = np.concatenate([u[:, np.newaxis], v[:, np.newaxis]], 1)
                uv[~np.isfinite(uv)] = 0.
                qv[~np.isfinite(qv)] = 0.

                if self.convert_cartesian:
                    lat_rad = np.radians(ds.lat.values)
                    lon_rad = np.radians(ds.lon.values)
                    a = np.cos(lat_rad) ** 2 * np.sin((lon_rad[1] - lon_rad[0]) / 2) ** 2
                    d = 2 * 6378.137 * np.arcsin(a ** 0.5)
                    size_per_pixel = np.repeat(np.expand_dims(d, -1), len(lon_rad), axis=1)  # kms
                    # Convert m/s to meters of displacement over the time step
                    uv = uv / size_per_pixel
                    uv = uv / 1000 # meters
                    uv = uv * (self.time_step / 1000) # seconds

                    

                qv = image_histogram_equalization(qv)
                images = [q[np.newaxis] for q in qv]
                flows = [_uv for _uv in uv]
                images_tensor, flow_tensor = self.rand_transform(images, flows)
                
                ds.close()
                return images_tensor, flow_tensor

            except Exception as e:
                print(f"Error processing files {self.files[start_idx:start_idx + self.frames]}: {e}")
                idx = (idx + 1) % len(self.valid_frame_sets)
        
        raise RuntimeError("Failed to load data after trying all valid frame sets")

class G5NRFlows(data.ConcatDataset):
    def __init__(self, directory, scale_factor=None, size=None, augment=False, frames=2, convert_cartesian=True, time_step=10800000):
        self.directory = directory
        print(f"Initializing G5NRFlows for directory: {directory}")
        
        tiles = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith('500')]
        print(f"Found {len(tiles)} tile directories")
        
        if not tiles:
            raise ValueError(f"No directories starting with '500' found in {directory}")

        tile_paths = [os.path.join(directory, t) for t in tiles]

        all_timestamps = []
        for p in tile_paths:
            files = np.array(sorted(glob.glob(os.path.join(p, '*.nc4'))))
            timestamps = np.array([parse_timestamp(f) for f in files])
            all_timestamps.append(timestamps)

        common_timestamps = set(all_timestamps[0])
        for timestamps in all_timestamps[1:]:
            common_timestamps.intersection_update(timestamps)
        common_timestamps = sorted(list(common_timestamps))

        print(f"Found {len(common_timestamps)} common timestamps across all tiles")

        if not common_timestamps:
            raise ValueError("No common timestamps found across all tiles")

        datasets = []
        for p in tile_paths:
            dataset = TileFlows(p, common_timestamps, scale_factor=scale_factor, size=size, augment=augment, 
                                frames=frames, convert_cartesian=convert_cartesian, time_step=time_step)
            if len(dataset) > 0:
                datasets.append(dataset)
            else:
                print(f"Warning: No valid frame sets found in {p}")

        if not datasets:
            raise ValueError("No valid datasets found. All TileFlows instances are empty.")

        print(f"Created G5NRFlows dataset with {len(datasets)} valid tile directories")
        data.ConcatDataset.__init__(self, datasets)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        return result

def create_g5nr_dataset(directory, **kwargs):
    try:
        return G5NRFlows(directory, **kwargs)
    except ValueError as e:
        print(f"Error creating G5NRFlows dataset: {e}")
        return None

if __name__ == '__main__':
    dataset = G5NRFlows('/ships19/cryo/daves/windflow/paul/G5NR_patches', size=128, time_step=10800000)
    print(f"Created dataset with {len(dataset)} samples")
    
    # Test loading a few samples
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            print(f"Successfully loaded sample {i}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")