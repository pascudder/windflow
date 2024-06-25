import os
import glob
import numpy as np

class TileFlowsTest:
    def __init__(self, directory, frames=2, time_step=10800):
        self.directory = directory
        self.files = sorted(glob.glob(os.path.join(self.directory, '*.nc4')))
        
        if len(self.files) == 0:
            raise ValueError(f"No .nc4 files found in directory: {self.directory}")
        
        self.timestamps = np.array([int(os.path.basename(f).split('.')[0]) for f in self.files])
        if not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("Files are not in strictly increasing timestamp order")
        
        self.frames = frames
        self.time_step = time_step * 1000  # Convert to milliseconds

    def __len__(self):
        return max(0, len(self.files) - 6)  # Ensure we have at least 6 files to look ahead

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} samples")
        
        start_idx = idx
        target_time = self.timestamps[start_idx] + self.time_step
        
        end_idx = start_idx + 1
        for i in range(start_idx + 1, min(start_idx + 13, len(self.files))):
            current_diff = abs(int(self.timestamps[i]) - int(target_time))
            best_diff = abs(int(self.timestamps[end_idx]) - int(target_time))
            if current_diff < best_diff:
                end_idx = i
            if int(self.timestamps[i]) >= int(target_time):
                break
        
        selected_files = [self.files[start_idx], self.files[end_idx]]
        selected_times = [self.timestamps[start_idx], self.timestamps[end_idx]]
        return selected_files, selected_times, start_idx, end_idx

def print_concise_info(sample_idx, files, timestamps):
    time_diff = int(timestamps[1]) - int(timestamps[0])  # Time difference in milliseconds
    print(f"Sample {sample_idx}: {os.path.basename(files[0])} -> {os.path.basename(files[1])}, "
          f"Time diff: {time_diff / (3600 * 1000):.2f} hours")

def main():
    directory = '/data/sreiner/windflow_patches/train/200_13.75_101.25'  # Replace with your actual directory path
    
    try:
        dataset = TileFlowsTest(directory, frames=2, time_step=10800)  # 3 hours in seconds
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        print(f"Contents of directory {directory}:")
        print(os.listdir(directory))
        return

    print(f"Total samples: {len(dataset)}")
    print(f"Total files found: {len(dataset.files)}")
    print(f"First few files: {[os.path.basename(f) for f in dataset.files[:5]]}")
    print(f"Last few files: {[os.path.basename(f) for f in dataset.files[-5:]]}")
    print("\nProcessing all samples:")
    
    for i in range(len(dataset)):
        try:
            files, timestamps, _, _ = dataset[i]
            print_concise_info(i, files, timestamps)
        except IndexError as e:
            print(f"Error processing sample {i}: {e}")

if __name__ == '__main__':
    main()