from torch.utils.data import Dataset
import numpy as np
from numpy.lib.format import open_memmap

class TrajectoryData(Dataset):
    def __init__(self, prefix: str, city: str):
        self.prefix, self.city = prefix, city

        self.data_path = f'{self.prefix}/{self.city}/{self.city}_train.npy'
        self.label_path = f'{self.prefix}/{self.city}/{self.city}_train_label.npy'
        self.traj_mean_path = f'{self.prefix}/{self.city}/{self.city}_traj_mean.npy'
        self.traj_std_path = f'{self.prefix}/{self.city}/{self.city}_traj_std.npy'
        self.label_mean_path = f'{self.prefix}/{self.city}/{self.city}_label_mean.npy'
        self.label_std_path = f'{self.prefix}/{self.city}/{self.city}_label_std.npy'

        if city == 'chengdu': self.traj_num = 5774528
        elif city == 'xian': self.traj_num = 3880527
        else: raise ValueError(f"Unknown city: {city}")

        # (departure_time, trip_distance, trip_time, original_length, trip_distance/(original_length-1), trip_distance/trip_time, start_region, end_region)
        self.label = np.load(self.label_path)
        assert self.label.shape[0] == self.traj_num
        # process label
        self.label[:, 0] = np.floor((self.label[:, 0] % 86400) / 300)
        self.label_mean, self.label_std = np.load(self.label_mean_path), np.load(self.label_std_path)
        self.label[:,1:6] = (self.label[:,1:6] - self.label_mean) / self.label_std
        self.label[self.label[:, -1] < 0, -1] = 0
        self.label[self.label[:, -2] < 0, -2] = 0

        self.traj_mean = np.load(self.traj_mean_path)
        self.traj_std = np.load(self.traj_std_path)

    def __len__(self):
        return self.traj_num

    def __getitem__(self, idx):
        fp_data = open_memmap(self.data_path, dtype='float64', mode='r', shape=(self.traj_num, 200, 2))
        data = np.array(fp_data[idx].copy())
        del fp_data

        data = (data - self.traj_mean) / self.traj_std
        assert data.dtype == np.float64

        return data.astype(np.float32), self.label[idx]
