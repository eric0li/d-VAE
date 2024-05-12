import os
import torch
import torch.utils.data as data
import scipy.io as sio
from scipy.signal import lfilter
import numpy as np
import pdb
class dataloader_raw(data.Dataset):
    def __init__(self, data_path, cfg):
        print(data_path)
        self.neural_data, self.kin_data = self.load_data(data_path, cfg['smooth_num'])
        self.sample_size = self.neural_data.shape[0]

    def __getitem__(self, index):
        return self.neural_data[index], self.kin_data[index]

    def __len__(self):
        return self.sample_size

    def load_data(self, data_path, window_size):
        raw_data = sio.loadmat(data_path)
        fr = raw_data['NeuralData']
        kin = raw_data['KinData']
        if kin.shape[0]<kin.shape[1]:
            fr = np.transpose(fr)
            kin = np.transpose(kin)
        fr = lfilter(np.ones(window_size)/window_size,1,fr,axis=0)
        fr = fr.astype(np.float32)
        kin = kin.astype(np.float32)
        return fr, kin
