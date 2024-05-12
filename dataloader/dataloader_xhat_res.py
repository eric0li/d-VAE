import os
import torch
import torch.utils.data as data
import scipy.io as sio
from scipy.signal import lfilter
import numpy as np
import pdb
class dataloader_xhat_res(data.Dataset):
    def __init__(self, data_path, cfg):
        self.neural_data, self.kin_data,self.raw_data = self.load_data(data_path, cfg['smooth_num'],cfg['dataset_prefix'])
        self.sample_size = self.neural_data.shape[0]

    def __getitem__(self, index):
        return self.neural_data[index], self.kin_data[index]

    def __len__(self):
        return self.sample_size

    def load_data(self, data_path, window_size,xhat_or_res):
        separated_data = sio.loadmat(data_path)
        if xhat_or_res == 'xhat':
            assert ('x_hat' in separated_data.keys())
            # pdb.set_trace()
            fr = separated_data['x_hat']
            fr = lfilter(np.ones(window_size)/window_size,1,fr,axis=0)
            kin = separated_data['y']
        elif xhat_or_res == 'res':
            assert ('res' in separated_data.keys())
            fr = separated_data['res']
            fr = lfilter(np.ones(window_size)/window_size,1,fr,axis=0)
            kin = separated_data['y']
        x = separated_data['x']
        x = lfilter(np.ones(window_size)/window_size,1,x,axis=0)
        fr = fr.astype(np.float32)
        kin = kin.astype(np.float32)
        return fr, kin, x
