import os
import numpy as np
import pandas as pd
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from torchvision import transforms

warnings.filterwarnings('ignore')


class EraDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M',
                 target='OT', scale=False, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        # self.data_path = data_path
        self.__read_data__()

    def __recurrently_read_data__(self, path, files):
        npys = []
        for file in files:
            path_file = os.path.join(path, file)
            npys += [np.load(path_file, allow_pickle=True)]
        return np.concatenate(npys, axis=0)

    def __read_data__(self):
        print('============ process 【{}】 dataset start =========='.format(self.flag))
        # self.raw_data = np.load(os.path.join(self.root_path, "temp_global_hourly_" + self.flag + ".npy"), allow_pickle=True)  # (17519, 34040, 3)
        files = os.listdir(self.root_path)
        files = [i for i in files if re.match('.*.npy', i)]
        files_time = [i for i in files if re.match('time.*.npy', i)]
        files_time.sort()
        files_data = [i for i in files if not re.match('time.*.npy', i)]
        files_data.sort()
        print('npy files to read: {}'.format(files_data))
        self.raw_data = self.__recurrently_read_data__(self.root_path, files_data)
        self.raw_time = self.__recurrently_read_data__(self.root_path, files_time)

        raw_data = self.raw_data
        raw_time = self.raw_time
        # print(self.raw_data.shape)
        # if self.features == 'S':
        #     raw_data = raw_data[:, :, :1]
        # if self.features == 'S_station':
        #     raw_data = raw_data[:, self.target:(self.target + 1), :1]
        data_len, height, width, feat = raw_data.shape
        # raw_data = raw_data.reshape(data_len, station * feat)  # (17519, 34040*3)

        data = raw_data.astype(np.float)
        df_stamp = raw_time
        df_stamp = pd.to_datetime(df_stamp)
        data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        train_start, train_end = 0, 0.5
        valid_start, valid_end = 0.5, 0.75
        test_start, test_end = 0.75, 1
        N = data.shape[0]

        if self.flag == 'train':
            self.data_x = data[int(train_start*N): int(train_end*N)]
            self.data_y = data[int(train_start*N): int(train_end*N)]
            self.data_stamp = data_stamp[int(train_start*N): int(train_end*N)]
        elif self.flag == 'val':
            self.data_x = data[int(valid_start*N): int(valid_end*N)]
            self.data_y = data[int(valid_start*N): int(valid_end*N)]
            self.data_stamp = data_stamp[int(valid_start*N): int(valid_end*N)]
        elif self.flag == 'test':
            self.data_x = data[int(test_start*N): int(test_end*N)]
            self.data_y = data[int(test_start*N): int(test_end*N)]
            self.data_stamp = data_stamp[int(test_start*N): int(test_end*N)]
        else:
            raise NotImplementedError

        # print("{} data load finished.".format(self.flag))
        print('data_x shape = 【{}】, time shape = 【{}】'.format(self.data_x.shape, self.data_stamp.shape))
        print('============ dataset end ==========')
        a = 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

