import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from statsmodels.tsa.seasonal import STL
import pickle

warnings.filterwarnings('ignore')

stl_position = 'stl/'


class Dataset_Custom(Dataset):
    """
    GPT data loader aligned with PatchTST for fair comparison.
    One sample = one multivariate time window.
    """

    def __init__(
        self,
        root_path,
        flag='train',
        size=None,                 # [seq_len, label_len, pred_len]
        features='S',
        data_path='ETTh1.csv',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        percent=10,
        data_name='weather',
        max_len=-1,
        train_all=False,
        output_attn_map=False
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.output_attn_map = output_attn_map
        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.__read_data__()

        # number of valid windows (PatchTST-style)
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    # ------------------------------------------------------------------
    # Data loading (UNCHANGED logic)
    # ------------------------------------------------------------------
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # 70 / 10 / 20 split
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len
        ]
        border2s = [
            num_train,
            num_train + num_vali,
            len(df_raw)
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features in ['M', 'MS']:
            df_data = df_raw.iloc[:, 1:]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Disable STL (keep placeholders for compatibility)
        trend_stamp = torch.zeros_like(torch.tensor(data, dtype=torch.float32))
        seasonal_stamp = torch.zeros_like(trend_stamp)
        resid_stamp = torch.zeros_like(trend_stamp)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values),
                freq=self.freq
            ).transpose(1, 0)
            

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]
        self.data_stamp = data_stamp

    # ------------------------------------------------------------------
    # One sample = one multivariate window ✅
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_trend = self.trend_stamp[s_begin:s_end]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end]
        seq_resid = self.resid_stamp[s_begin:s_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
            seq_trend,
            seq_seasonal,
            seq_resid
        )

    # ------------------------------------------------------------------
    # Length = PatchTST test size ✅
    # ------------------------------------------------------------------
    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
