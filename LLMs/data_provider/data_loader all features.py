import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
import pickle
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')

stl_position = 'stl/'




class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, data_name='weather', max_len=-1, train_all=False, output_attn_map=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

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
   
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        # self.save_stl = 'stl/'

    def stl_resolve(self, data_raw):
        """
        STL Global Decomposition
        """

        save_stl = stl_position + self.data_name
        # save_stl = 'stl/' + 'weather'

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n, m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()

                if 'weather' in self.data_name:  # == 'weather':
                    res = STL(df, period=24 * 6).fit()
                elif 'ill' in self.data_name:
                    res = STL(df).fit()  # , period = 7 52？
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period=24).fit()
                elif 'ettm1' in self.data_name or 'ettm2' in self.data_name:
                    res = STL(df, period=24 * 4).fit()
                elif 'traffic' in self.data_name or 'electricity' in self.data_name:
                    res = STL(df, period=24).fit()
                else:
                    res = STL(df).fit()

                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # After we get data, we do the stl resolve
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))
        # Disable STL completely
        trend_stamp = torch.zeros_like(torch.tensor(data, dtype=torch.float32))
        seasonal_stamp = torch.zeros_like(trend_stamp)
        resid_stamp = torch.zeros_like(trend_stamp)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id + 1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id + 1]
        seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        # Distribute the 10,000 test cap evenly across all channels,
        # matching the test sample limit used in data_loader_patchtst.py
        # for a fair comparison between GPT/TEMPO and PatchTST models.
        windows_per_channel = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.set_type == 2:  # 2 = test
            capped_windows = max(1, 10000 // self.enc_in)
            windows_per_channel = min(capped_windows, windows_per_channel)
        return windows_per_channel * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)