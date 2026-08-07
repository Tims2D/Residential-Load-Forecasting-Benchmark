import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from pathlib import Path
import pickle
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')

stl_position = 'stl/'


class Dataset_Custom(Dataset):
    """
    Custom Dataset for time series forecasting.

    KEY DESIGN DECISION — Target-only output:
    ─────────────────────────────────────────
    Regardless of how many input features exist (enc_in = 1 or 12),
    __getitem__ always returns a window for the TARGET channel only.

    This makes the dataset general across two cases:
      • Univariate data  (e.g. 15-min load, enc_in=1):
            target_feat_id = 0  → only column available, no change in behaviour.

      • Multivariate data (e.g. hourly load + 11 weather cols, enc_in=12):
            target_feat_id = enc_in-1  → Load is always moved to the last
            column by __read_data__, so we always slice [:, -1:].

    In both cases:
      seq_x  shape : [seq_len,           1]
      seq_y  shape : [label_len+pred_len, 1]
      __len__       : tot_len   (NOT tot_len × enc_in)

    This matches data_loader_patchtst.py sample count exactly, enabling
    a fair comparison between GPT/TEMPO and PatchTST on the same target.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, data_name='weather', max_len=-1,
                 train_all=False, output_attn_map=False):

        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len   = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len  = 24 * 4
        else:
            self.seq_len   = size[0]
            self.label_len = size[1]
            self.pred_len  = size[2]

        assert flag in ['train', 'test', 'val']
        type_map      = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag     = flag

        self.features        = features
        self.target          = target
        self.scale           = scale
        self.timeenc         = timeenc
        self.freq            = freq
        self.percent         = percent
        self.output_attn_map = output_attn_map
        self.root_path       = root_path
        self.data_path       = data_path
        self.data_name       = data_name

        self.__read_data__()

        # ── Derived dimensions ────────────────────────────────────────
        self.enc_in  = self.data_x.shape[-1]   # total feature columns loaded
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        # Target channel index:
        #   • Multivariate (enc_in > 1): __read_data__ reorders columns so
        #     target is always LAST  →  target_feat_id = enc_in - 1
        #   • Univariate   (enc_in = 1): only one column  →  target_feat_id = 0
        self.target_feat_id = self.enc_in - 1
        # ─────────────────────────────────────────────────────────────

        print(f"[Dataset_Custom | {flag}] "
              f"data={data_name} | enc_in={self.enc_in} | "
              f"tot_len={self.tot_len} | "
              f"target='{self.target}' (feat_id={self.target_feat_id}) | "
              f"samples={len(self)}")

    # ─────────────────────────────────────────────────────────────────
    # STL decomposition (kept intact, not used by default)
    # ─────────────────────────────────────────────────────────────────
    def stl_resolve(self, data_raw):
        save_stl    = stl_position + self.data_name
        self.save_stl = save_stl
        trend_pk    = save_stl + '/trend.pk'
        seasonal_pk = save_stl + '/seasonal.pk'
        resid_pk    = save_stl + '/resid.pk'

        if (os.path.isfile(trend_pk) and
                os.path.isfile(seasonal_pk) and
                os.path.isfile(resid_pk)):
            with open(trend_pk,    'rb') as f: trend_stamp    = pickle.load(f)
            with open(seasonal_pk, 'rb') as f: seasonal_stamp = pickle.load(f)
            with open(resid_pk,    'rb') as f: resid_stamp    = pickle.load(f)
        else:
            os.makedirs(save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)
            n, m = data_raw.shape
            trend_stamp    = torch.zeros([n, m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([n, m], dtype=torch.float32)
            resid_stamp    = torch.zeros([n, m], dtype=torch.float32)
            for i, col in enumerate(data_raw.columns):
                df = data_raw[col]
                if   'weather'     in self.data_name: res = STL(df, period=24*6).fit()
                elif 'ill'         in self.data_name: res = STL(df).fit()
                elif 'etth1'       in self.data_name or 'etth2' in self.data_name: res = STL(df, period=24).fit()
                elif 'ettm1'       in self.data_name or 'ettm2' in self.data_name: res = STL(df, period=24*4).fit()
                elif 'traffic'     in self.data_name or 'electricity' in self.data_name: res = STL(df, period=24).fit()
                else: res = STL(df).fit()
                trend_stamp[:,    i] = torch.tensor(res.trend.values,    dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(res.seasonal.values, dtype=torch.float32)
                resid_stamp[:,    i] = torch.tensor(res.resid.values,    dtype=torch.float32)
            with open(trend_pk,    'wb') as f: pickle.dump(trend_stamp,    f)
            with open(seasonal_pk, 'wb') as f: pickle.dump(seasonal_stamp, f)
            with open(resid_pk,    'wb') as f: pickle.dump(resid_stamp,    f)
        return trend_stamp, seasonal_stamp, resid_stamp

    # ─────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Reorder: ['date', ...other features..., target]
        # This guarantees target is always the LAST column,
        # so target_feat_id = enc_in - 1 works for any dataset.
        cols = list(df_raw.columns)
        cols.remove('date')

        if self.target in cols:
            # Multivariate or explicit target column present
            cols.remove(self.target)
            df_raw = df_raw[['date'] + cols + [self.target]]
        else:
            # Target column not found — treat as univariate (data IS the target)
            print(f"[Dataset_Custom] Warning: target='{self.target}' not found in columns. "
                  f"Available: {cols}. Using all columns as-is.")
            df_raw = df_raw[['date'] + cols]

        # Split boundaries — identical to data_loader_patchtst.py
        num_train = int(len(df_raw) * 0.7)
        num_test  = int(len(df_raw) * 0.2)
        num_vali  = len(df_raw) - num_train - num_test
        border1s  = [0,
                     num_train - self.seq_len,
                     len(df_raw) - num_test - self.seq_len]
        border2s  = [num_train,
                     num_train + num_vali,
                     len(df_raw)]
        border1   = border1s[self.set_type]
        border2   = border2s[self.set_type]

        # Percent-based training truncation (GPT-style few-shot support)
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # Feature selection
        if self.features in ['M', 'MS']:
            df_data = df_raw.iloc[:, 1:]        # all columns except date
        else:  # 'S'
            df_data = df_raw[[self.target]]     # target only

        # Scale: fit on train, transform all
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            print(f"[Dataset_Custom | {self.flag}] "
                  f"Scaler mean={np.round(self.scaler.mean_, 4)} | "
                  f"std={np.round(self.scaler.scale_, 4)}")
        else:
            data = df_data.values

        # STL stamps — disabled (zeros), kept for API compatibility with TEMPO
        trend_stamp    = torch.zeros_like(torch.tensor(data, dtype=torch.float32))
        seasonal_stamp = torch.zeros_like(trend_stamp)
        resid_stamp    = torch.zeros_like(trend_stamp)

        # Time features
        df_stamp         = df_raw[['date']].iloc[border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month']   = df_stamp['date'].dt.month
            df_stamp['day']     = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour']    = df_stamp['date'].dt.hour
            df_stamp['minute']  = df_stamp['date'].dt.minute // 15 * 15
            data_stamp = df_stamp.drop('date', axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x         = data[border1:border2]
        self.data_y         = data[border1:border2]
        self.trend_stamp    = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp    = resid_stamp[border1:border2]
        self.data_stamp     = data_stamp

    # ─────────────────────────────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────────────────────────────
    def __getitem__(self, index):
        """
        Returns one sliding window for the TARGET channel only.

        Works for both:
          • Univariate  (enc_in=1): target_feat_id=0, index=window directly
          • Multivariate (enc_in>1): target_feat_id=enc_in-1, same logic

        Returned shapes:
          seq_x, seq_y        : [seq_len / label_len+pred_len, 1]
          seq_x_mark,
          seq_y_mark          : [seq_len / label_len+pred_len, time_features]
          seq_trend,
          seq_seasonal,
          seq_resid           : [seq_len, 1]
        """
        feat_id = self.target_feat_id   # always the target column
        s_begin = index                 # index IS the window start position

        s_end   = s_begin + self.seq_len
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x        = self.data_x[s_begin:s_end,   feat_id:feat_id + 1]
        seq_y        = self.data_y[r_begin:r_end,    feat_id:feat_id + 1]
        seq_trend    = self.trend_stamp[s_begin:s_end,    feat_id:feat_id + 1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id + 1]
        seq_resid    = self.resid_stamp[s_begin:s_end,    feat_id:feat_id + 1]
        seq_x_mark   = self.data_stamp[s_begin:s_end]
        seq_y_mark   = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        """
        Number of sliding windows over the TARGET channel only.

        ORIGINAL: tot_len * enc_in  (all channels cycled)
        NEW:      tot_len           (target channel only)

        Test split is capped at 10,000 to match data_loader_patchtst.py,
        ensuring both models are evaluated on the same number of windows.
        """
        windows = self.tot_len
        if self.set_type == 2:          # test split only
            windows = min(10000, windows)
        return windows

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)