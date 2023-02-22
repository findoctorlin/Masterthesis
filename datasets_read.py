import os
import h5py
from pandas.core.frame import DataFrame
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

# from CMAPSS.utils.transformations import MinMaxScaler

class CMAPSS(Dataset):
    """

    Attributes:
        kind: {'TSF', 'RUL'}
        split: {'train', 'test', 'test_plus'}
     
    """ 

    def __init__(
            self,
            dataset_name='FD001',
            dir_path='/home/linyuhan/Dokumente/Masterarbeit/dataset/Turbofan_Engine_Degradation_Simulation_Dataset2',
            steps_before=None,
            steps_after=None,
            transformations=[],
            kind='TSF',
            data_split='train',
            random_seed=42,
            val_size=0.2,
            rul_args={'return_last_rul': True, 'max_rul': 125},
            return_idx=False,
            drop_sensors=None,
    ):
    
        super().__init__()

        self.dataset_name = dataset_name
        self.dir_path = dir_path
        self.data_split = data_split
        self.kind = kind

        self.random_seed = random_seed
        self.val_size = val_size

        self.index_names = ['unit_idx', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = ['s_{}'.format(i) for i in range(1,22)]
        
        self.steps_before = steps_before
        self.steps_after = steps_after
        self.transformations = transformations

        self.rul_args = rul_args

        if drop_sensors is None:
            self.drop_sensors = self.get_drop_sensors()
        else:
            self.drop_sensors = drop_sensors

        self.df = self.load_data()

        self.num_features = 21 - len(self.drop_sensors)

        if self.steps_before is not None and self.steps_after is not None: 
            self.index_map = self._getindexmap()

        self.return_idx = return_idx

        self.check_parsed_args()

    def check_parsed_args(self):
        if self.kind == 'RUL':
            assert self.steps_after == 0, 'steps_after has to be set to 0'
            # assert self.steps_before is not None
        
    def __len__(self):

        if self.steps_before is None or (self.kind == 'RUL' and self.data_split == 'test'):
            return self.get_num_units()
        else:
            return len(self.index_map)

    def _getindexmap(self):
        index_map = dict()
        # num_units = self.get_num_units()
        
        # assert np.isin(self.df['unit_idx'].unique(), range(num_units)).all()

        unit_ids = self.df['unit_idx'].unique()
        
        idx_count = 0
        for unit_idx in unit_ids:
            unit_seq = self.df[self.df['unit_idx'] == unit_idx]
            max_offset = len(unit_seq) - self.steps_before - self.steps_after
            for offset in range(max_offset+1):
                curr_timeidx = offset + self.steps_before - 1
                index_map[idx_count] = [unit_idx, curr_timeidx]
                idx_count += 1
        return index_map

    def get_num_units(self):
        return len(self.df.groupby(by="unit_idx"))

    def __getitem__(self, index):

        if index >= len(self): raise IndexError

        if self.kind == 'TFS':
            return self._getitem_TSF(index)
        elif self.kind == 'RUL':
            return self._getitem_RUL(index, **self.rul_args)
        else:
            raise ValueError('Not supported kind')

    def _getitem_TSF(self, index):
        if self.steps_before is None and self.steps_after is None:
            # get the whole seq of the unit 
            seq_df = self._get_unit_seq(index)
        else:          
            # get cropped sequence
            seq_df = self._get_crop_seq(index)

        drop_columns = self.index_names + self.setting_names
        seq_df.drop(labels=drop_columns, axis=1, inplace=True)

        res_seq = np.array(seq_df)

        for tranform in self.transformations:
            res_seq = tranform(res_seq)

        if self.steps_before is None and self.steps_after is None:
            return res_seq

        else:
            input_seq = res_seq[:self.steps_before]
            gt_seq = res_seq[self.steps_before: self.steps_before + self.steps_after] 
            return input_seq, gt_seq

    def _getitem_RUL(self, index: int, return_last_rul=True, **kwargs):

        if self.steps_before is None:
            # get the whole seq of the unit 
            seq_df = self._get_unit_seq(self.df['unit_idx'].unique()[index])
        elif self.data_split in ['train', 'val', 'test_plus']:
            # get cropped sequence
            seq_df = self._get_crop_seq(index)
        elif self.data_split == 'test':
            # get the whole seq of the unit 
            seq_df = self._get_unit_seq(self.df['unit_idx'].unique()[index])

            # get cropped sequence
            seq_df = seq_df.iloc[-self.steps_before:]

        drop_columns = self.index_names + self.setting_names
        seq_df = seq_df.drop(labels=drop_columns, axis=1)

        input_seq = np.array(seq_df.iloc[:, :-1])

        if return_last_rul:
            gt_rul = np.array(seq_df.iloc[-1, -1:])
        else:
            gt_rul = np.array(seq_df.iloc[:, -1:])

        # for tranform in self.transformations:
        #     input_seq = tranform(input_seq)
        if hasattr(self, 'return_idx') and self.return_idx:
            return input_seq, gt_rul, index
        else:
            return input_seq, gt_rul

    def _get_unit_seq(self, unit_idx):
        unit_seq_df = self.df[self.df['unit_idx'] == unit_idx]
        return unit_seq_df

    def _get_crop_seq(self, index):
        unit_idx, curr_timeidx = self.index_map[index]
        unit_seq_df = self._get_unit_seq(unit_idx)
        
        # get current cropped sequence
        seq_df = unit_seq_df[curr_timeidx - self.steps_before + 1 : curr_timeidx + self.steps_after + 1]
        return seq_df

    def get_drop_sensors(self):
    
        drop_sensors = {
            'FD001': ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19'],
            'FD002': [],
            'FD003': ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19'],
            'FD004': []
        }

        return drop_sensors[self.dataset_name]


    def load_data(self):

        if self.data_split in ['train', 'val']:
            filename = 'train_' + self.dataset_name + '.txt'
        elif self.data_split in ['test', 'test_plus']:
            filename = 'test_' + self.dataset_name + '.txt'
        else:
            raise ValueError

        col_names = self.index_names + self.setting_names + self.sensor_names
        # filename = 'train' + self.dataset_name + '.txt'
        df = pd.read_csv(os.path.join(self.dir_path, filename), header=None, sep='\s+', names=col_names)
        df['unit_idx'] -= 1
        df = self.add_remaining_useful_life(df)

        df = df.drop(labels=self.drop_sensors, axis=1)

        for transfom in self.transformations:
            df.iloc[:, 5:-1] = transfom(df.iloc[:, 5:-1])

        unit_ids = df['unit_idx'].unique()

        # Shuffle units indices
        np.random.seed(self.random_seed)
        np.random.shuffle(unit_ids)

        if self.data_split == 'train':
            unit_ids_train = np.unique(unit_ids[:len(unit_ids)-int(len(unit_ids)*self.val_size)])
            df = df[df['unit_idx'].isin(unit_ids_train)]
        elif self.data_split == 'val':
            unit_ids_val = np.unique(unit_ids[len(unit_ids)-int(len(unit_ids)*self.val_size):])
            df = df[df['unit_idx'].isin(unit_ids_val)]

        if self.rul_args['max_rul']:
            df['RUL'][df['RUL'] > self.rul_args['max_rul']] = self.rul_args['max_rul']

        return df

    def add_remaining_useful_life(self, df):
        # Get the total number of cycles for each unit
        grouped_by_unit = df.groupby(by="unit_idx")

        max_cycle = grouped_by_unit["time_cycles"].max()

        if self.data_split in ['test', 'test_plus']:
            rul_test = pd.read_csv(os.path.join(self.dir_path, 'RUL_' + self.dataset_name + '.txt'),
                                   header=None, sep='\s+', names=['RUL'])
            max_cycle = max_cycle + rul_test['RUL'] 

        # Merge the max cycle back into the original frame
        # (Add max_cycle into the df)
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_idx', right_index=True)
        
        # Calculate remaining useful life for each row
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
        result_frame["RUL"] = remaining_useful_life
        
        # drop max_cycle as it's no longer needed
        result_frame = result_frame.drop("max_cycle", axis=1)
        return result_frame


class NCMAPSS(CMAPSS):
    def __init__(
            self,
            *args,
            dataset_name='DS03-012',
            dir_path='/home/linyuhan/Dokumente/Masterarbeit/dataset/Turbofan_Engine_Degradation_Simulation_Dataset2',
            # random_seed=42,
            # val_size=0.33,
            downsampling=10,
            **kwargs):

        # self.random_seed = random_seed
        # self.val_size = val_size
        self.downsampling = downsampling

        super().__init__(*args, dataset_name=dataset_name, dir_path=dir_path, **kwargs)

        self.num_features = 14 - len(self.get_drop_sensors())

    def __len__(self):

        if self.steps_before is None:
            return self.get_num_units()
        else:
            return len(self.index_map)

    def get_drop_sensors(self):
    
        drop_sensors = []

        return drop_sensors

    def downsample(self, df):
        downsampled_df = DataFrame(columns=df.columns)
        unit_ids = df['unit_idx'].unique()
        
        for unit_idx in unit_ids:
            df_ = df[df['unit_idx'] == unit_idx]
            df_ = df_.iloc[::self.downsampling, :]
            downsampled_df = pd.concat([downsampled_df, df_])
            
        return downsampled_df

    def load_data(self):

        filename = 'N-CMAPSS_' + self.dataset_name + '.h5'

        with h5py.File(os.path.join(self.dir_path, filename), 'r') as hdf:
            # Development set: Units: 2, 5, 10, 18, 18, 20
            if self.data_split in ['train', 'val']:
                X_s = np.array(hdf.get('X_s_dev'))
                Y = np.array(hdf.get('Y_dev'))           # RUL  
                A = np.array(hdf.get('A_dev'))           # Auxiliary: Shape: [*, 4]
            else:
                # Test set: Units; 11, 14, 15
                X_s = np.array(hdf.get('X_s_test'))
                A = np.array(hdf.get('A_test'))           # Auxiliary: Shape: [*, 4]
                Y = np.array(hdf.get('Y_test'))           # RUL  
            
            # Varnams
            
            X_s_var = np.array(hdf.get('X_s_var'))  # [b'T24', b'T30', b'T48', b'T50', b'P15', b'P2', b'P21', b'P24',
                                                    #  b'Ps30', b'P40', b'P50', b'Nf', b'Nc', b'Wf']

            # [b'unit', b'cycle', b'Fc', b'hs']: Fc and hs?? Fc: Flight class, hs: Health state
            A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            X_s_var = list(np.array(X_s_var, dtype='U20'))
            A_var = list(np.array(A_var, dtype='U20'))

        df_X = DataFrame(data=X_s, columns=X_s_var)
        df_A = DataFrame(data=A, columns=A_var)
        df_Y = DataFrame(data=Y, columns=['RUL'])

        df_A = self.add_unit_idx(df_A)

        df = pd.concat([df_A, df_X, df_Y], axis=1)

        df = self.downsample(df)
        
        for transfom in self.transformations:
            df.iloc[:, 5:-1] = transfom(df.iloc[:, 5:-1])

        unit_ids = df['unit_idx'].unique()

        # Shuffle units indices
        np.random.seed(self.random_seed)
        np.random.shuffle(unit_ids)

        if self.data_split == 'train':
            unit_ids_train = np.unique(unit_ids[:len(unit_ids)-int(len(unit_ids)*self.val_size)])
            df = df[df['unit_idx'].isin(unit_ids_train)]
        elif self.data_split == 'val':
            unit_ids_val = np.unique(unit_ids[len(unit_ids)-int(len(unit_ids)*self.val_size):])
            df = df[df['unit_idx'].isin(unit_ids_val)]

        return df

    def add_unit_idx(self, df):
        df_idx = DataFrame(data=df['unit'].unique(), columns=['unit'])
        df_idx['unit_idx'] = df_idx.index # The index (row labels) of the DataFrame
        df = df_idx.merge(df, left_on='unit', right_on='unit')

        return df

    def _getitem_RUL(self, index: int, return_last_rul=True, **kwargs):

        if self.steps_before is None:
            # get the whole seq of the unit 
            seq_df = self._get_unit_seq(index)
        else:
            # get cropped sequence
            seq_df = self._get_crop_seq(index)

        input_seq = np.array(seq_df.iloc[:, 5:-1])

        if return_last_rul:
            gt_rul = np.array(seq_df.iloc[-1, -1:], dtype=float)
        else:
            gt_rul = np.array(seq_df.iloc[:, -1:])

        return input_seq, gt_rul


if __name__ == "__main__":

    # dataset_train = NCMAPSS(steps_before=None, steps_after=0, kind='RUL', data_split='train',
    #                         transformations=[MinMaxScaler()], rul_args={'return_last_rul': True})

    # print(dataset_train[0])

    # dataset_val = NCMAPSS(steps_before=30, steps_after=0, kind='RUL', data_split='val',
    #                         transformations=[MinMaxScaler()], rul_args={'return_last_rul': False})

    # print(dataset_val[0])

    # dataset_test = NCMAPSS(steps_before=50, steps_after=0, kind='RUL', data_split='test',
    #                         transformations=[MinMaxScaler()],rul_args={'return_last_rul': True})

    # print(dataset_test[0])

    # dataset_train = PHM08(steps_before=15, steps_after=0, kind='RUL', data_split='train')

    # print(dataset_train[0])

    # dataset_val = PHM08(steps_before=None, steps_after=0, kind='RUL', data_split='val')

    # print(dataset_val[0])

    # dataset_test = PHM08(steps_before=None, steps_after=0, kind='RUL', data_split='test')

    # print(dataset_test[0])

    # dataset_dev = CMAPSS('FD001', steps_before=31, steps_after=0, kind='RUL', data_split='train', val_size=0,
    #                        transformations=[MinMaxScaler(feature_range=(-1, 1))])

    # print(dataset_dev[0])

    # dataset_train = CMAPSS('FD001', steps_before=31, steps_after=0, kind='RUL', data_split='train',
    #                        transformations=[MinMaxScaler(feature_range=(-1, 1))])

    # print(dataset_train[0])


    # dataset_train = CMAPSS('FD001', steps_before=31, steps_after=0, kind='RUL', data_split='val',
    #                        transformations=[MinMaxScaler(feature_range=(-1, 1))])

    # print(dataset_train[0])

    dataset_test = CMAPSS('FD001', steps_before=31, steps_after=0, kind='RUL', data_split='test', return_idx=True) 

    print(dataset_test[0])

    dataset_test = CMAPSS('FD001', steps_before=31, steps_after=0, kind='RUL', data_split='test_plus', return_idx=True) 

    print(dataset_test[0])
