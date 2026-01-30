import numpy as np
import pandas as pd


class WalkForwardSplit:
    """
    A time-series cross-validator that provides train/test indices to split data into
    expanding windows, respecting temporal order.
    """
    def __init__(self, n_splits: int, train_blocks_init: int, test_blocks: int, date_init, assets):
        self.n_splits = n_splits
        self.train_blocks_init = train_blocks_init
        self.test_blocks = test_blocks
        self.date_init = date_init
        self.assets = assets
    
    def expanding_split(self, data):
        n_samples = data.shape[0]
        indices = np.arange(n_samples)

        len_blocks = n_samples // self.n_splits
        start_test_index = len_blocks * self.train_blocks_init + n_samples % self.n_splits
        end_test_index = start_test_index + len_blocks * self.test_blocks

        for k in range(self.n_splits):
            train_indices = indices[:start_test_index]
            test_indices = indices[start_test_index : end_test_index]

            yield train_indices, test_indices        
            
            if end_test_index == n_samples:
                yield indices, np.array([], dtype=int)
                break
            
            start_test_index += len_blocks
            end_test_index = start_test_index + len_blocks * self.test_blocks
            
    def expanding(self, data):
        splits_values = []
        for train_idx, test_idx in self.expanding_split(data):
            train_values = data[train_idx,:]
            test_values = data[test_idx,:]
            splits_values.append((train_values, test_values))
           
        return splits_values
    
    '''
    def rolling_split(self, data):
        test_date_init = self.date_init
        train_date_init = list(self.assets.keys())[next(i for i, k in enumerate(self.assets) if k == self.date_init) - self.train_blocks_init]
        
        n_samples = data.shape[0]
        indices = np.arange(n_samples)

        len_blocks = n_samples // self.n_splits
        
        start_train_index = 0
        start_test_index = len_blocks * self.train_blocks_init + n_samples % self.n_splits
        end_test_index = start_test_index + len_blocks * self.test_blocks

        for k in range(self.n_splits):
            train_indices = indices[start_train_index : start_test_index]
            test_indices = indices[start_test_index : end_test_index]

            yield train_indices, test_indices        
            
            if end_test_index == n_samples:
                yield indices, np.array([], dtype=int)
                break
            
            start_train_index += len_blocks
            start_test_index += len_blocks
            end_test_index = start_test_index + len_blocks * self.test_blocks
    '''       
    def rolling_split(self, data):
        
        q_dates = list(self.assets.keys())
        test_index = q_dates.index(pd.to_datetime(self.date_init))
        train_index = test_index - self.train_blocks_init
        
        while True:      
            assets = self.assets[q_dates[test_index]]
            
            try:
                data_split = (
                    data.loc[
                        (data.index >= q_dates[train_index]) &
                        (data.index <= q_dates[test_index + 1]),
                        assets
                        ]
                    .ffill()
                    .iloc[4:]
                    .dropna(axis=1)
                )
            except IndexError:
                data_split = (
                    data.loc[(data.index >= q_dates[train_index]), assets]
                    .ffill()
                    .iloc[4:]
                    .dropna(axis=1)
                )
            max_repeats = data_split.apply(lambda s: s.value_counts().max())
            data_split = data_split.loc[:, max_repeats <= 10]
            
            data_train = data_split[data_split.index < q_dates[test_index]]
            data_test = data_split[data_split.index >= q_dates[test_index]]

            yield data_train, data_test
            
            if test_index == len(q_dates) - 1:
                break
            
            test_index += 1
            train_index += 1
            
            
    def rolling(self, data: pd.DataFrame):
        splits_values = []
        for data_train, data_test in self.rolling_split(data):
            splits_values.append((data_train, data_test))
           
        return splits_values
    
