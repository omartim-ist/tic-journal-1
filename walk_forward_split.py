import numpy as np
  
class WalkForwardSplit:
    """
    A time-series cross-validator that provides train/test indices to split data into
    expanding windows, respecting temporal order.
    """
    def __init__(self, n_splits, train_blocks_init, test_blocks):
        self.n_splits = n_splits
        self.train_blocks_init = train_blocks_init
        self.test_blocks = test_blocks
    
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
    
    
    def rolling_split(self, data):
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
            
    def rolling(self, data):
        splits_values = []
        for train_idx, test_idx in self.rolling_split(data):
            train_values = data[train_idx,:]
            test_values = data[test_idx,:]
            splits_values.append((train_values, test_values))
           
        return splits_values
    
'''
np.random.seed(100)
X_rand = np.random.normal(size=100)
X = 100 * np.exp(np.cumsum(X_rand*0.01))

tscv = WalkForwardSplit(n_splits=8, train_blocks_init=3, test_blocks=2)
splits = tscv.expanding(X)
'''