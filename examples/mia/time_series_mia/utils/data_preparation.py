import os
import pickle
import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from torch import float32
from torch import tensor
from torch.utils.data import DataLoader, Dataset, Subset

class ECGDataset(Dataset):
    def __init__(self, x:tensor, y:tensor):
        self.x = x
        self.y = y

        self.lookback = x.size(1)
        self.horizon = y.size(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, ...], self.y[idx, ...]
    
    def subset(self, indices):
        return ECGDataset(self.x[indices], self.y[indices])

def to_sequences(data, lookback, horizon):
    x, y = [], []
    timesteps = len(data)
    num_samples = timesteps - (lookback + horizon) + 1

    for t in range(num_samples):
        x.append(data[t:t + lookback, :])
        y.append(data[t + lookback:t + lookback + horizon, :])
    return tensor(x, dtype=float32), tensor(y, dtype=float32)

def preprocess_ECG_dataset(path, lookback, horizon):
    """Get and preprocess the dataset."""

    dataset = None
    if os.path.exists(os.path.join(path, "ECG_E00001.pkl")):
        with open(os.path.join(path, "ECG_E00001.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.x.shape[1] != lookback or dataset.y.shape[1] != horizon:
        file = os.path.join(path, "ECG_E00001.npy")
        data = np.load(file, allow_pickle=True)

        # Scale all variables to range [0, 1]
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Create sequence dataset
        x, y = to_sequences(data_scaled, lookback, horizon)
        dataset = ECGDataset(x, y)

        # Save dataset
        with open(f"{path}/ECG_E00001.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}.pkl") 

    return dataset

def get_ECG_dataloaders(dataset: ECGDataset, train_fraction):

    seq_len = dataset.lookback + dataset.horizon
    non_overlapping_len = len(dataset) - seq_len - 1    # the number of samples when splitting dataset in two parts such that no time points overlap

    train_end_idx = int(non_overlapping_len  * train_fraction)
    test_begin_idx = train_end_idx + seq_len
    train_indices = np.arange(0, train_end_idx)
    test_indices = np.arange(test_begin_idx, len(dataset))

    train_subset = Subset(dataset, train_indices) 
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

    return train_loader, test_loader