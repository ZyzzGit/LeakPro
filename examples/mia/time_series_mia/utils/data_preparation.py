import os
import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from numpy import float32
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset

class ECGDataset(TensorDataset):
    def __init__(self, x:tensor, y:tensor, indices:np.ndarray):
        super().__init__(x, y)
        self.indices = indices  # the indices correspond to the raw timesteps (before creating sequences)

def to_sequences(data, lookback, horizon):
    x, y = [], []
    timesteps = len(data)
    num_samples = timesteps - (lookback + horizon) + 1

    for t in range(num_samples):
        x.append(data[t:t + lookback, :])
        y.append(data[t + lookback:t + lookback + horizon, :])
    return np.array(x, dtype=float32), np.array(y, dtype=float32)


def get_ECG_dataloaders(lookback, horizon, train_fraction):
    """Preprocesses data and returns data loaders."""

    path = os.path.join(os.getcwd(), "data")
    file = os.path.join(path, "ECG_E00001.npy")
    data = np.load(file, allow_pickle=True)

    timesteps = len(data)
    split_idx = int(timesteps * train_fraction)
    data_train, data_test = data[:split_idx, :], data[split_idx:, :]

    # Scale all variables to range [0, 1]
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)

    # Save preprocessed dataset as pickle
    processed_data = np.concatenate([data_train, data_test], axis=0)
    with open(f"{path}/ECG_E00001.pkl", "wb") as file:
        pickle.dump(processed_data, file)
        print(f"Save data to {path}.pkl")  

    x_train, y_train = to_sequences(data_train, lookback, horizon)
    x_test, y_test = to_sequences(data_test, lookback, horizon)

    train_dataset = ECGDataset(
        tensor(x_train), 
        tensor(y_train),
        indices = np.arange(0, split_idx)
    )
    test_dataset = ECGDataset(
        tensor(x_test), 
        tensor(y_test),
        indices = np.arange(split_idx, timesteps)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader