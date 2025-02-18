import os
import pickle
import joblib
import torch
import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch import tensor, float32
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

class ECGDataset(Dataset):
    def __init__(self, x:tensor, y:tensor, individual_indices:list[tuple[int,int]]):
        self.x = x
        self.y = y

        self.lookback = x.size(1)
        self.horizon = y.size(1)

        self.individual_indices = individual_indices    # individual_indices[i] is a tuple [start_index, end_index) for individual i
        self.num_individuals = len(individual_indices)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def subset(self, indices):
        return Subset(self, indices)
        #return ECGDataset(self.x[indices], self.y[indices], self.individual_indices)
    
def read_data(path, file):
    file_path = os.path.join(path, file)
    return loadmat(file_path)['val'].T  # transpose to get shape (#timesteps, #variables)

def to_sequences(data, lookback, horizon):
    x, y = [], []
    timesteps = len(data)
    num_samples = timesteps - (lookback + horizon) + 1

    for t in range(num_samples):
        x.append(data[t:t + lookback, :])
        y.append(data[t + lookback:t + lookback + horizon, :])
    return tensor(np.array(x), dtype=float32), tensor(np.array(y), dtype=float32)

def preprocess_ECG_dataset(path, lookback, horizon):
    """Get and preprocess the dataset."""

    dataset = None
    timesteps = 5000   # number of timesteps in raw time series
    if os.path.exists(os.path.join(path, "ECG.pkl")):
        with open(os.path.join(path, "ECG.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.x.shape[2] != lookback or dataset.y.shape[2] != horizon:
        raw_data_path = os.path.join(path, 'ECG')
        all_raw_time_series = list(filter(
            lambda ts: ts.shape[0] == timesteps, # keep time series with 5000 timesteps (only 52/10344 individuals don't satisfy this) 
            map(lambda f: read_data(raw_data_path, f), os.listdir(raw_data_path)[0:10]) # TODO: delete pkl and remove '[0:10]' for testing on all data
        ))

        # Scale all variables to range [0, 1]
        scaler = MinMaxScaler()
        data = np.concatenate(all_raw_time_series)
        data_scaled = scaler.fit_transform(data)

        # Reshape to include individual dimension again (this is OK since all time series have equal length)
        num_variables = data_scaled.shape[-1]
        num_individuals = len(all_raw_time_series)
        data_scaled = data_scaled.reshape((num_individuals, timesteps, num_variables))

        x = []  # lists to store samples for all individuals
        y = []
        for time_series in data_scaled:
            # Create sequences separately for each individual
            xi, yi = to_sequences(time_series, lookback, horizon)
            x.append(xi)
            y.append(yi)

        # Keep track of sample indices for each individual time series
        num_samples_per_individual = len(x[0])
        individual_indices = [(0 + num_samples_per_individual*i, num_samples_per_individual*(i+1)) for i in range(len(x))]

        # Concatenate samples and save dataset
        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        dataset = ECGDataset(x, y, individual_indices)
        with open(f"{path}/ECG.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}.pkl") 

    return dataset

def get_ECG_dataloaders(dataset: ECGDataset, train_fraction=0.5, test_fraction=0.3):

    train_individuals, test_individuals = train_test_split(np.arange(dataset.num_individuals), train_size=train_fraction, test_size=test_fraction)
    
    train_ranges = [dataset.individual_indices[i] for i in train_individuals]
    test_ranges = [dataset.individual_indices[i] for i in test_individuals]

    train_indices = np.concatenate([np.arange(start, stop) for (start, stop) in train_ranges], axis=0)
    test_indices = np.concatenate([np.arange(start, stop) for (start, stop) in test_ranges], axis=0)

    train_subset = Subset(dataset, train_indices) 
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

    return train_loader, test_loader