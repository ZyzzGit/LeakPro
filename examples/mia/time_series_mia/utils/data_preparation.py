import os
import pickle
import joblib
import torch
import random
import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler
from torch import tensor, float32
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from mne.io import read_raw_edf

class IndividualizedDataset(Dataset):
    def __init__(self, x:tensor, y:tensor, individual_indices:list[tuple[int,int]], scaler):
        self.x = x
        self.y = y
        self.scaler = scaler
        
        self.lookback = x.size(1)
        self.horizon = y.size(1)
        self.num_variables = y.size(2)

        self.individual_indices = individual_indices    # individual_indices[i] is a tuple [start_index, end_index) for individual i
        self.num_individuals = len(individual_indices)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, ...], self.y[idx, ...]
    
    def subset(self, indices):
        return Subset(self, indices)


def read_mat_data(path, file):
    file_path = os.path.join(path, file)
    return loadmat(file_path)['val'].T  # transpose to get shape (#timesteps, #variables)

def to_sequences(data, lookback, horizon, stride):
    x, y = [], []
    timesteps = len(data)
    num_samples = timesteps - (lookback + horizon) + 1

    for t in range(0, num_samples, stride):
        x.append(data[t:t + lookback, :])
        y.append(data[t + lookback:t + lookback + horizon, :])
    return tensor(np.array(x), dtype=float32), tensor(np.array(y), dtype=float32)

def preprocess_ECG_dataset(path, lookback, horizon, num_individuals, k_lead=12, stride=1):
    """Get and preprocess the dataset."""

    dataset = None
    timesteps = 5000   # number of timesteps in raw time series
    if os.path.exists(os.path.join(path, "ECG.pkl")):
        with open(os.path.join(path, "ECG.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.lookback != lookback or dataset.horizon != horizon or dataset.num_individuals != num_individuals:
        raw_data_path = os.path.join(path, 'ECG')
        individual_files = random.sample(os.listdir(raw_data_path), num_individuals)
        all_raw_time_series = np.array(list(filter(
            lambda ts: ts.shape[0] == timesteps, # keep time series with 5000 timesteps (only 52/10344 individuals don't satisfy this) 
            map(lambda f: read_mat_data(raw_data_path, f), individual_files) 
        )))
        all_raw_time_series = all_raw_time_series[..., :k_lead]

        # IQR scaling
        scaler = RobustScaler()
        data = np.concatenate(all_raw_time_series)
        data_scaled = scaler.fit_transform(data)

        # Reshape to include individual dimension again (this is OK since all time series have equal length)
        data_scaled = data_scaled.reshape(all_raw_time_series.shape)

        x = []  # lists to store samples for all individuals
        y = []
        for time_series in data_scaled:
            # Create sequences separately for each individual
            xi, yi = to_sequences(time_series, lookback, horizon, stride)
            x.append(xi)
            y.append(yi)

        # Keep track of sample indices for each individual time series
        num_samples_per_individual = len(x[0])
        individual_indices = [(0 + num_samples_per_individual*i, num_samples_per_individual*(i+1)) for i in range(len(x))]

        # Concatenate samples and save dataset
        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        dataset = IndividualizedDataset(x, y, individual_indices, scaler)
        with open(f"{path}/ECG.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}/ECG.pkl") 

    return dataset


def get_edf_time_series(edf_data, k_lead):
    time_series = edf_data.get_data()
    time_series = time_series.T # transpose to get sample dimension first
    return time_series[:,:k_lead] # select k first variables

def preprocess_EEG_dataset(path, lookback, horizon, num_individuals, k_lead=3, stride=1):
    """Get and preprocess the dataset. Assuming subset of first 100 patients (EEG/000)."""

    dataset = None
    if os.path.exists(os.path.join(path, "EEG.pkl")):
        with open(os.path.join(path, "EEG.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.lookback != lookback or dataset.horizon != horizon or dataset.num_individuals != num_individuals or dataset.num_variables != k_lead:
        data_path = os.path.join(path, 'EEG/000')
        subjects = os.listdir(data_path)
        random.shuffle(subjects)   # randomize order of individuals

        individuals = []    # individuals[i][j] is the j:th token (time series) of individual i
        for subject in subjects:
            individual_data = []    # data for current subject
            for session in os.listdir(os.path.join(data_path, subject)):
                dirs = os.listdir(os.path.join(data_path, f'{subject}/{session}'))
                if len(dirs) > 1:
                    raise Exception(f'Expected single montage, but {subject}/{session} has {len(dirs)} montage definitions!')
                montage_definition = dirs[0]
                for token in os.listdir(os.path.join(data_path, f'{subject}/{session}/{montage_definition}')):
                    file = os.path.join(data_path, f'{subject}/{session}/{montage_definition}/{token}')
                    data = read_raw_edf(file, verbose=False)
                    if data.info['sfreq'] == 250:   # only keep data sampled at a frequency of 250 Hz
                        individual_data.append(data)

            if (len(individual_data) > 0):
                individuals.append(individual_data)

        # Get time series of individuals with similar size
        individuals.sort(key=lambda ls: len(ls))
        individuals = [[get_edf_time_series(data, k_lead) for data in ind] for ind in individuals[:num_individuals]]

        # IQR scaling
        scaler = RobustScaler()
        all_values = np.vstack([time_series for ind in individuals for time_series in ind])
        scaler.fit(all_values)
        scaled_individuals = [
            [scaler.transform(time_series) for time_series in ind]
            for ind in individuals
        ]

        x = []  # lists to store samples for all individuals and series
        y = []
        curr_idx = 0  # index of dataset to be constructed (x and y)
        individual_indices = [] # keep track of sample indices for each individual (will span over multiple time series)
        for individual in scaled_individuals:
            start_idx = curr_idx
            individual_length = 0
            for time_series in individual:
                # Create sequences separately for each time series
                xi, yi = to_sequences(time_series, lookback, horizon, stride)
                x.append(xi)
                y.append(yi)
                individual_length += len(xi)

            curr_idx += individual_length
            individual_indices.append((start_idx, curr_idx))

        # Concatenate samples and save dataset
        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        dataset = IndividualizedDataset(x, y, individual_indices, scaler)
        with open(f"{path}/EEG.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}/EEG.pkl") 

    return dataset


def get_dataloaders(dataset: IndividualizedDataset, train_fraction=0.5, test_fraction=0.3, batch_size=128):

    train_size = int(train_fraction * dataset.num_individuals)
    test_size = int(test_fraction * dataset.num_individuals)
    train_individuals, test_individuals = train_test_split(np.arange(dataset.num_individuals), train_size=train_size, test_size=test_size)
        
    train_ranges = [dataset.individual_indices[i] for i in train_individuals]
    test_ranges = [dataset.individual_indices[i] for i in test_individuals]

    train_indices = np.concatenate([np.arange(start, stop) for (start, stop) in train_ranges], axis=0)
    test_indices = np.concatenate([np.arange(start, stop) for (start, stop) in test_ranges], axis=0)

    train_subset = Subset(dataset, train_indices) 
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader