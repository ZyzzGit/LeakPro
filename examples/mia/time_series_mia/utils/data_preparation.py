import os, pickle, joblib, torch, random, numpy as np, pandas as pd

from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torch import tensor, float32
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from mne.io import read_raw_edf
from leakpro.utils.logger import logger

class IndividualizedDataset(Dataset):
    def __init__(self, x:tensor, y:tensor, individual_indices:list[tuple[int,int]], scaler, stride):
        self.x = x
        self.y = y
        self.scaler = scaler
        self.stride = stride
        
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
    
    @property
    def input_dim(self):
        return self.x.shape[-1]
    
    @property 
    def output_dim(self):
        return self.y.shape[-1]


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

def preprocess_ELD_dataset(path, lookback, horizon, num_individuals, stride=1, **kwargs):
    """Get and preprocess the dataset."""

    dataset = None
    if os.path.exists(os.path.join(path, "ELD.pkl")):
        with open(os.path.join(path, "ELD.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.lookback != lookback or dataset.horizon != horizon or dataset.num_individuals != num_individuals or dataset.stride != stride:

        df = pd.read_csv(os.path.join(path, "ELD", "LD2011_2014.txt"), delimiter=";", decimal=",")
        # Set a name for date column
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        
        # Resample to hourly frequency and sum the values
        df.set_index("Date", inplace=True)
        df = df.resample('h').sum()

        load_data = []

        for indiv in df.columns:
            load = df[indiv]

            load = np.array(load, dtype=np.float32)

            # Find the index of the first non-zero element
            first_non_zero = np.argmax(load != 0)
        
            # Find the index of the last non-zero element
            last_non_zero = len(load) - np.argmax(load[::-1] != 0) - 1
            
            # Slice the array to remove the first and last zeros
            load = load[first_non_zero:last_non_zero + 1]

            load_data.append(load)

        load_data.sort(key=lambda x: len(x), reverse=True)
        assert num_individuals <= len(load_data), "Too many individuals for dataset"
        load_data = load_data[:num_individuals]
        min_length = min(len(load) for load in load_data)
        load_data = [load[:min_length] for load in load_data]

        data = np.array(load_data)

        data = np.expand_dims(data, -1)
        #Scale data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(np.concatenate(data))
        data_scaled = data_scaled.reshape(data.shape)

        x = []  # lists to store samples for all individuals
        y = []
        for time_series in data_scaled:
            # Create sequences separately for each individual
            xi, yi = to_sequences(time_series, lookback, horizon, stride)
            x.append(xi)
            y.append(yi)

        num_samples_per_individual = len(x[0])
        individual_indices = [(0 + num_samples_per_individual*i, num_samples_per_individual*(i+1)) for i in range(len(x))]

        # Concatenate samples and save dataset
        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        dataset = IndividualizedDataset(x, y, individual_indices, scaler, stride)
        with open(os.path.join(path, "ELD.pkl"), "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}/ELD.pkl") 

    return dataset 

def preprocess_LCL_dataset(path, lookback, horizon, num_individuals, stride=1, **kwargs):
    """Get and preprocess the dataset."""

    dataset = None
    if os.path.exists(os.path.join(path, "LCL.pkl")):
        with open(os.path.join(path, "LCL.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.lookback != lookback or dataset.horizon != horizon or dataset.num_individuals != num_individuals or dataset.stride != stride:
        
        df = pd.read_csv(os.path.join(path, "LCL", "daily_dataset.csv"))
        individuals = list(df["LCLid"].value_counts(sort=True, ascending=False)[:num_individuals].index)
        assert len(individuals) == num_individuals

        load_data = []
        time_data = []

        for indiv in individuals:
            indiv_df = df[df["LCLid"] == indiv]
            load = indiv_df["energy_mean"]
            time = indiv_df["day"]

            # Convert float32 and np.datetimes
            load = np.array(load, dtype=np.float32)
            time = np.array(time, dtype=np.datetime64)

            # Remove duplicates and replace by mean of the duplicates
            unique_time, _ = np.unique(time, return_index=True)
            avg_load = np.zeros_like(unique_time, dtype=np.float32)
            for i, t in enumerate(unique_time):
                mask = time == t
                avg_load[i] = np.mean(load[mask])  # Take average of duplicates
            time = unique_time
            load = avg_load

            # Generate the expected time range (30-minute intervals)
            start, end = time[0], time[-1]
            expected_time = np.arange(start, end + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))

            # Fill missing timestamps with NaN
            filled_load = np.full_like(expected_time, np.nan, dtype=np.float32)
            filled_load[np.isin(expected_time, time)] = load

            time = expected_time
            load = filled_load

            # Interpolate missing values linearly
            nan_mask = np.isnan(load)
            load[nan_mask] = np.interp(
                np.flatnonzero(nan_mask),
                np.flatnonzero(~nan_mask),
                load[~nan_mask]
            )

            load_data.append(load)
            time_data.append(time)

        seq_len = min(len(ts) for ts in load_data)
        time_data = [ts[:seq_len] for ts in time_data]
        load_data = [ts[:seq_len] for ts in load_data]
        time_data = np.expand_dims(np.array(time_data), -1)
        load_data = np.expand_dims(np.array(load_data), -1)

        #Scale data
        scaler = MinMaxScaler()
        data = np.concatenate(load_data)
        data_scaled = scaler.fit_transform(data)
        data_scaled = data_scaled.reshape(load_data.shape)

        x = []  # lists to store samples for all individuals
        y = []
        for time_series in data_scaled:
            # Create sequences separately for each individual
            xi, yi = to_sequences(time_series, lookback, horizon, stride)
            x.append(xi)
            y.append(yi)

        num_samples_per_individual = len(x[0])
        individual_indices = [(0 + num_samples_per_individual*i, num_samples_per_individual*(i+1)) for i in range(len(x))]

        # Concatenate samples and save dataset
        x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
        dataset = IndividualizedDataset(x, y, individual_indices, scaler, stride)
        with open(os.path.join(path, "LCL.pkl"), "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}/LCL.pkl") 

    return dataset 


def preprocess_ECG_dataset(path, lookback, horizon, num_individuals, k_lead=12, stride=1, **kwargs):
    """Get and preprocess the dataset."""

    dataset = None
    timesteps = 5000   # number of timesteps in raw time series
    if os.path.exists(os.path.join(path, "ECG.pkl")):
        with open(os.path.join(path, "ECG.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.lookback != lookback or dataset.horizon != horizon or dataset.num_individuals != num_individuals or dataset.stride != stride or dataset.num_variables != k_lead:
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
        dataset = IndividualizedDataset(x, y, individual_indices, scaler, stride)
        with open(os.path.join(path, "ECG.pkl"), "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}/ECG.pkl") 

    return dataset


def get_edf_time_series(edf_data, k_lead, num_time_steps):
    time_series = edf_data.get_data()
    time_series = time_series.T                     # transpose to get sample dimension first
    return time_series[:num_time_steps, :k_lead]    # select first num_timesteps of the k first variables

def preprocess_EEG_dataset(path, lookback, horizon, num_individuals, k_lead=3, stride=1, num_time_steps=75000, **kwargs):
    """Get and preprocess the dataset. Assuming subset of first 100 patients (EEG/000)."""
    # num_time_steps is the fixed number of steps to use from each individual; cutting the longer series and ignoring shorter ones
    # default: 75000 steps (equivalent to 5 minutes when sampling at 250Hz)

    dataset = None
    if os.path.exists(os.path.join(path, "EEG.pkl")):
        with open(os.path.join(path, "EEG.pkl"), "rb") as f:
            dataset = joblib.load(f)

    if dataset is None or dataset.lookback != lookback or dataset.horizon != horizon or dataset.num_individuals != num_individuals or dataset.stride != stride or dataset.num_variables != k_lead:
        data_path = os.path.join(path, 'EEG/000')
        subjects = os.listdir(data_path)
        random.shuffle(subjects)   # randomize order of individuals

        individuals = []    # individuals[i] is the token (time series) of individual i closest to num_time_steps (but not shorter) in length
        for subject in subjects:
            best_token = None
            for session in os.listdir(os.path.join(data_path, subject)):
                dirs = os.listdir(os.path.join(data_path, f'{subject}/{session}'))
                if len(dirs) > 1:
                    raise Exception(f'Expected single montage, but {subject}/{session} has {len(dirs)} montage definitions!')
                montage_definition = dirs[0]
                for token in os.listdir(os.path.join(data_path, f'{subject}/{session}/{montage_definition}')):
                    file = os.path.join(data_path, f'{subject}/{session}/{montage_definition}/{token}')
                    data = read_raw_edf(file, verbose=False)
                    if data.info['sfreq'] != 250 or data.n_times < num_time_steps:
                        continue    # skip data not sampled at a frequency of 250 Hz or long enough
                    if best_token == None or data.n_times < best_token.n_times:
                        best_token = data   # update best token to be data closest to num_time_steps

            if best_token:
                individuals.append(best_token)

        # Get the shortest individual time series and trim to the requested length
        # (this ensures we discard as few time steps as possible)
        individuals.sort(key=lambda ts: ts.n_times)
        selected_individuals = individuals[:num_individuals]
        if len(selected_individuals) < num_individuals:
            logger.warning(f"num_individuals = {num_individuals} but only found {len(selected_individuals)} with num_time_steps >= {num_time_steps}. Proceeding with {len(selected_individuals)} individuals.")
        trimmed_selected_time_series = np.array([get_edf_time_series(ind, k_lead, num_time_steps) for ind in selected_individuals])

        # IQR scaling
        scaler = RobustScaler()
        data = np.concatenate(trimmed_selected_time_series)
        data_scaled = scaler.fit_transform(data)

        # Reshape to include individual dimension again (this is OK since all time series have equal length)
        data_scaled = data_scaled.reshape(trimmed_selected_time_series.shape)

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
        dataset = IndividualizedDataset(x, y, individual_indices, scaler, stride)
        with open(f"{path}/EEG.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}/EEG.pkl") 

    return dataset

def preprocess_dataset(dataset, path, lookback, horizon, num_individuals, **kwargs):
    if dataset == "ECG":
        return preprocess_ECG_dataset(path, lookback, horizon, num_individuals, **kwargs)
    if dataset == "EEG":
        return preprocess_EEG_dataset(path, lookback, horizon, num_individuals, **kwargs)
    if dataset == "LCL":
        return preprocess_LCL_dataset(path, lookback, horizon, num_individuals, **kwargs)
    if dataset == "ELD":
        return preprocess_ELD_dataset(path, lookback, horizon, num_individuals, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")


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