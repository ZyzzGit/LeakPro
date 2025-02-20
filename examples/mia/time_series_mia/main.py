import os, sys, yaml, numpy as np, matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from examples.mia.time_series_mia.utils.data_preparation import preprocess_ECG_dataset, get_ECG_dataloaders
from examples.mia.time_series_mia.utils.model_preparation import create_trained_model_and_metadata
from examples.mia.time_series_mia.utils.models.LSTM import LSTM
from examples.mia.time_series_mia.utils.models.TCN import TCN
from examples.mia.time_series_mia.utils.models.DLinear import DLinear
from examples.mia.time_series_mia.utils.models.TimesNet import TimesNet
from examples.mia.time_series_mia.utils.models.NBeats import NBeats
from examples.mia.time_series_mia.utils.models.TFT import TFT


if __name__ == "__main__":
    audit_config_path = "audit.yaml"
    train_config_path = "train_config.yaml"

    # Load the yaml files
    with open(audit_config_path, 'r') as file:
        audit_config = yaml.safe_load(file)

    with open(train_config_path, 'r') as file:
        train_config = yaml.safe_load(file)


    epochs = train_config["train"]["epochs"]
    batch_size = train_config["train"]["batch_size"]
    optimizer = train_config["train"]["optimizer"]

    lookback = train_config["data"]["lookback"]
    horizon = train_config["data"]["horizon"]
    num_individuals = train_config["data"]["num_individuals"]
    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    dataset = train_config["data"]["dataset"]
    data_dir = train_config["data"]["data_dir"]

    # Define constants
    input_dim = 12 # input size will be the amount of variables in the Time-Series

    # Get data loaders
    path = os.path.join(os.getcwd(), data_dir)
    dataset = preprocess_ECG_dataset(path, lookback, horizon, num_individuals)
    train_loader, test_loader = get_ECG_dataloaders(dataset, train_fraction, test_fraction, batch_size=batch_size)

    # Train the model
    model_name = audit_config["target"]["model_class"]

    if model_name == "LSTM":
        model = LSTM(input_dim, horizon)
    elif model_name == "TCN":
        model = TCN(input_dim, horizon)
    elif model_name == "DLinear":
        model = DLinear(input_dim, lookback, horizon)
    elif model_name == "TimesNet":
        model = TimesNet(input_dim, lookback, horizon)
    elif model_name == "NBeats":
        model = NBeats(input_dim, lookback, horizon)
    elif model_name == "TFT":
        model = TFT(input_dim, lookback, horizon)
    else:
        raise NotImplementedError()

    train_loss, test_loss = create_trained_model_and_metadata(model, train_loader, test_loader, epochs)

    from ECG_handler import ECGInputHandler
    from leakpro import LeakPro

    # Prepare leakpro object
    leakpro = LeakPro(ECGInputHandler, audit_config_path)

    # Run the audit 
    leakpro.run_audit()