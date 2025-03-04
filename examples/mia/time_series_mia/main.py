import os, sys, yaml, numpy as np, matplotlib.pyplot as plt, torch, random

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

from examples.mia.time_series_mia.utils.data_preparation import preprocess_ECG_dataset, preprocess_EEG_dataset, preprocess_LCL_dataset, get_dataloaders
from examples.mia.time_series_mia.utils.model_preparation import create_trained_model_and_metadata
from examples.mia.time_series_mia.utils.models.LSTM import LSTM
from examples.mia.time_series_mia.utils.models.TCN import TCN
from examples.mia.time_series_mia.utils.models.DLinear import DLinear
from examples.mia.time_series_mia.utils.models.TimesNet import TimesNet
from examples.mia.time_series_mia.utils.models.NBeats import NBeats
from examples.mia.time_series_mia.utils.models.TFT import TFT
from examples.mia.time_series_mia.utils.models.WaveNet import WaveNet

from data_handler import IndividualizedInputHandler
from leakpro import LeakPro


if __name__ == "__main__":
    audit_config_path = "audit.yaml"
    train_config_path = "train_config.yaml"

    # Load the yaml files
    with open(audit_config_path, 'r') as file:
        audit_config = yaml.safe_load(file)

    with open(train_config_path, 'r') as file:
        train_config = yaml.safe_load(file)


    random_seed = train_config["run"]["random_seed"]
    log_dir = train_config["run"]["log_dir"]

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
    stride = train_config["data"]["stride"]
    k_lead = train_config["data"]["k_lead"] # number of leading variables to use

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Get data loaders
    path = os.path.join(os.getcwd(), data_dir)
    target_data_file = audit_config["target"]["data_path"].split('/')[-1]

    if dataset == 'ECG' and target_data_file == 'ECG.pkl':
        dataset = preprocess_ECG_dataset(path, lookback, horizon, num_individuals, k_lead=k_lead, stride=stride)
    elif dataset == 'EEG' and target_data_file == 'EEG.pkl':
        dataset = preprocess_EEG_dataset(path, lookback, horizon, num_individuals, k_lead=k_lead, stride=stride)
    elif dataset == 'LCL' and target_data_file == 'LCL.pkl':
        dataset = preprocess_LCL_dataset(path, lookback, horizon, num_individuals, stride=stride)
    else:
        raise Exception(f"Received unknown dataset or mismatching target file: dataset={dataset}, target={target_data_file}.")

    train_loader, test_loader = get_dataloaders(dataset, train_fraction, test_fraction, batch_size=batch_size)

    # Train the model
    input_dim = dataset.input_dim
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
    elif model_name == "WaveNet":
        model = WaveNet(input_dim, horizon)
    else:
        raise NotImplementedError()

    train_loss, test_loss = create_trained_model_and_metadata(model, train_loader, test_loader, epochs, optimizer)

    # Prepare leakpro object
    leakpro = LeakPro(IndividualizedInputHandler, audit_config_path)

    # Run the audit 
    leakpro.run_audit()