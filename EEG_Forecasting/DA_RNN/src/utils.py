import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

def read_data(datapath, debug_mode, latency):
    """Read data in the assigned path.
    Args:
        input_path (str): directory to input dataset.
        debug_mode (bool): use only 100K samples for speed 
        latency (int): #samples force meas. occur recorded before eeg,
                       empirically calculated with a linear model.
    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.
    """
    df = pd.read_csv(datapath, nrows=100000 if debug_mode else None)
    # Read the first 14 columns: EEG measurements(input) 
    # and Force-x measurement(output)
    cols = ['EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-6',
            'EEG-7', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 
            'EEG-13', 'EEG-14', 'Force-x']
    df = df[cols]
    # Based on our linear regression model, EEG occurs 330 samples after Force.
    # That corresponds to 275msec (given the sampling frequency is 1.2kHz)
    # Count the latency: 
    N_trials = df.values.shape[0]
    data = np.concatenate((df.values[latency:,:14], 
                           df.values[:-latency,14].reshape(N_trials-latency,1)), 
                           axis=1)
    df = pd.DataFrame(data, columns=cols) 
    # Scale the data linearly
    min_max_scaler = MinMaxScaler()
    vals_scaled = min_max_scaler.fit_transform(df.values)
    df = pd.DataFrame(vals_scaled)
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    return X, y

def load_model(model, device, path):
    ''' Loads the pretrained saved model for inference on new data'''
    if device: # load the saved model trained on gpu to gpu
        model.load_state_dict(torch.load(path))
    else: # load the saved model trained on gpu/cpu to cpu
        storage = 'cpu'
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    



