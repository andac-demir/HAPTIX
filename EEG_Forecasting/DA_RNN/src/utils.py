import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data(datapath, test_mode=False):
    """Read data in the assigned path.

    Args:
        input_path (str): directory to input dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.
    """
    df = pd.read_csv(datapath, nrows=200 if test_mode else None)
    # Read the first 14 columns: EEG measurements(input) 
    # and Force-x measurement(output)
    df = df[['EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-6',
             'EEG-7', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13',
             'EEG-14', 'Force-x']]
    # Based on our linear regression model, EEG occurs 330 samples after Force.
    # That corresponds to 275msec (given the sampling frequency is 1.2kHz)
    # Count the latency: 
    
    # Scale the data linearly
    min_max_scaler = MinMaxScaler()
    vals_scaled = min_max_scaler.fit_transform(df.values)
    df = pd.DataFrame(vals_scaled)
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    return X, y

