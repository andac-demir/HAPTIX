import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data(input_path, test_mode=False):
    """Read data in the assigned path.

    Args:
        input_path (str): directory to input dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.
    """
    df = pd.read_csv(input_path, nrows=200 if test_mode else None)
    min_max_scaler = MinMaxScaler()
    vals_scaled = min_max_scaler.fit_transform(df.values)
    df = pd.DataFrame(vals_scaled)
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    return X, y

