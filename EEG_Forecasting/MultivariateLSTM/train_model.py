from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

'''
    (Persitance Forecasting)
    Dataset is a 2D numpy array where the columns represent the fetures 
    (variables) and rows represent each time step t-2, t-1, t, t+1, t+2, etc.
    n_in is the number of samples to go back in the past and
    n_out is the number of samples to predict in the future at each time step
    We predict the EEG at the current time step
    Supervised data is a 2D array. 
    Rows show the time steps and columns show the features
    Features are past EEG data of one selected channel (out of 15), 
    4 EMG channels and force measurement (z-dimension) and
    surface category (digital trigger) in this order, 7 features in total  
    There are n_features * n_in number of columns in total.
'''

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("-cond", type=int, help=("Enter the condition " 
                        "number. Acceptable entries: 0, 1, 2, ..., 17"), 
                        default=0)
    parser.add_argument("-eeg_ch", type=int, help=("Enter the EEG channel "), 
                        default=0)
    parser.add_argument("-all_eeg", type=str2bool, help=("Enter true to parse "
                        "and load all EEG channels for one condition."), 
                        default='false')
    parser.add_argument("-all", type=str2bool, help=("Enter true to parse "
                        "and load all EEG channels for all conditions."), 
                        default='false')
    args = parser.parse_args()
    return args

def passLegalArgs():
    args = getArgs()
    acceptable_conds = list(range(0,18))
    acceptable_eeg_ch = list(range(0,14))
    assert args.cond in acceptable_conds, "Condition must be 0, 1, 2, ... 17"
    assert args.eeg_ch in acceptable_eeg_ch, "EEG ch. must be 0, 1, 2, ... 13"
    assert args.all_eeg!=True or args.all!=True
    return args

def series_to_supervised(data, n_in, n_out, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data) 
	cols, names = list(), list()
	# input sequence (t-n_in, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n_out)
	for i in range(n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
def preprocessing(filename, n_in, n_out):
    dataset = read_csv(filename, header=0, index_col=0)
    values = dataset.values
    # ensures all data is float
    values = values.astype('float32')
    # feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out)
    return scaler, reframed
 
def train_testSplit(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_samples = int(values.shape[0] * 0.8)
    train = values[:n_train_samples, :]
    test = values[n_train_samples:, :]
    return train, test

# splits train and test sets into input and outputs
def input_outputSplit(train, test, n_in, n_features, all_eeg, all):
    n_obs = n_in * n_features
    if not(all_eeg or all):
        train_X, train_y = train[:, :n_obs], train[:, -n_features]
        test_X, test_y = test[:, :n_obs], test[:, -n_features]
    else:
        train_X, train_y = train[:,:n_obs], train[:,-n_features:-n_features+14]
        test_X, test_y = test[:,:n_obs], test[:,-n_features:-n_features+14]    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_features))
    return train_X, train_y, test_X, test_y
 
# design network and train the network
def trainNetwork(train_X, train_y, test_X, test_y, all_eeg, all):
    model = Sequential()
    # input shape that is passed into the network: [timesteps, features]
    model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
    if not(all_eeg or all): # train per channel
        model.add(Dense(1))
    else: # train all 14 channels 
        model.add(Dense(14))
    sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    # fit network, batch_size is 116 because there are 116 trials in total
    history = model.fit(train_X, train_y, epochs=200, batch_size=116, 
                        validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Loss by Epochs')
    plt.legend()
    plt.show()
    return model

def plotResults(actual_eeg, predicted_eeg, channel):
    plt.plot(actual_eeg, label='Actual EEG')
    plt.plot(predicted_eeg, label='Predicted EEG')
    plt.title("EEG Channel:%d"%channel)
    plt.legend()
    plt.show()
    plt.close()

'''
    Calculates the standard error of regression (S-value).
    A goodness of fit test designed for nonlinear regression like 
    R^2 in linear regression.
    For our model's predicitons to be within +/-5% of the observed values
    to be useful, it must have a required precision S less than 2.5
'''
def stdErr(actual, pred):
    sum = 0
    N_samples = len(actual)
    for i in range(N_samples):
        sum += (pred[i] - actual[i]) ** 2 
    return np.sqrt(sum/(N_samples-2)) 

def main():
    # specify the number of lag samples, n_out must be always 1
    n_in, n_out = 5, 1
    args = passLegalArgs()
    if not(args.all_eeg or args.all):
        filename = "../haptix_eegch%d_cond%d.csv" %(args.eeg_ch, args.cond)
        n_features = 8 
    else:
        if args.all_eeg:
            filename = "../haptix_alleeg_cond%d.csv" %(args.cond)    
            n_features = 21
        if args.all:
            filename = "../haptix_alleeg_allcond.csv"    
            n_features = 21 
            
    scaler, reframed = preprocessing(filename, n_in, n_out)
    train, test = train_testSplit(reframed)
    train_X, train_y, test_X, test_y = input_outputSplit(train, test, 
                                                         n_in, n_features, 
                                                         args.all_eeg,
                                                         args.all)
    model = trainNetwork(train_X, train_y, test_X, test_y, args.all_eeg, 
                                                              args.all)
    # make a prediction
    yhat = model.predict(test_X)
    # reshape test input from 3D back to 2D
    test_X = test_X.reshape((test_X.shape[0], n_in*n_features))
    # inverse scaling for forecast using the scaled EEG data of past time steps
    if not(args.all_eeg or args.all):
        inv_yhat = np.concatenate((yhat, test_X[:,-(n_features-1):]), axis=1)
    else:
        inv_yhat = np.concatenate((yhat, test_X[:,-(n_features-14):]), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    # inverse scaling for actual EEG
    if not(args.all_eeg or args.all):
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
    else:
        inv_y = np.concatenate((test_y, test_X[:, -(n_features-14):]), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    if not(args.all_eeg or args.all):
        actual = inv_y[:,0]
        pred = inv_yhat[:,0] 
        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(actual, pred))
        print('Test RMSE: %.8f' %rmse)
        plotResults(actual, pred, args.eeg_ch)
        # To see the results more clearly
        plotResults(actual[0:100], pred[0:100], args.eeg_ch)
        print(stdErr(actual, pred))
    else:
        for i in range(14):
            actual = inv_y[:,i]
            pred = inv_yhat[:,i]
            # calculate RMSE
            rmse = np.sqrt(mean_squared_error(actual, pred))
            print('Test RMSE: %.8f' %rmse)
            plotResults(actual, pred, i)
            # To see the results more clearly
            plotResults(actual[0:100], pred[0:100], i)
            print(stdErr(actual, pred))
        

if __name__ == "__main__":
    main()

