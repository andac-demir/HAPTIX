# for plotting  in the remote machine
# add these 3 line sin the very beginning of the source code
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, concat
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV
import pickle
from sklearn.metrics import mean_absolute_error

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("--cond", type=int, help=("Enter the condition " 
                        "number. Acceptable entries: 0, 1, 2, ..., 17"), 
                        default=0)
    parser.add_argument("--eeg_ch", type=int, help=("Enter the EEG channel "), 
                        default=0)
    parser.add_argument("--all_eeg", type=str2bool, help=("Enter true to parse "
                        "and load all EEG channels for one condition."), 
                        default='false')
    parser.add_argument("--all", type=str2bool, help=("Enter true to parse "
                        "and load all EEG channels for all conditions."), 
                        default='false')
    parser.add_argument("--n_in", type=int, help=("Number of samples from  "
                        "past."), default=1)
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

def series_to_supervised(data, n_in, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data) 
    cols, names = list(), list()
	# input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n_out) -->n_out=1 always in this case 
    for i in range(1):
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

def preprocessing(filename, n_in):
    dataset = read_csv(filename, header=0, index_col=0)
    values = dataset.values
    # ensures all data is float
    values = values.astype('float32')
    reframed = series_to_supervised(values, n_in)
    return reframed

def train_testSplit(data):
    # split into train and test sets
    values = data.values
    n_train_samples = int(values.shape[0] * 0.8)
    train = values[:n_train_samples, :]
    test = values[n_train_samples:, :]
    return train, test

# splits train and test sets into input and outputs
def input_outputSplit(train, test, n_in, n_features, all_eeg, all):
    n_obs = n_in * n_features
    if not(all_eeg or all):
        trainX, trainY = train[:, :n_obs], train[:, -n_features]
        testX, testY = test[:, :n_obs], test[:, -n_features]
    else:
        trainX, trainY = train[:,:n_obs], train[:,-n_features:-n_features+14]
        testX, testY = test[:,:n_obs], test[:,-n_features:-n_features+14]    
    return trainX, trainY, testX, testY

def plotResults(actualEEG, predictedEEG, channel):
    plt.plot(actualEEG, label='Actual EEG')
    plt.plot(predictedEEG, label='Predicted EEG')
    plt.title("EEG Channel:%d"%channel)
    plt.legend()
    plt.savefig("EEG Channel:%d"%channel)
    plt.close()

def main():
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
    data = preprocessing(filename, args.n_in)
    train, test = train_testSplit(data)
    trainX, trainY, testX, testY = input_outputSplit(train, test, args.n_in, 
                                                     n_features,
                                                     args.all_eeg, args.all)
    # Cross-validate to find the best alpha value for lasso regression
    if args.all_eeg or args.all:
        clf = RidgeCV(alphas=[1e-20, 1e-15, 1e-10,1e-9, 1e-8, 1e-7, 
                              1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    else:
        clf = LassoCV(alphas=[1e-20, 1e-15, 1e-10,1e-9, 1e-8, 1e-7, 
                              1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    
    
    reg = clf.fit(trainX, trainY)
    coefs = clf.coef_  # dense np.array
    intercept = clf.intercept_ # intercept
    print("Weight matrix:\n", coefs) 
    print("Intercept:\n", intercept)
    print("Best alpha regularizer for lasso regression: ", reg.alpha_)

    if args.all_eeg or args.all:
        file = open('ridgeParams.txt', 'wb')
    else:
        file = open('lassoParams.txt', 'wb')

    pred = reg.predict(testX)
    mas = mean_absolute_error(testY, pred)
    print("Mean sabsolute error: %.8f" %mas)
    dict = {'weight matrix': coefs, 'intercept': intercept,
            'alpha': reg.alpha_, 'mas': mas}
    pickle.dump(dict, file)
    file.close()
    
    if not(args.all_eeg or args.all):
        plotResults(testY, pred, args.eeg_ch)
        # To see the results more clearly
        plotResults(testY[0:100], pred[0:100], args.eeg_ch) 
    else:
        for i in range(14):
            plotResults(testY[:,i], pred[:,i], i)
            # To see the results more clearly
            plotResults(testY[0:100, i], pred[:100, i], i)


if __name__ == "__main__":
    main()

