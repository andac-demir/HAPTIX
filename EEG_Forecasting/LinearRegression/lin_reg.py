# for plotting  in the remote machine
# add these 3 lines in the very beginning of the source code
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, concat
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
                        default='true')
    parser.add_argument("--n_in", type=int, help=("Number of samples from  "
                        "past."), default=1)
    parser.add_argument("--latency", type=str2bool, help=("Chronistically, "
                        "force signals have an impact on the EEG signals and "
                        "EEG signals have an impact on the EMG signals. "
                        "Instead of assuming taking the Force, EEG and EMG "
                        "samples at the same time instant t, find the timing "
                        "differences between each one of them."), 
                        default=False)
    parser.add_argument("--verbose", type=str2bool, help=("If true, displays "
                        "the best regularizer for regression and the plots"),
                        default=False)
    parser.add_argument("--no_emg", type=str2bool, help=("If true, do not use "
                        "the emg signals."), default=False)
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

def preprocessing(filename, n_in, eeg_delay, emg_delay, latency, no_emg):
    '''
    Order of occurrence: Force --> Eeg --> Emg
    Latency between force and eeg is --> eeg_delay
    Latency between eeg and emg is --> emg_delay
    Latency between force and emg is --> eeg_delay + emg_delay
    '''
    dataset = read_csv(filename, header=0, index_col=0)
    values = dataset.values.astype('float32')
    
    if no_emg == False:
        if latency==True and values.shape[1]==21:
            N_rows = values.shape[0]
            noLatencyValues = np.zeros((N_rows-emg_delay-eeg_delay,21))
            noLatencyValues[:,:14] = values[eeg_delay:N_rows-emg_delay,:14]
            noLatencyValues[:,14:18] = values[eeg_delay+emg_delay:,14:18]
            noLatencyValues[:,18:] = values[:N_rows-eeg_delay-emg_delay,18:] 
            reframed = series_to_supervised(noLatencyValues, n_in)
        elif latency==True and values.shape[1]==8:
            N_rows = values.shape[0]
            noLatencyValues = np.zeros((N_rows-emg_delay-eeg_delay,8))
            noLatencyValues[:,:1] = values[eeg_delay:N_rows-emg_delay,:1]
            noLatencyValues[:,1:5] = values[eeg_delay+emg_delay:,1:5]
            noLatencyValues[:,5:] = values[:N_rows-eeg_delay-emg_delay,5:] 
            reframed = series_to_supervised(noLatencyValues, n_in)
        else:    
            reframed = series_to_supervised(values, n_in)
    else:
        if latency==True and values.shape[1]==21:
            N_rows = values.shape[0]
            values = np.delete(values, [14,15,16,17], 1)
            noLatencyValues = np.zeros((N_rows-eeg_delay, 17))
            noLatencyValues[:,:14] = values[eeg_delay:N_rows,:14]
            noLatencyValues[:,14:] = values[:N_rows-eeg_delay,14:] 
            reframed = series_to_supervised(noLatencyValues, n_in)
        elif latency==True and values.shape[1]==8:
            N_rows = values.shape[0]
            values = np.delete(values, [1,2,3,4], 1)
            noLatencyValues = np.zeros((N_rows-eeg_delay,4))
            noLatencyValues[:,:1] = values[eeg_delay:N_rows,:1]
            noLatencyValues[:,1:] = values[:N_rows-eeg_delay,1:] 
            reframed = series_to_supervised(noLatencyValues, n_in)
        elif latency==False and values.shape[1]==21:    
            values = np.delete(values, [14,15,16,17], 1)
            reframed = series_to_supervised(values, n_in)
        elif latency==False and values.shape[1]==8:
            values = np.delete(values, [1,2,3,4], 1)
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

def plotResults(actualEEG, predictedEEG, channel, plts_on):
    if plts_on == True:
        plt.plot(actualEEG, label='Actual EEG')
        plt.plot(predictedEEG, label='Predicted EEG')
        plt.title("EEG Channel:%i"%channel)
        plt.legend()
        path = "../RegressionResults/eegChannel_%i"%(channel)
        plt.savefig(path)
        plt.show()
        #plt.close()
    else:
        pass

def main():
    args = passLegalArgs()
    n_in = 1 # AR order
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
    
    # if latency is true: explore the latency using past 600 samples 
    # (0.5 sec-1.2kHz)
    # otherwise just use the current time step (a poor assumption - 
    # that force(t), emg(t) and eeg(t) impact eeg(t+1))
    max_range = 480 
    if args.latency == True and args.no_emg == False:
        step_emg, step_eeg = 20, 20
    
    if args.latency == True and args.no_emg == True:
        step_emg = max_range
        step_eeg = 60

    if args.latency == False:
        step_emg, step_eeg = max_range, max_range

    # create a heatmap to determine the latency between 
    # the force-eeg and eeg-emg 
    N_rows = (max_range - step_eeg) / step_eeg + 1
    N_cols = (max_range - step_emg) / step_emg + 1
    heatMap = np.zeros((N_rows, N_cols))

    row = 0
    for eeg_delay in list(range(0, max_range, step_eeg)):
        col = 0
        for emg_delay in list(range(0, max_range, step_emg)):
            if args.no_emg == False:
                print("Solving for EEG delay %i and EMG delay %i" %(eeg_delay, 
                                                                    emg_delay))
            if args.no_emg == True:
                print("Solving for EEG delay %i" %(eeg_delay))

            data = preprocessing(filename, n_in, eeg_delay, emg_delay, 
                                 args.latency, args.no_emg)
            train, test = train_testSplit(data)
            trainX, trainY, testX, testY = input_outputSplit(train, test,  
                                                             args.n_in,
                                                             n_features,
                                                             args.all_eeg, 
                                                             args.all)
            # Cross-validate to find the best alpha value for lasso regression
            if args.all_eeg==True or args.all==True:
                clf = RidgeCV(alphas=[1e-20, 1e-15, 1e-10,1e-9, 1e-8, 1e-7, 
                                      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
            else:
                clf = LassoCV(alphas=[1e-20, 1e-15, 1e-10,1e-9, 1e-8, 1e-7, 
                                      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
            
            reg = clf.fit(trainX, trainY)
            coefs = clf.coef_  # dense np.array
            intercept = clf.intercept_ # intercept
            if args.verbose == True:
                print("Weight matrix:\n", coefs) 
                print("Intercept:\n", intercept)
                print("Best alpha regularizer for regression: ", reg.alpha_)

            if args.all_eeg or args.all:
                file = open('ridgeParams.txt', 'wb')
            else:
                file = open('lassoParams.txt', 'wb')

            pred = reg.predict(testX)
            mas = mean_absolute_error(testY, pred)
            mse = mean_squared_error(testY, pred)
            heatMap[row,col] = mse
            if args.no_emg == False:
                col += 1
            print("Mean squared error: %.16f" %mse)
            dict = {'weight matrix': coefs, 'intercept': intercept,
                    'alpha': reg.alpha_, 'mas': mas, 'mse': mse, 
                    'eeg_delay': eeg_delay, 'emg_delay': emg_delay}
            pickle.dump(dict, file)
            file.close()
            
            if not(args.all_eeg or args.all):
                plotResults(testY, pred, args.eeg_ch, args.verbose)
                # To see the results more clearly
                plotResults(testY[0:100], pred[0:100], args.eeg_ch, 
                            args.verbose) 
            else:
                for i in range(14):
                    plotResults(testY[:,i], pred[:,i], i, args.verbose)
                    # To see the results more clearly
                    plotResults(testY[0:100, i], pred[:100, i], i, 
                                args.verbose)
            del clf   
        row += 1
    # Save the heatMap for latencies
    np.save('latencyHeatmap', heatMap)


if __name__ == "__main__":
    main()

