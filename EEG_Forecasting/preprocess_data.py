'''
    MATLAB file: YData.m
    __header__: b'MATLAB 5.0 MAT-file, Platform: PCWIN64, 
    Created on: Fri Nov 30 11:14:21 2018' 
    __version__: 1.0                    
    __globals__: []  
    N_conditions: 18
    within each condition the data have been segmented into trials.
    Sequence_ [[ 10  20  30  40  50  60  70  80  90 100 110 120 130 140
                 150 160 170 180]] (18 different conditions)
    data['EEGSeg_Ch'] --> (1 x 14 numpy array) x (1 x 18 numpy array) 
                          x (116 x 101)
    data['EMGSeg_Ch'] --> 1 x 4 numpy array x (1 x 18 numpy array) 
                          x (116 x 101)
    data['ForceSeg_Ch'] -- > 1 x 3 numpy array x (1 x 18 numpy array) 
                          x (116 x 101)
    preprocess_data.py converts this .mat file to a csv file, where
    we restructure the data as 2D matrix.
'''
from scipy.io import loadmat
import pandas as pd
from argparse import ArgumentParser, ArgumentTypeError

data = loadmat('YData.m')

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("-cond", type=int, help=("Enter the condition " 
                        "number. Acceptable entries: 0, 1, 2, ..., 17"), 
                        default=0)
    parser.add_argument("-eeg_ch", type=int, help=("Enter the EEG channel."), 
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
    return args

def parseChannel(cond, eeg_ch):
        df = pd.DataFrame(columns=['EEG','EMG-1','EMG-2','EMG-3','EMG-4',
                                   'Force-x', 'Force-y', 'Force-z'])
        df.iloc[:,0] = data['EEGSeg_Ch'][0,eeg_ch][0,cond].flatten()
        for i in range(4):
            df.iloc[:,i+1] = data['EMGSeg_Ch'][0,i][0,cond].flatten()
        for i in range(3):
            df.iloc[:,i+5] = data['ForceSeg_Ch'][0,i][0,cond].flatten()
        df.to_csv('haptix_eegch%d_cond%d.csv'%(eeg_ch, cond))

'''
    Parses all EEG channels given one condition.
'''       
def parseAllEEG(cond):
        df = pd.DataFrame(columns=['EEG-1','EEG-2','EEG-3','EEG-4','EEG-5',
                                   'EEG-6','EEG-7','EEG-8','EEG-9','EEG-10',
                                   'EEG-11','EEG-12','EEG-13','EEG-14','EMG-1',
                                   'EMG-2','EMG-3','EMG-4','Force-x', 
                                   'Force-y','Force-z'])
        for i in range(14):
            df.iloc[:,i] = data['EEGSeg_Ch'][0,i][0,cond].flatten()
        for i in range(4):
            df.iloc[:,i+14] = data['EMGSeg_Ch'][0,i][0,cond].flatten()
        for i in range(3):
            df.iloc[:,i+18] = data['ForceSeg_Ch'][0,i][0,cond].flatten()
        df.to_csv('haptix_alleeg_cond%d.csv'%(cond))

'''
    Parses all EEG channels for all conditions.
'''       
def parseAll():
        df = pd.DataFrame(columns=['EEG-1','EEG-2','EEG-3','EEG-4','EEG-5',
                                   'EEG-6','EEG-7','EEG-8','EEG-9','EEG-10',
                                   'EEG-11','EEG-12','EEG-13','EEG-14','EMG-1',
                                   'EMG-2','EMG-3','EMG-4','Force-x', 
                                   'Force-y','Force-z'])
        temp1 = df.copy()
        temp2 = df.copy()
        for cond in range(18):
            for i in range(14):
                temp1.iloc[:,i] = data['EEGSeg_Ch'][0,i][0,cond].flatten()
            for i in range(4):
                temp1.iloc[:,i+14] = data['EMGSeg_Ch'][0,i][0,cond].flatten()
            for i in range(3):
                temp1.iloc[:,i+18] = data['ForceSeg_Ch'][0,i][0,cond].\
                                                            flatten()
            df = df.append(temp1, ignore_index=True)
            temp1 = temp2.copy()
        df.to_csv('haptix_alleeg_allcond.csv')

def main():
    args = passLegalArgs()
    if not(args.all_eeg or args.all):
        parseChannel(args.cond, args.eeg_ch)
    else:
        if args.all_eeg:
            parseAllEEG(args.cond)
        if args.all:
            parseAll()

if __name__ == "__main__":
    main()





