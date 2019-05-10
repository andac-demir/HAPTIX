"""
    Based on the paper "A Dual Stage Attention Based Recurrent Neural Network"
    for Time-Series Prediction (Qin, 2017)
"""
# First 2 lines required to plot in the GPU cluster.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from model import *
from utils import *

def parseArgs():
    # Parameters settings
    parser = argparse.ArgumentParser(description="DA-RNN")

    # Dataset setting
    parser.add_argument('--datapath', type=str, default="../../haptix_alleeg_allcond.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=64, help='input batch size [64]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train [20]')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate [0.01] reduced when the loss reaches a plateu')
    parser.add_argument('--save', type=str2bool, default="true", help='save [true] the trained model into the dir. pretrainedModel')
    parser.add_argument('--train_split', type=int, default=0.7, help='ratio of training data [0.7]')
    args = parser.parse_args()
    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    args = parseArgs()
    # Read dataset
    X, y = read_data(args.datapath)
    print("Data extracted and standardized.")

    # Initialize model
    device = True if torch.cuda.is_available() else False
    model = DA_rnn(X, y, device, args.ntimestep, args.nhidden_encoder, 
                   args.nhidden_decoder, args.batchsize, args.lr, args.epochs,
                   args.train_split)

    # Train
    if device == True:
        print("Training on: ", torch.cuda.get_device_name(0))
        print("Number of devices: ", torch.cuda.device_count()) 
    else:
        print("Training on CPU.")  
    model.train()

    # Save the trained model:
    if args.save:
        PATH = "../pretrained/savedModel.model"
        torch.save(model.state_dict(), PATH)
        print("Trained model saved to the path: ", PATH)

    # Prediction
    y_pred = model.eval()

    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.title("Losses by Batch")
    plt.xlabel("Batch")
    plt.ylabel("MSE Loss")
    plt.savefig("1.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.title("Losses by Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.savefig("2.png")
    plt.close(fig2)

    # Plot the first 500 prediction
    fig3 = plt.figure()
    plt.plot(y_pred[:500], label='Predicted')
    plt.plot(model.y[model.train_timesteps:500], label="True")
    plt.title("Prediction and Ground Truth")
    plt.legend(loc = 'upper left')
    plt.savefig("3.png")
    plt.close(fig3)
    print('Finished Training')


if __name__ == "__main__":
    main()
