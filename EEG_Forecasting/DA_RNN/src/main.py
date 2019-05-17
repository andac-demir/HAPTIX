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
import utils
from torchviz import make_dot

PATH = "../pretrained/savedModel.pt"

def parseArgs():
    # Parameters settings
    parser = argparse.ArgumentParser(description="DA-RNN")

    # Dataset setting
    parser.add_argument('--datapath', type=str, default="../../haptix_alleeg_allcond.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=64, help='input batch size [64]')
    parser.add_argument('--latency', type=int, default=330, help='latency between force and eeg measurements [330]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--debug_mode', type=str2bool, default="false", help=("If using the deubgging mode, only take 100K samples"
                                                                              "from the csv file for shorter computation time."
                                                                              " [false]"))
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train [20]')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate [0.01] reduced when the loss reaches a plateu')
    parser.add_argument('--save', type=str2bool, default="true", help='save [true] the trained model into the dir. pretrainedModel')
    parser.add_argument('--train_split', type=int, default=0.7, help='ratio of training data [0.7]')
    parser.add_argument('--optimizer', type=str, default="adam", help='parameter optimization [adam]. Choose adam or lbfgs.')
    parser.add_argument('--run_pretrained', type=str2bool, default="false", 
                        help='Runs the pretrained model instead of training a model [false]')
    args = parser.parse_args()
    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def trainMode(model, device, save):
    if device == True:
        print("Training on: ", torch.cuda.get_device_name(0))
        print("Number of devices: ", torch.cuda.device_count()) 
    else:
        print("Training on CPU.")  
    model.train()
    # Prediction
    y_pred = model.test()
    # Save the trained model:
    if save:
        torch.save(model.state_dict(), PATH)
        print("Trained model saved to the path: ", PATH)
    
    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.title("Loss by Batch")
    plt.xlabel("Batch")
    plt.ylabel("MSE Loss")
    plt.savefig("loss_by_batch.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.title("Loss by Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.savefig("loss_by_epoch.png")
    plt.close(fig2)

    return y_pred


def printAttributes():
    print(("Model Attributes:\n"
           "weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,\n"
           "of shape (hidden_size, input_size) for k = 0. Otherwise, the shape is\n"
           "(hidden_size, num_directions * hidden_size)\n"
           "weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,\n"
           "of shape (hidden_size, hidden_size)\n"
           "bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,\n"
           "of shape (hidden_size)\n"
           "bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,\n"
           "of shape (hidden_size)\n"))

def main():
    args = parseArgs()
    # Read dataset
    X, y = utils.read_data(args.datapath, args.debug_mode, args.latency)
    print("\nData extracted and standardized.\n")

    # Initialize model
    device = True if torch.cuda.is_available() else False
    model = DA_rnn(X, y, device, args.ntimestep, args.nhidden_encoder, 
                   args.nhidden_decoder, args.batchsize, args.lr, args.epochs,
                   args.train_split, args.optimizer)

    # Train mode
    if args.run_pretrained == False:
        y_pred = trainMode(model, device, args.save)
        print('Finished Training')
    # run the pretrained model and plt the model architecture on Visdom
    else: 
        utils.load_model(model, device, PATH)
        printAttributes()
        print("Model Structure:")
        for k in sorted(model.state_dict().keys()):
            v = model.state_dict()[k]
            print(k, v.shape)
            print(type(v))
            model.state_dict()[k] = Variable(v, requires_grad=True)
        print('\nTotal parameters:', sum(v.numel() for v in model.state_dict().\
                                                                    values()))
        print("\nPretrained model is loaded successfully.")
        y_pred = model.test()
 
    fig3 = plt.figure()
    start_idx=0
    plt.plot(y_pred[start_idx:start_idx+500], label='Predicted')
    plt.plot(model.y[start_idx+model.train_timesteps:
                     start_idx+model.train_timesteps+500], 
             label="True")
    plt.title("Prediction and Ground Truth")
    plt.legend(loc='upper left')
    plt.savefig("pred_vs_gt.png")
    plt.close(fig3)


if __name__ == "__main__":
    main()
