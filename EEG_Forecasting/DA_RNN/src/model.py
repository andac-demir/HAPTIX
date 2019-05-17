"""
Based on the paper "A Dual Stage Attention Based Recurrent Neural Network
for Time-Series Prediction"
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from visdom import Visdom
import sys

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "10.75.15.194" #(CSL - GPU IP address)
viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)
assert viz.check_connection(timeout_seconds=5), \
'No connection could be formed quickly'

class Encoder(nn.Module):
    def __init__(self, T, input_size,  encoder_num_hidden):
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1, out_features=1, bias=True)

    def forward(self, X):
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # hidden, cell: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T - 1):
            # batch_size * input_size * (2*hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # encoder LSTM
            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.
        Args:
            X
        Returns:
            initial_hidden_states
        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(
            1, X.size(0), self.encoder_num_hidden).zero_())
        return initial_states


class Decoder(nn.Module):
    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
                                        nn.Tanh(),
                                        nn.Linear(encoder_num_hidden, 1))
        self.lstm_layer = nn.LSTM(
            input_size=1, hidden_size=decoder_num_hidden)
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoed, y_prev):
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoed), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1), dim=1)
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))
                # 1 * batch_size * decoder_num_hidden
                d_n = final_states[0]
                # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.
        Args:
            X
        Returns:
            initial_hidden_states
        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(
            1, X.size(0), self.decoder_num_hidden).zero_())
        return initial_states


class DA_rnn(nn.Module):
    def __init__(self, X, y, device, T, encoder_num_hidden, decoder_num_hidden, 
                 batch_size, learning_rate, epochs, train_split, optimizer):
        super(DA_rnn, self).__init__()
        self.device = device
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = True
        self.epochs = epochs
        self.T = T
        self.X = X
        self.y = y
        self.train_split = train_split
        self.optimizer = optimizer.lower()

        if self.device:
            self.Encoder = Encoder(input_size=X.shape[1],
                                encoder_num_hidden=encoder_num_hidden,
                                T=T).cuda()
            self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                                decoder_num_hidden=decoder_num_hidden,
                                T=T).cuda()
        else:
            self.Encoder = Encoder(input_size=X.shape[1],
                            encoder_num_hidden=encoder_num_hidden,
                            T=T)
            self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                                decoder_num_hidden=decoder_num_hidden,
                                T=T)

        # Loss function
        # lr will be reduced when the quantity monitored has stopped decreasing
        self.criterion = nn.MSELoss()
        if self.optimizer == "adam":
            self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                            self.Encoder.parameters()),
                                                lr=self.learning_rate)
            self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                            self.Decoder.parameters()),
                                                lr=self.learning_rate)
        elif self.optimizer == "lbfgs": 
            self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                            self.Encoder.parameters()),
                                                lr=self.learning_rate)
            self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                            self.Decoder.parameters()),
                                                lr=self.learning_rate)           
        else:
            sys.exit("Network optimizer can be selected either as adam or lbfgs.")
        
        self.encoder_scheduler = ReduceLROnPlateau(self.encoder_optimizer, 
                                                   mode='min', factor=0.3,
                                                   patience=0, verbose=True)
        self.decoder_scheduler = ReduceLROnPlateau(self.decoder_optimizer,
                                                    mode='min', factor=0.3,
                                                    patience=0, verbose=True)

        # Training set
        self.train_timesteps = int(self.X.shape[0] * self.train_split)
        self.input_size = self.X.shape[1]

    def train(self):
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps):
                trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
                marker={'color': 'red', 'symbol': 104, 'size': "10"},
                text=["one", "two", "three"], name='1st Trace')
                layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

                vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]:(indices[bs] + self.T - 1)]

                loss = self.train_iteration(x, y_prev, y_gt)
                print("Epoch: %i, Batch: %i, Loss: " %(epoch, int(idx/self.batch_size)), loss)
                self.iter_losses[int(epoch * iter_per_epoch + idx/self.batch_size)] = loss
                idx += self.batch_size
                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(epoch * iter_per_epoch, 
                                                                    (epoch + 1) * iter_per_epoch)])

            print("\nEpochs: %i, Loss: %.8f" %(epoch, self.epoch_losses[epoch]))
            print("Epoch %i training over." %epoch)
            print(50*"*")

            if epoch == self.epochs - 1:
                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.ioff()
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)),
                         self.y, label="True")
                plt.plot(range(self.T, len(y_train_pred) + self.T),
                         y_train_pred, label='Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
                         y_test_pred, label='Predicted - Test')
                plt.legend(loc='upper left')
                plt.show()
            # scheduler steps after each epoch not iteration
            self.encoder_scheduler.step(loss)
            self.decoder_scheduler.step(loss)

    def train_iteration(self, X, y_prev, y_gt):
        # zero gradients
        if self.optimizer == "adam":
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

        if self.device:
            input_weighted, input_encoded = self.Encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda(), 
                                                       requires_grad=False))
            y_pred = self.Decoder(input_encoded, Variable(
                torch.from_numpy(y_prev).type(torch.FloatTensor).cuda(),
                                                  requires_grad=False))
            y_true = Variable(torch.from_numpy(
                y_gt).type(torch.FloatTensor).cuda())
        else:
            input_weighted, input_encoded = self.Encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor),
                                               requires_grad=False))
            y_pred = self.Decoder(input_encoded, Variable(
                torch.from_numpy(y_prev).type(torch.FloatTensor),
                                           requires_grad=False))
            y_true = Variable(torch.from_numpy(
                y_gt).type(torch.FloatTensor))  

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()

        # LBFGS needs to reevaluate the function multiple times, so it is passed 
        # in a closure that allows them to recompute your model. 
        # The closure should clear the gradients, compute the loss, and return it.
        # TODO:
        def closure():
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss

        if self.optimizer == "adam":
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        # TODO:
        elif self.optimizer == "lbfgs":
            self.encoder_optimizer.step(closure)
            self.decoder_optimizer.step(closure)
            
        return loss.item()


    def test(self, on_train=False):
        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)
        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_timesteps - self.T,  batch_idx[j]+ self.train_timesteps - 1)]

            if self.device:
                y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cuda())
                _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).cuda()))
            else: 
                y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor))
                _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))

            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size

        return y_pred