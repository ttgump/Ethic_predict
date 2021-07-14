import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
import math

def buildNetwork(layers, activation="relu", dropout=0.):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation=="lrelu":
            net.append(nn.LeakyReLU(negative_slope=0.2))
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)

class autoencoder(nn.Module):
    def __init__(self, input_dim, encodeLayer=[], decodeLayer=[], activation='elu', 
            dropout=0.1, device="cpu"):
        super(autoencoder, self).__init__()
        self.input_dim = input_dim
        self.device=device
        self.encoder = buildNetwork([input_dim]+encodeLayer, activation=activation, dropout=dropout)
        self.enc_out = nn.Linear(encodeLayer[-1], 2)
        self.decoder = buildNetwork([2]+decodeLayer, activation=activation, dropout=dropout)
        self.dec_out = nn.Linear(decodeLayer[-1], input_dim)
        self.mse = nn.MSELoss(reduce="mean").to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self.enc_out(h)
        x_ = self.dec_out(self.decoder(z))
        return z, x_

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.data.cpu().numpy()

    def train_model(self, X, epochs=300, batch_size=256, lr=1e-3, weights='AE_weights.pth.tar'):
        optim_adam = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        dataset = TensorDataset(torch.Tensor(X))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for i in range(epochs):
            loss_val = 0
            for j, (x_batch) in enumerate(dataloader):
                x_batch = x_batch[0]
                x_tensor = Variable(x_batch).to(self.device)
                _, x_ = self.forward(x_tensor)
                loss = self.mse(x_, x_tensor)
                self.zero_grad()
                loss.backward()
                optim_adam.step()

                loss_val += loss.item() * x_tensor.size()[0]
            loss_val = loss_val/X.shape[0]
            print("Epoch {}, loss:{:.4f}".format(i+1, loss_val))

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')

    args = parser.parse_args()

    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()
    x = x.T

    print(x.shape)
    print(y.shape)

    model = autoencoder(input_dim=x.shape[1], encodeLayer=[32,16], decodeLayer=[16,32])

    model.train_model(X=x)

    z = model.encodeBatch(x)

    np.savetxt("AE_latent.txt", z, delimiter=",")