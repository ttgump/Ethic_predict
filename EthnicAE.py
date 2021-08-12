import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import math, os
import numpy as np

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

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir='./'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss

class EthnicAE(nn.Module):
    def __init__(self, input_dim, encodeLayers=[], decodeLayers=[], batch_size=256, 
            activation='elu', z_dim=32, gamma=1., n_labels=2, dropoutE=0.1, dropoutD=0.1, device="cuda"):
        super(EthnicAE, self).__init__()
        self.input_dim = input_dim
        self.encodeLayers = encodeLayers
        self.decodeLayers = decodeLayers
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.gamma = gamma
        self.n_labels = n_labels
        self.encoder = buildNetwork(encodeLayers+[z_dim], activation=activation, dropout=dropoutE)
        self.pred = nn.Sequential(nn.Linear(z_dim, n_labels), nn.Softmax(dim=n_labels))
        self.decoder = buildNetwork([z_dim]+decodeLayers, activation=activation, dropout=dropoutD)
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.device = device

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def aeForward(self, x):
        z = self.encoder(x)
        y_pred = self.pred(z)
        x_ = self.decoder(z)
        return z, y_pred, x_

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_,_ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.data.cpu().numpy()

    def train_model(self, X_unlabel, Mask_unlabel, X_label, Mask_label, Y, epochs=300, lr=1e-3, weights='AE_weights.pth.tar'):
        self.train()
        optim_adam = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        dataset_unlabel = TensorDataset(torch.tensor(X_unlabel, dtype=torch.float),
                                        torch.tensor(Mask_unlabel, dtype=torch.long))
        dataloader_unlabel = DataLoader(dataset_unlabel, batch_size=self.batch_size, shuffle=True)
        dataset_label = TensorDataset(torch.tensor(X_label, dtype=torch.float), 
                                        torch.tensor(Mask_label, dtype=torch.long), 
                                        torch.tensor(Y, dtype=torch.long))
        dataloader_label = DataLoader(dataset_label, batch_size=self.batch_size, shuffle=True)
        for i in range(epochs):
            unlabel_loss_val = 0
            for j, (x_batch, mask_batch) in enumerate(dataloader_unlabel):
                x_tensor = Variable(x_batch).to(self.device)
                mask_batch = Variable(mask_batch).to(self.device)
                _, x_, _ = self.forward(x_tensor)
                loss = torch.mean(mask_batch*(x_ - x_tensor)**2)
                self.zero_grad()
                loss.backward()
                optim_adam.step()

                unlabel_loss_val += loss.item() * x_tensor.size()[0]
            unlabel_loss_val = unlabel_loss_val/X_unlabel.shape[0]
            print("Epoch {}, unlabel loss:{:.8f}".format(i+1, unlabel_loss_val))

            label_loss_val = 0
            label_mse_val = 0
            label_ce_val = 0
            for j, (x_batch, mask_batch, y_batch) in enumerate(dataloader_label):
                x_tensor = Variable(x_batch).to(self.device)
                mask_batch = Variable(mask_batch).to(self.device)
                y_tensor = Variable(y_batch).to(self.device)
                _, x_, y_ = self.forward(x_tensor)
                mse_loss = torch.mean(mask_batch*(x_ - x_tensor)**2)
                ce_loss = self.ce(y_, y_tensor)
                loss = mse_loss + self.gamma * ce_loss
                self.zero_grad()
                loss.backward()
                optim_adam.step()

                label_mse_val += mse_loss.item() * x_tensor.size()[0]
                label_ce_val += ce_loss.item() * x_tensor.size()[0]
                label_loss_val += loss.item() * x_tensor.size()[0]
            label_mse_val = label_mse_val/X_label.shape[0]
            label_ce_val = label_ce_val/X_label.shape[0]
            label_loss_val = label_loss_val/X_label.shape[0]
            print("Epoch {}, label loss:{:.8f}, mse loss:{:.8f}, CE loss:{:.8f}".format(i+1, label_loss_val, label_mse_val, label_ce_val))