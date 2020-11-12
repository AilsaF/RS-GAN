import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from torchvision import transforms, datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig, clf, scatter, legend, xlim, ylim, hist

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils as myutil
import datasets
seed = 0
torch.manual_seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class generator(torch.nn.Module):
    def __init__(self, d):
        super(generator, self).__init__()
        self.d = d
        self.fc1 = torch.nn.Linear(self.d, g_dim)
        self.fc2 = torch.nn.Linear(g_dim, g_dim)
        self.fc3 = torch.nn.Linear(g_dim, 2)

    def forward(self, z):
        x = self.fc1(z)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x

class discriminator(torch.nn.Module):
    # initializers
    def __init__(self, dim):
        super(discriminator, self).__init__()
        self.dim = dim
        l = [torch.nn.Linear(self.dim, d_dim), nn.LeakyReLU(0.2)]
        for _ in range(Dlayer_num-2):
            l.append(torch.nn.Linear(d_dim, d_dim))
            l.append(nn.LeakyReLU(0.2))
        l.append(torch.nn.Linear(d_dim, 1, bias=False))
        self.net = nn.Sequential(*l)

    def forward(self, input):
        input = input.float()
        return self.net(input)


def get_rsgan_gloss(dis_fake, dis_real):
    scalar = torch.FloatTensor([0]).to(device)
    z = dis_real - dis_fake
    z_star = torch.max(z, scalar.expand_as(z))
    return (z_star + torch.log(torch.exp(z - z_star) + torch.exp(0 - z_star))).mean()


def get_rsgan_dloss(dis_fake, dis_real):
    scalar = torch.FloatTensor([0]).to(device)
    z = dis_fake - dis_real
    z_star = torch.max(z, scalar.expand_as(z))
    return (z_star + torch.log(torch.exp(z - z_star) + torch.exp(0 - z_star))).mean()


def get_gloss(dis_fake, dis_real):
    if model_type == 'rsgan':
        return get_rsgan_gloss(dis_fake, dis_real)
    elif model_type == 'vanilla':
        return (F.softplus(-dis_fake)).mean()

def get_dloss(dis_fake, dis_real):
    if model_type == 'rsgan':
        return get_rsgan_dloss(dis_fake, dis_real)
    elif model_type == 'vanilla':
        return (F.softplus(-dis_real)).mean() + (F.softplus(dis_fake)).mean()


def train():
    dir_name = '{}_{}gaussian_dlr{}_glr{}_ddim{}_gdim{}_gfreq{}_dfreq{}_seed{}/'.format(
                        model_type, gaussian_num, d_lr, g_lr, d_dim, g_dim, g_freq, d_freq, seed)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_path = dir_name + 'models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_path = dir_name + 'images/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    d = discriminator(dim=2)
    g = generator(input_dim)
    d.to(device)
    g.to(device)

    d_optimizer = torch.optim.Adam(d.parameters(), lr=d_lr, betas=(0, 0.9))
    g_optimizer = torch.optim.Adam(g.parameters(), lr=g_lr, betas=(0, 0.9))

    g_losses = []
    d_losses = []
    grad_normD, grad_normG = [], []

    loader = datasets.toy_DataLoder(n=data_num//gaussian_num, batch_size=batch_size, gaussian_num=gaussian_num)
    data_iter = iter(loader)

    for i in range(1, num_iters + 1):
        try:
            x = next(data_iter)[0].to(device)
            z = torch.randn(batch_size, 2).to(device)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)[0].to(device)
            z = torch.randn(batch_size, 2).to(device)

        for _ in range(d_freq):
            d_optimizer.zero_grad()
            x_hat = g(z).detach()
            y_hat = d(x_hat)
            y = d(x)
            d_loss = get_dloss(y_hat, y)
            d_losses.append(d_loss.item())

            d_loss.backward()
            d_optimizer.step()
            if model_type == 'rsgan':
                list(d.children())[-1][-1].weight.data = torch.nn.functional.normalize(
                        list(d.children())[-1][-1].weight.data, dim=1)
            grad_normD.append(myutil.getGradNorm(d))

        for _ in range(g_freq):
            g_optimizer.zero_grad()
            x_hat = g(z)
            y_hat = d(x_hat)
            y = d(x)

            g_loss = get_gloss(y_hat, y)
            g_losses.append(g_loss.item())
            g_loss.backward()
            g_optimizer.step()
            grad_normG.append(myutil.getGradNorm(g))

        if i % print_freq == 0:
            print('Iteration: {}; G-Loss: {}; D-Loss: {};'.format(i, g_loss, d_loss))

        if i % save_freq == 0:
            torch.save(g.state_dict(), save_path + 'G_varseed{}_epoch{}.pth'.format(seed, i))
            torch.save(d.state_dict(), save_path + 'D_varseed{}_epoch{}.pth'.format(seed, i))

        if i == 1 or i % plot_freq == 0:
            fake = g(z)
            gen_data = fake.data.cpu().numpy()
            x = x.detach().cpu().numpy()
            plt.scatter(x[:, 0], x[:, 1], c='r')
            plt.scatter(gen_data[:, 0], gen_data[:, 1], c='y')
            plt.xlim([-5, 5])
            plt.ylim([-5, 5])
            plt.xticks(fontsize=10 * scale)
            plt.yticks(fontsize=10 * scale)
            savefig(img_path + '/gaussian_plot%05d.jpg'%(i/plot_freq))
            clf()
            myutil.saveproj(y.cpu(), y_hat.cpu(), i, save_path)

if __name__ == '__main__':
    g_dim = 128
    d_dim = 128
    Dlayer_num = 3

    data_num = 10000
    batch_size = 128
    g_lr = 1e-4
    d_lr = 1e-4
    input_dim = 2
    num_iters = 20000
    print_freq = 500
    plot_freq = 100
    save_freq = 100
    g_freq = 1
    d_freq = 1

    model_type = 'rsgan'
    # model_type = 'vanilla'
    loss_type = 'log'
    gaussian_num = 5

    img_size = 6
    scale = img_size / 3

    train()
