import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff

def getGradNorm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print name
    total_norm = total_norm ** (1. / 2)
    return total_norm

def plot_losses(g_losses, d_losses, grad_normG, grad_normD, dir_path):
    plt.plot(g_losses)
    plt.title('g loss')
    plt.savefig(dir_path + "/g_losses.jpg")
    plt.clf()
    plt.plot(d_losses)
    plt.title('d loss')
    plt.savefig(dir_path + "/d_losses.jpg")
    plt.clf()
    plt.plot(grad_normG)
    plt.title('G norm (square root of sum G norm)')
    plt.savefig(dir_path + "/grad_normG.jpg")
    plt.clf()
    plt.plot(grad_normD)
    plt.title('D norm (square root of sum D norm)')
    plt.savefig(dir_path + "/grad_normD.jpg")
    plt.clf()
    np.save(dir_path + 'Dloss.npy', d_losses)
    np.save(dir_path + 'Gloss.npy', g_losses)
    np.save(dir_path + 'Dgram.npy', grad_normD)
    np.save(dir_path + 'Ggram.npy', grad_normG)

def saveproj(y, y_hat, i ,dir_path):
    plt.scatter(y_hat.data.numpy(), np.zeros(y_hat.shape[0]), label='Fake', s=100)
    plt.scatter(y.data.numpy(), np.zeros(y.shape[0]), label='Real', s=50)
    plt.title('Disc_projection')
    plt.legend()
    plt.savefig(dir_path + '/dprojection_epoch%05d.pdf'%(i), bbox_inches='tight')
    plt.clf()

# EMA trick
def soft_copy_param(ema_netG, netG, beta):
    netG_para = netG.state_dict()
    for name, param in ema_netG.named_parameters():
        param.data *= beta
        param.data += (1-beta) * netG_para[name].cpu().data
