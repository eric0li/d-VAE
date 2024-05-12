import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import pdb
from torch.autograd import Function
from utils import *
import scipy.io as sio
import copy
def restore_eval(cfg,model,device,test_loader,save_path):
    model.eval()
    # pdb.set_trace()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        [total_loss, recon_loss, kld_loss, reg_loss, reg_hat_loss, z, y_pred, y_pred_hat] = model(x, y)
        prior_distribution = model._prior_encode(y)
        prior_mu = prior_distribution[:,:cfg['latent_dim']]
        prior_logvar = prior_distribution[:,cfg['latent_dim']:]
        distribution = model._encode(x)
        mu = distribution[:,:cfg['latent_dim']]
        logvar = distribution[:,cfg['latent_dim']:]
        kld_loss2=kl_divergence(mu,logvar,prior_mu,prior_logvar)
        x_hat = model._decode(z)
        recon_loss2 = reconstruction_loss(x,x_hat,'poisson')
        print('recon loss res:'+str((recon_loss-recon_loss2).cpu().detach().numpy()))
        distribution_hat = model._encode(x_hat)
        mu_hat = distribution_hat[:,:cfg['latent_dim']]
        logvar_hat = distribution_hat[:,cfg['latent_dim']:]
        if cfg['save_xhat_res']:
            sio.savemat(save_path,{'prior_mu':prior_mu.cpu().detach().numpy(),'prior_logvar':prior_logvar.cpu().detach().numpy(),'mu':mu.cpu().detach().numpy(), 'logvar':logvar.cpu().detach().numpy(),
                                   'x_hat':x_hat.cpu().detach().numpy(), 'x':x.cpu().detach().numpy(), 'y_pred':y_pred.cpu().detach().numpy(), 'y':y.cpu().detach().numpy(),
                                   'z':z.cpu().detach().numpy(),'mu_hat':mu_hat.cpu().detach().numpy(),'logvar_hat':logvar_hat.cpu().detach().numpy(),'res':(x-x_hat).cpu().detach().numpy()})

def test(cfg,model,device,test_loader,epoch,test_log=None,test_writer=None):
    model.eval()
    losses=[]
    for x, y in  test_loader:
        x, y = x.to(device), y.to(device)
        [total_loss, recon_loss, kld_loss, reg_loss, reg_hat_loss, z, y_pred, y_pred_hat] = model(x, y)
        losses.append([total_loss.item(), recon_loss.item(), kld_loss.item(), reg_loss.item(),\
                      reg_hat_loss.item()])
    losses = np.array(losses).reshape([-1,len(losses[0])])
    losses = np.mean(losses, axis=0)
    if not (test_writer is None):
        test_writer.add_scalar('Loss', losses[0], epoch)
        test_writer.add_scalar('recon_loss', losses[1], epoch)
        test_writer.add_scalar('kld_loss', losses[2], epoch)
        test_writer.add_scalar('reg_loss', losses[3], epoch)
        test_writer.add_scalar('reg_hat_loss', losses[4], epoch)
    if not (test_log is None):
        test_log.writerow([epoch,'val',losses[0],losses[1],losses[2],losses[3],losses[4]])
    return losses[0]

def train(args,model,device,train_loader,optimizer,epoch,train_log=None,train_writer=None):
    model.train()
    losses=[]
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        [total_loss, recon_loss, kld_loss, reg_loss, reg_hat_loss, z, y_pred, y_pred_hat] = model(x, y)
        losses.append([total_loss.item(), recon_loss.item(), kld_loss.item(), reg_loss.item(),\
                      reg_hat_loss.item()])
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    losses = np.array(losses).reshape([-1,len(losses[0])])
    losses = np.mean(losses, axis=0)
    if not (train_writer is None):
        train_writer.add_scalar('Loss', losses[0], epoch)
        train_writer.add_scalar('recon_loss', losses[1], epoch)
        train_writer.add_scalar('kld_loss', losses[2], epoch)
        train_writer.add_scalar('reg_loss', losses[3], epoch)
        train_writer.add_scalar('reg_hat_loss', losses[4], epoch)
    if not (train_log is None):
        train_log.writerow([epoch,'train',losses[0],losses[1],losses[2],losses[3],losses[4]])
    return losses[0]

def kl_divergence(mu, logvar, p_mu=None, p_logvar=None):
    if p_mu is None:
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    else:
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - p_logvar - \
                                            (((mu - p_mu).pow(2) + logvar.exp())/p_logvar.exp()), dim=1), dim=0)
    return kld_loss

def reconstruction_loss(x, x_recon, distribution='poisson'):
    # x_recon = x_recon.clamp(min=1e-7, max=1e7)
    recon_loss =  torch.mean(torch.sum(x_recon-x*torch.log(x_recon),dim=1),dim=0)
    return recon_loss

class dvae(nn.Module):
    def __init__(self,cfg):
        super(dvae, self).__init__()
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']
        self.latent_dim= cfg['latent_dim']
        self.likelihood = cfg['likelihood']
        input_dim = cfg['input_dim']
        output_dim = cfg['output_dim']
        prior_hidden_dims = [cfg['prior_hidden_dim1']]
        hidden_dims = [cfg['hidden_dim1'],cfg['hidden_dim2']]

        # Build Prior Encoder
        self.prior_layers_dim=[output_dim, *prior_hidden_dims]
        pe_modules = []
        for in_dim, out_dim in zip(self.prior_layers_dim[:-1], self.prior_layers_dim[1:]):
            pe_modules.append(nn.Linear(in_dim, out_dim))
            pe_modules.append(nn.ReLU(True))
        pe_modules.append(nn.Linear(self.prior_layers_dim[-1], 2*self.latent_dim))
        self.prior_encoder = nn.Sequential(*pe_modules)

        # Build Encoder
        self.layers_dim = [input_dim, *hidden_dims] # [96, (300, 100)]
        e_modules = []
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            e_modules.append(nn.Linear(in_dim, out_dim))
            e_modules.append(nn.ReLU(True))
        e_modules.append(nn.Linear(self.layers_dim[-1], 2*cfg['latent_dim']))
        self.encoder = nn.Sequential(*e_modules)


        # Build Decoder
        self.layers_dim.reverse()
        d_modules = []
        d_modules.append(nn.Linear(self.latent_dim, self.layers_dim[0]))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            d_modules.append(nn.ReLU(True))
            d_modules.append(nn.Linear(in_dim, out_dim))
        d_modules.append(nn.Softplus())
        self.decoder = nn.Sequential(*d_modules)
        
        self.regress = nn.Sequential(nn.Linear(self.latent_dim, output_dim))

    def _prior_encode(self, x):
        return self.prior_encoder(x)

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, y):
        prior_distribution = self._prior_encode(y)
        prior_mu = prior_distribution[:,:self.latent_dim]
        prior_logvar = prior_distribution[:,self.latent_dim:]

        distribution = self._encode(x)
        mu = distribution[:,:self.latent_dim]
        logvar = distribution[:,self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        x_hat = self._decode(z)
        y_pred = self.regress(mu)
        
        distribution_hat = self._encode(x_hat)
        mu_hat = distribution_hat[:,:self.latent_dim]
        logvar_hat = distribution_hat[:,self.latent_dim:]
        y_pred_hat = self.regress(mu_hat)

        recon_loss = reconstruction_loss(x, x_hat, self.likelihood)
        kld_loss = kl_divergence(mu, logvar, prior_mu, prior_logvar)
        reg_loss = F.mse_loss(y, y_pred)
        reg_hat_loss = F.mse_loss(y, y_pred_hat)
        loss = recon_loss + self.beta * kld_loss +\
                self.alpha * 0.5 * (reg_loss + reg_hat_loss)
        return [loss, recon_loss, kld_loss, reg_loss, reg_hat_loss, z, y_pred, y_pred_hat]

