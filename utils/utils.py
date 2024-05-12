import numpy as np
import random
import torch
import yaml
import pdb
from torch import nn
from sklearn.metrics import mean_squared_error
from numpy.linalg import pinv,inv

def get_poisson_likelihood(x,x_hat):
    # x_hat = np.maximum(x_hat,0)
    likelihood =  np.mean(np.sum(x_hat-x*np.log(x_hat+1e-10),axis=1),axis=0)
    return likelihood
def get_recon_criteria(x,x_hat):
    assert x.shape==x_hat.shape,"array shape is not same"
    likelihood = get_poisson_likelihood(x,x_hat)
    rmse = RMSE(x, x_hat)
    vaf=VAF(x,x_hat)
    return likelihood, rmse, vaf

def RMSE(y,y_est):
    assert y.shape==y_est.shape,"array shape is not same"
    return np.sqrt(mean_squared_error(y,y_est))

def CC(y, y_est):
    assert y.shape==y_est.shape,"array shape is not same"
    cc=0.0
    for i in range(y.shape[1]):
        cc+=np.corrcoef(y[:,i],y_est[:,i])[0,1]
    cc=cc/y.shape[1]
    return cc


def VAF(y,y_est):
    assert y.shape==y_est.shape,"array shape is not same"
    # y---N*feature-dim
    esp=1e-10
    y_mean = np.mean(y,0)
    v = np.sum((y-y_est)**2,0)/(np.sum((y-y_mean)**2,0)+esp)
    v = 1-v
    v[v<0]=0
    v = np.mean(v)
    return v

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

def join_string(cfg,params,delimiter):
    # return delimiter.join([str(cfg[i]) for i in cfg[params]])
    # pdb.set_trace()
    return delimiter.join([str(cfg[i]) for i in params])

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

