import torch
# import torch.nn as nn
from torch import nn,optim
import torch.nn.functional as F
from utils import *

def test(cfg,model,device,test_loader,epoch=0,test_log=None,test_writer=None):
    model.eval()
    outputs=np.array([])
    targets=np.array([])
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            if outputs.size == 0:
                outputs = output.cpu().detach().numpy()
            else:
                outputs = np.concatenate([outputs, output.cpu().detach().numpy()], 0)
            if targets.size == 0:
                targets = target.cpu().detach().numpy()
            else:
                targets = np.concatenate([targets, target.cpu().detach().numpy()], 0)
    rmse = RMSE(targets, outputs)
    cc = CC(targets, outputs)
    vaf = VAF(targets, outputs)
    if not (test_log is None):
        test_log.writerow([epoch,'val',rmse,cc,vaf])
    if epoch % cfg['log_interval'] ==0:
        print("Epoch: {:>5d}, val_rmse  : {:.4f}, val_cc  : {:.4f}, val_vaf  : {:.4f}".format(epoch,rmse,cc,vaf))
    return rmse,cc,vaf

def train(cfg,model,device,train_loader,optimizer,epoch,train_log=None,train_writer=None):
    model.train()
    outputs=np.array([])
    targets=np.array([])
    for data,target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(target,output)
        loss.backward()
        optimizer.step()
        if outputs.size==0:
            outputs=output.cpu().detach().numpy()
        else:
            outputs=np.concatenate([outputs,output.cpu().detach().numpy()],0)
        if targets.size==0:
            targets=target.cpu().detach().numpy()
        else:
            targets=np.concatenate([targets,target.cpu().detach().numpy()],0)
    rmse = RMSE(targets,outputs)
    cc = CC(targets,outputs)
    vaf=VAF(targets,outputs)
    if not (train_log is None):
        train_log.writerow([epoch,'train',rmse,cc,vaf])
    if epoch % cfg['log_interval'] ==0:
        print("Epoch: {:>5d}, train_rmse: {:.4f}, train_cc: {:.4f}, train_vaf: {:.4f}".format(epoch,rmse,cc,vaf))
    return rmse,vaf,cc

class ann(nn.Module):
    def __init__(self,cfg):
        super(ann,self).__init__()
        #self.BN = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(cfg['input_dim'],cfg['hidden_dim1'])
        #self.BN1 = nn.BatchNorm1d(hidden_dim1)
        self.linear2 = nn.Linear(cfg['hidden_dim1'],cfg['hidden_dim2'])
        #self.BN2 = nn.BatchNorm1d(hidden_dim2)
        self.linear3 = nn.Linear(cfg['hidden_dim2'],cfg['output_dim'])


    def forward(self,x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        self.logits=out.detach()
        out = self.linear3(out)
        return out
