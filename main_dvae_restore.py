import numpy as np
import fcntl
from torch import nn, optim
import csv
from torch.utils.data import DataLoader
from utils import *
import torch
import os
import scipy.io as sio
import pdb
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Pool
from dataloader.dataloader_raw import dataloader_raw
from model.dvae import dvae
from model.dvae import restore_eval
torch.multiprocessing.set_start_method('spawn',force=True)

def main(cfg):
    setup_seed(cfg['seed'])
    use_cuda = cfg['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_train.mat'
    val_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_val.mat'
    test_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_test.mat'
    save_params = join_string(cfg,cfg['save_params'],'_')
    
    train_data = dataloader_raw(train_data_path,cfg)
    val_data = dataloader_raw(val_data_path,cfg)
    test_data = dataloader_raw(test_data_path,cfg)

    cfg['input_dim'] = train_data.neural_data.shape[1]
    cfg['output_dim'] = train_data.kin_data.shape[1]

    train_loader = DataLoader(dataset=train_data, batch_size=train_data.__len__(), shuffle=False, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=val_data.__len__(), shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_data.__len__(), shuffle=False, drop_last=True)
    model = dvae(cfg)
    model = model.to(device)
    datasetname_modelname = cfg['dataset_name'] + '_' + cfg['model_name']
    ckpt_path = 'checkpoint/checkpoint_loss_' + datasetname_modelname
    checkpoint = torch.load(ckpt_path +'/' + save_params + '_best_loss_model.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])

    save_path = 'xhat_res/' + datasetname_modelname
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    restore_eval(cfg,model,device,train_loader,save_path+'/'+save_params+'_train.mat')
    restore_eval(cfg,model,device,val_loader,save_path+'/'+save_params+'_val.mat')
    restore_eval(cfg,model,device,test_loader,save_path+'/'+save_params+'_test.mat')

if __name__ == '__main__':
    # mp_flag = 0
    mp_flag = 1
    if mp_flag:
        pool = Pool(processes=5)
    # dataset_names = ['open','C']
    dataset_names = ['open']
    model_names = ['dvae']
    folds = ['1', '2', '3','4','5']
    betas = [0.001]
    alphas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # alphas=[0.8]
    for dataset_name in dataset_names:
        dataset_cfg = read_yaml('config/'+dataset_name+'.yaml')
        if dataset_name == 'open':
            data_names = ['2017012401']
        elif dataset_name == 'C':
            data_names = ['20161007','20161011']
        for model_name in model_names:
            model_cfg = read_yaml('config/'+model_name+'.yaml')
            datasetname_modelname = dataset_name + '_' + model_name
            cfg = dataset_cfg.copy()
            cfg.update(model_cfg)

            cfg_list=[]
            for alpha in alphas:
                for beta in betas:
                    for data_name in data_names:
                        for fold in folds:
                            tuner_params = {'alpha':alpha, 'beta':beta,\
                                            'data_name':data_name, 'fold': fold}
                            cfg.update(tuner_params)
                            cfg_list.append(cfg.copy())
                            if mp_flag:
                                pool.apply_async(main, (cfg_list[-1],))
                            else:
                                main(cfg_list[-1])
    if mp_flag:
        pool.close()
        pool.join()
    print("All process done.")
