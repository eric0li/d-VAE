import numpy as np
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
from model.dvae import train, test
torch.multiprocessing.set_start_method('spawn',force=True)

def main(cfg):
    setup_seed(cfg['seed'])
    use_cuda = cfg['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_train.mat'
    val_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_val.mat'
    test_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_test.mat'
    save_params = join_string(cfg,cfg['save_params'],'_')
    
    print(train_data_path)
    train_data = dataloader_raw(train_data_path,cfg)
    val_data = dataloader_raw(val_data_path,cfg)
    test_data = dataloader_raw(test_data_path,cfg)

    cfg['input_dim'] = train_data.neural_data.shape[1]
    cfg['output_dim'] = train_data.kin_data.shape[1]

    train_loader = DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=val_data.__len__(), shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_data.__len__(), shuffle=False, drop_last=True)
    
    model = dvae(cfg)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])

    datasetname_modelname = cfg['dataset_name'] + '_' + cfg['model_name']

    if cfg['use_tensorboard']:
        train_log_dir = os.path.join('tensorboard/tensorboard_loss_'+datasetname_modelname, 'train',save_params)
        val_log_dir = os.path.join('tensorboard/tensorboard_loss_'+datasetname_modelname, 'val', save_params)
        train_writer = SummaryWriter(log_dir=train_log_dir)
        val_writer = SummaryWriter(log_dir=val_log_dir)
    else:
        train_writer = None
        val_writer = None

    if cfg['use_log']:
        csv_dir= os.path.join('log/log_loss_'+datasetname_modelname)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        f = open(csv_dir+'/'+save_params+'.csv','w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(cfg['log_header'])
        train_log = csv_writer
        val_log = csv_writer
    else:
        train_log = None
        val_log = None

    ckpt_path='checkpoint/checkpoint_loss_'+datasetname_modelname
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    best_loss = 10000000
    best_epoch=0
    for epoch in range(1, cfg['epoches'] + 1):
    # for epoch in range(1, 2):
        train_loss =train(cfg,model,device,train_loader,optimizer,epoch,train_log,train_writer)
        if epoch >= 1 and epoch % 1 == 0:
            val_loss = test(cfg,model,device,val_loader,epoch,val_log,val_writer)
            if np.isnan(val_loss):
                print("loss appear NaN!!!")
                break
            if best_loss < 0:
                if val_loss > (1+cfg['patience_improve_percentage'])*best_loss and epoch-best_epoch>cfg['patience_epoch']:
                    print('break at epoch {}'.format(epoch))
                    break
            else:
                if val_loss > (1-cfg['patience_improve_percentage'])*best_loss and epoch-best_epoch>cfg['patience_epoch']:
                    print('break at epoch {}'.format(epoch))
                    break
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': best_loss}, ckpt_path + '/' + save_params + '_best_loss_model.tar')
                print('save done')
if __name__ == '__main__':
    mp_flag=1
    # mp_flag=0
    if mp_flag:
        pool = Pool(processes=10)
    dataset_names = ['open','C']
    model_names = ['dvae']
    folds = ['1', '2', '3','4','5']
    betas = [0.001]
    alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for dataset_name in dataset_names:
        dataset_cfg = read_yaml('config/'+dataset_name+'.yaml')
        if dataset_name == 'open':
            data_names = ['2017012401']
        elif dataset_name == 'C':
            data_names = ['20161007','20161011']

        for model_name in model_names:
            model_cfg = read_yaml('config/'+model_name+'.yaml')
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
