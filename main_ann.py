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
from dataloader.dataloader_xhat_res import dataloader_xhat_res
from model.ann import ann
from model.ann import train, test
torch.multiprocessing.set_start_method('spawn',force=True)

def main(cfg):
    setup_seed(cfg['seed'])
    use_cuda = cfg['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    evalmodelparams=cfg['additional_params'][cfg['dataset_name'].split('_')[1]]
    additional_params = join_string(cfg,evalmodelparams,'_')
    train_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_' + additional_params + '_train.mat'
    val_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_' + additional_params +'_val.mat'

    test_data_path = os.path.join(cfg['dataset_base_dir'],cfg['dataset_dir'], cfg['data_name']) + '_' + cfg['fold'] + '_' + additional_params +'_test.mat'
    save_params = join_string(cfg,cfg['save_params'],'_') + '_'+additional_params
    
    print(train_data_path)
    train_data = dataloader_xhat_res(train_data_path,cfg)
    val_data = dataloader_xhat_res(val_data_path,cfg)
    test_data = dataloader_xhat_res(test_data_path,cfg)

    cfg['input_dim'] = train_data.neural_data.shape[1]
    cfg['output_dim'] = train_data.kin_data.shape[1]
    
    train_loader = DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=val_data.__len__(), shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_data.__len__(), shuffle=False, drop_last=True)

    model = ann(cfg)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])

   
    datasetname_modelname = cfg['dataset_name'] + '_' + cfg['model_name'] + '_' + cfg['dataset_prefix']

    if cfg['use_tensorboard']:
        train_log_dir = os.path.join('tensorboard/tensorboard_r2_'+datasetname_modelname, 'train',save_params)
        val_log_dir = os.path.join('tensorboard/tensorboard_r2_'+datasetname_modelname, 'val', save_params)
        train_writer = SummaryWriter(log_dir=train_log_dir)
        val_writer = SummaryWriter(log_dir=val_log_dir)
    else:
        train_writer = None
        val_writer = None
    if cfg['use_log']:
        csv_dir= os.path.join('log/log_r2_'+datasetname_modelname)
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)
        f = open(csv_dir+'/'+save_params+'.csv','w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(cfg['log_header'])
        train_log = csv_writer
        val_log = csv_writer
    else:
        train_log = None
        val_log = None

    ckpt_path='checkpoint/checkpoint_r2_'+datasetname_modelname
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    best_vaf = -1
    best_epoch = 0
    for epoch in range(1, cfg['epoches'] + 1):
    # for epoch in range(1,2):
        train_rmse, train_cc, train_vaf = train(cfg,model,device,train_loader,optimizer,epoch,train_log,train_writer)
        if epoch >= 1 and epoch % 1 == 0:
            val_rmse, val_cc, val_vaf = test(cfg,model,device,val_loader,epoch,val_log, val_writer)
            if np.isnan(val_vaf):
                print("vaf appear NaN!!!")
                break
            if val_vaf <(1+cfg['patience_improve_percentage'])*best_vaf and epoch-best_epoch>cfg['patience_epoch']:
                print('break at epoch {}'.format(epoch))
                break
            if val_vaf==0 and epoch-best_epoch>=30:
                print('break at epoch {}'.format(epoch))
                break
            if val_vaf > best_vaf:
                best_vaf = val_vaf
                best_epoch = epoch
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_vaf': best_vaf}, ckpt_path + '/' + save_params + '_best_vaf_model.tar')
                print('save done')

if __name__ == '__main__':
    mp_flag=1
    # mp_flag=0
    if mp_flag:
        pool = Pool(processes=10)
    xhat_res=['xhat','res']
    # dataset_names = ['open','C']
    dataset_names = ['open']
    model_names=['ann']
    eval_model_names = ['dvae']
    folds = ['1', '2', '3','4','5']
    alphas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # alphas=[0.9]
    betas = [0.001]
    dataset_cfg = read_yaml('config/xhat_res.yaml')
    for xhat_or_res in xhat_res:
        dataset_cfg['dataset_prefix']=xhat_or_res
        for dataset_name in dataset_names:
            dataset_cfg['dataset_name']=dataset_name
            if dataset_name=='open':
                data_names = ['2017012401']
            elif dataset_name=='C':
                data_names = ['20161007','20161011']
            model_cfg = read_yaml('config/ann.yaml')
            for eval_model_name in eval_model_names:
                dataset_cfg['dataset_dir']=dataset_name+'_'+eval_model_name
                dataset_cfg['dataset_name']=dataset_name+'_'+eval_model_name
                cfg = dataset_cfg.copy()
                cfg.update(model_cfg)
                cfg_list=[]
                for alpha in alphas:
                    for beta in betas:
                        for data_name in data_names:
                            for fold in folds:
                                tuner_params = {'eval_model_name':eval_model_name,'alpha':alpha, 'beta':beta,\
                                                'data_name':data_name, 'fold': fold}
                                print(cfg['smooth_num'])
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
