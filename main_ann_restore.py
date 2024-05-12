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

    train_data = dataloader_xhat_res(train_data_path,cfg)
    val_data = dataloader_xhat_res(val_data_path,cfg)
    test_data = dataloader_xhat_res(test_data_path,cfg)
    
    cfg['input_dim'] = train_data.neural_data.shape[1]
    cfg['output_dim'] = train_data.kin_data.shape[1]

    train_loader = DataLoader(dataset=train_data, batch_size=train_data.__len__(), shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=val_data.__len__(), shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_data.__len__(), shuffle=False, drop_last=True)
    
    model = ann(cfg)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])

    datasetname_modelname = cfg['dataset_name'] + '_' + cfg['model_name'] + '_' + cfg['dataset_prefix']

    ckpt_path = 'checkpoint/checkpoint_r2_' + datasetname_modelname
    checkpoint = torch.load(ckpt_path +'/' + save_params + '_best_vaf_model.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_rmse, val_cc, val_vaf = test(cfg,model,device,val_loader)
    test_rmse, test_cc, test_vaf = test(cfg,model,device,test_loader)
    save_params_value = [cfg[i] for i in cfg['save_params']]
    additional_params_value = [cfg[i] for i in cfg['additional_params'][cfg['dataset_name'].split('_')[1]]]

    a = save_params_value + additional_params_value + [val_rmse, val_cc, val_vaf]
    b = save_params_value + additional_params_value + [test_rmse, test_cc, test_vaf]

    with open('result/'+ datasetname_modelname +'_val.csv', 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        csv_writer = csv.writer(f)
        csv_writer.writerow(a)
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
    with open('result/'+ datasetname_modelname + '_test.csv', 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        csv_writer = csv.writer(f)
        csv_writer.writerow(b)
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


if __name__ == '__main__':
    # mp_flag=0
    mp_flag=1
    if mp_flag:
        pool = Pool(processes=10)
    dataset_names=['open','C']
    xhat_res=['xhat','res']
    model_names=['ann']
    eval_model_names = ['dvae']
    folds = ['1', '2', '3','4','5']
    csv_file_has_created_flag=\
            {xhat_or_res: {dataset_name:{model_name:0 for model_name in eval_model_names} for dataset_name in dataset_names} for xhat_or_res in xhat_res}
    print(csv_file_has_created_flag)
    alphas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    betas = [0.001]

    dataset_cfg = read_yaml('config/xhat_res.yaml')
    for xhat_or_res in xhat_res:
        dataset_cfg['dataset_prefix']=xhat_or_res
        for dataset_name in dataset_names:
            dataset_cfg['dataset_name']=dataset_name
            if dataset_name == 'open' :
                data_names = ['2017012401']
            elif dataset_name == 'C':
                data_names = ['20161007','20161011']
            model_cfg = read_yaml('config/ann.yaml')
            for eval_model_name in eval_model_names:
                dataset_cfg['dataset_dir']=dataset_name+'_'+eval_model_name
                dataset_cfg['dataset_name']=dataset_name+'_'+eval_model_name
                cfg = dataset_cfg.copy()
                cfg.update(model_cfg)
                if not csv_file_has_created_flag[xhat_or_res][dataset_name][dataset_cfg['dataset_name'].split('_')[1]]:
                    if not os.path.exists('result'):
                        os.makedirs('result')
                    # val csv
                    f = open('result/'+dataset_cfg['dataset_name']+'_ann_'+xhat_or_res+'_val.csv','w')
                    csv_writer = csv.writer(f)
                    additional_params_keys = cfg['additional_params'][cfg['dataset_name'].split('_')[1]].copy()

                    csv_writer.writerow(cfg['save_params']+additional_params_keys+cfg['save_criterias'])
                    f.close()
                    # test csv 
                    f = open('result/'+dataset_cfg['dataset_name']+'_ann_'+xhat_or_res+'_test.csv','w')
                    csv_writer = csv.writer(f)
                    additional_params_keys = cfg['additional_params'][cfg['dataset_name'].split('_')[1]].copy()
                    csv_writer.writerow(cfg['save_params']+additional_params_keys+cfg['save_criterias'])
                    f.close()
                    csv_file_has_created_flag[xhat_or_res][dataset_name][dataset_cfg['dataset_name'].split('_')[1]]=1
                cfg_list=[]
                for alpha in alphas:
                    for beta in betas:
                        for data_name in data_names:
                            for fold in folds:
                                tuner_params = {'eval_model_name':eval_model_name,'alpha':alpha, 'beta':beta,\
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
