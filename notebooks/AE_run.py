#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 14:43
# @Author  : Li Xiao
# @File    : AE_run.py
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import autoencoder_model
import torch
import torch.utils.data as Data
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def work(data, in_feas, names, weights, lr=0.001, bs=32, epochs=100, device=torch.device('cpu'), mode=0, topn=100, num_omics = 246):
    #name of sample
    sample_name = data['Sample'].tolist()
    print("work", num_omics)

    #change data to a Tensor
    X,Y = data.iloc[:,1:].values, np.zeros(data.shape[0])
    TX, TY = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
    
    #train a AE model
    if mode == 0 or mode == 1:
        print('Training model...')
        Tensor_data = Data.TensorDataset(TX, TY)
        train_loader = Data.DataLoader(Tensor_data, batch_size=bs, shuffle=True)

        #initialize a model
        mmae = autoencoder_model.MMAE(in_feas, latent_dim=100, weights=weights, num_omics=num_omics)
        mmae.to(device)
        mmae.train()
        mmae.train_MMAE(train_loader, learning_rate=lr, device=device, epochs=epochs)
        mmae.eval()       #before save and test, fix the variables
        torch.save(mmae, 'model/AE/MMAE_model.pkl')

    #load saved model, used for reducing dimensions
    if mode == 0 or mode == 2:
        print('Get the latent layer output...')
        mmae = torch.load('model/AE/MMAE_model.pkl')
        omics = {}
        start = 0
        end = 0
        for i in range(num_omics):
            start = end
            end = end + in_feas[i]
            omics[i] = TX[:, start:end]

   
        latent_data, decoded_omics = mmae.forward(omics)
        latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
        latent_df.insert(0, 'Sample', sample_name)
        #save the integrated data(dim=100)
        latent_df.to_csv('result/latent_data.csv', header=True, index=False)

    print('Extract features...')
    extract_features(data, in_feas, epochs, names, topn, num_omics)
    return

def extract_features(data, in_feas, epochs, names, topn=100, num_omics = 246):

    print("extract_features", num_omics)


    # extract features
    #get each omics data
    data_omics = {}
    start = 1
    end = 1
    for i in range(num_omics):
        start = end
        end = end + in_feas[i]
        data_omics[i] = data.iloc[:, start: end]


    #get all features of each omics data
    feas_omics = [data_omic.columns.tolist() for data_omic in list(data_omics.values())]
    
    #calculate the standard deviation of each feature
    std_omics = [data_omic.std(axis=0) for data_omic in list(data_omics.values())]

    #record top N features every 10 epochs
    topn_omics = [pd.DataFrame() for i in range(num_omics)]


    #used for feature extraction, epoch_ls = [10,20,...], if epochs % 10 != 0, add the last epoch
    epoch_ls = list(range(10, epochs+10,10))
    if epochs %10 != 0:
        epoch_ls.append(epochs)
    for epoch in tqdm(epoch_ls):
        #load model
        mmae = torch.load('model/AE/model_{}.pkl'.format(epoch))
        #get model variables
        model_dict = mmae.state_dict()


        #get the absolute value of weights, the shape of matrix is (n_features, latent_layer_dim)
        weight_omics = [np.abs(model_dict['encoder_omics.' + str(i) + '.0.weight'].detach().cpu().numpy().T) for i in range(num_omics)]
        weight_omics_df = [pd.DataFrame(weight_omics[i], index=feas_omics[i]) for i in range(num_omics)]
       

        #calculate the weight sum of each feature --> sum of each row
        for i in range (num_omics):
            weight_omics_df[i]['Weight_sum'] = weight_omics_df[i].apply(lambda x:x.sum(), axis=1)
            weight_omics_df[i]['Std'] = std_omics[i]
            #importance = Weight * Std
            weight_omics_df[i]['Importance'] = weight_omics_df[i]['Weight_sum']*weight_omics_df[i]['Std']
            #select top N features
            
        fea_omics_top = [weight_omics_df[i].nlargest(topn, 'Importance').index.tolist() for i in range(num_omics)]
    

        #save top N features in a dataframe
        col_name = 'epoch_'+str(epoch)
        for i in range(num_omics):
            topn_omics[i][col_name] = fea_omics_top[i] 
            #all of top N features
            topn_omics[i].to_csv('result/topn_omics_'+ names[i] +'.csv', header=True, index=False)
       

    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=int, choices=[0,1,2], default=0,
                        help='Mode 0: train&intagrate, Mode 1: just train, Mode 2: just intagrate, default: 0.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--path', '-p', type=str, required=True, help='The omics folder name.')
    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='Training batchszie, default: 32.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Training epochs, default: 100.')
    parser.add_argument('--latent', '-l', type=int, default=100, help='The latent layer dim, default: 100.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu', 'mps'], default='cpu', help='Training on cpu, mps or gpu, default: cpu.')
    parser.add_argument('--topn', '-n', type=int, default=100, help='Extract top N features every 10 epochs, default: 100.')
    parser.add_argument('--level', '-lev', type=str, choices=['1', '2', '3', '4', '5'], default='1', help='Level of aggregation. Level 5 is negative control.')
    args = parser.parse_args()


    #read data
    omics_data = []
    names = {}
    i = 0
    for filename in os.listdir(args.path):
        if not filename == ".DS_Store":
            data_path = os.path.join(args.path, filename)
            omics_data.append(pd.read_csv(data_path, header=0, index_col=None))
            names[i] = filename
            i+=1
      


    #Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    #set random seed
    setup_seed(args.seed)


    #dims of each omics data
    in_feas = [data.shape[1] - 1 for data in omics_data]
   
    omics_data = [data.rename(columns={data.columns.tolist()[0]: 'Sample'}) for data in omics_data]

    omics_data = [data.sort_values(by='Sample', ascending=True) for data in omics_data]

    #merge the multi-omics data, calculate on common samples
    Merge_data = None
    for i, data in enumerate(omics_data):
        if Merge_data is None:
            Merge_data = data
        else:
            Merge_data = pd.merge(Merge_data, data, on='Sample', how='inner', suffixes=('', names[i] ))
            

    print(Merge_data.shape)
    
    level = args.level
    weights = []
    if level == "1": # Level 1
        for i in range(len(names)):
            name = names[i]
            modality =  name.split("_")[0]

            if modality == "citeRNA":
                weights.append(1/209 * 1/6)
            elif modality == "bulkRNA":
                weights.append(1/12 * 1/6)
            elif modality == "adt":
                weights.append(1/11 * 1/6)
            elif modality == "facs":
                weights.append(1 * 1/6)
            elif modality == "luminex":
                weights.append(1 * 1/6)
            elif modality == "cytof":
                weights.append(1/12 * 1/6)
    elif level == "2": # Level 2
        for i in range(len(names)):
            name = names[i]
            modality =  name.split("_")[0]

            if modality == "citeRNA":
                weights.append(1/11 * 1/6)
            elif modality == "bulkRNA":
                weights.append(1 * 1/6)
            elif modality == "adt":
                weights.append(1/11 * 1/6)
            elif modality == "facs":
                weights.append(1 * 1/6)
            elif modality == "luminex":
                weights.append(1 * 1/6)
            elif modality == "cytof":
                weights.append(1/12 * 1/6)
    elif level == "5": # Level 5
        for i in range(len(names)):
            name = names[i]
            modality =  name.split("_")[0]

            if modality == "citeRNA":
                weights.append(1/(209+31) * 1/6)
            elif modality == "bulkRNA":
                weights.append(1/(12+1) * 1/6)
            elif modality == "adt":
                weights.append(1/(11+1) * 1/6)
            elif modality == "facs":
                weights.append(1 * 1/6)
            elif modality == "luminex":
                weights.append(1 * 1/6)
            elif modality == "cytof":
                weights.append(1/(12+1) * 1/6)
    else:
        for i in range(len(names)):
            weights.append(1/6)
    
    
    print(len(names), weights)
   
    #train model, reduce dimensions and extract features
    work(Merge_data, in_feas, names, weights, lr=args.learningrate, bs=args.batchsize, epochs=args.epoch, device=device, mode=args.mode, topn=args.topn, num_omics = len(weights))
    print('Success! Results can be seen in result file')
    print(names)
