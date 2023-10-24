#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 14:01
# @Author  : Li Xiao
# @File    : autoencoder_model.py
import torch
from torch import nn
from matplotlib import pyplot as plt

class MMAE(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, weights, num_omics = 246):
        '''
        :param in_feas_dim: a list, input dims of omics data
        :param latent_dim: dim of latent layer
        :param weights: weight of omics data 
        '''
        super(MMAE, self).__init__()
        self.weights = weights
        self.in_feas = in_feas_dim
        self.latent = latent_dim
        self.num_omics = num_omics
        print("Ae_model, MMAE, __init__", self.num_omics)

        #encoders, multi channel input
        self.encoder_omics = nn.ModuleList([nn.Sequential(
            nn.Linear(self.in_feas[i], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        ) for i in range(self.num_omics)])
        
        #decoders
        self.decoder_omics = nn.ModuleList([nn.Sequential(nn.Linear(self.latent, self.in_feas[i])) for i in range(self.num_omics)])


        #Variable initialization
        for name, param in MMAE.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)
     

    def forward(self, omics):
        '''
        :param omics: omics data
        '''
        encoded_omics = [self.encoder_omics[i](omics[i]) for i in range(self.num_omics)]

        latent_data = None
        for i in range(self.num_omics):
            if latent_data is None:
                latent_data  = torch.mul(encoded_omics[i], self.weights[i])
            else:
                latent_data = latent_data + torch.mul(encoded_omics[i], self.weights[i])

        decoded_omics = [self.decoder_omics[i](latent_data) for i in range(self.num_omics)]
     
        return latent_data, decoded_omics

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        loss_ls = []
        for epoch in range(epochs):
            train_loss_sum = 0.0       #Record the loss of each epoch
            for (x,y) in train_loader:
                omics = {}
                start = 0
                end = 0
                for i in range(self.num_omics):
                    start = end
                    end = end + self.in_feas[i]
                    omics[i] = x[:, start:end]
                    omics[i].to(device)
                    


                latent_data, decoded_omics = self.forward(list(omics.values()))
                loss = 0
                for i in range(self.num_omics):
                    loss = loss + self.weights[i] * loss_fn(decoded_omics[i], omics[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.sum().item()

            loss_ls.append(train_loss_sum)
            print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))

            #save the model every 10 epochs, used for feature extraction
            if (epoch+1) % 10 ==0:
                torch.save(self, 'model/AE/model_{}.pkl'.format(epoch+1))

        #draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('result/AE_train_loss.png')