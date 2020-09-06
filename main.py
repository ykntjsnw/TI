#!/usr/bin/env python
# coding: utf-8

# In[40]:
'''
add RNN and LSTM
combine GRU,RNN and LSTM together
'''


#方案一：只使用GRU提取特征向量,因为使用SAGAN中的self-attention方法来考虑各个采样点之间的联系可能会丧失时序性。固定采样点的位置。
#方案二：使用transformer来处理序列
import importlib,sys
importlib.reload(sys)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np
import datasets
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

import sklearn
from sklearn.metrics import roc_auc_score



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", type=int, default=49)
    parser.add_argument("--list_len", type=int, default=100)
    parser.add_argument("--key_dim", type=int, default=16)
    parser.add_argument("--value_dim", type=int, default=16)
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--category", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--h_list_path", type=str, default="h_list_100_5000_49_2.npy")
    parser.add_argument("--labels_path", type=str, default="labels_100_5000_49_2.npy")
    parser.add_argument("--model", type=str, default="GRU")
    parser.add_argument("--noise", type=bool, default=False)
    parser.add_argument("--smoke_test", type=bool, default=False)
    args = parser.parse_args()
    return args


class classification_with_RNNs(nn.Module):
    def __init__(self, args, config):
        super(classification_with_RNNs, self).__init__()
        self.sensors = args.sensors
        self.hidden_size = config["hidden_size"]
        if args.model == "RNN":
            self.model = nn.RNN(args.sensors, config["hidden_size"], num_layers=1, batch_first=True)
        elif args.model == "GRU":
            self.model = nn.GRU(args.sensors, config["hidden_size"], num_layers=1, batch_first=True)
        elif args.model == "LSTM":
            self.model = nn.LSTM(args.sensors, config["hidden_size"], num_layers=1, batch_first=True)
        else:
            raise ValueError
        self.main = nn.Sequential(
            #input_size (n x hidden_size)
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["hidden_size"], 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=config["dropout"]),

            nn.Linear(256, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(64, args.category),
            #output_size (n x 1)
        )
        
    def forward(self, h_list):
        # h_list.size():(N, L, sensors)
        hidden_state, _ = self.model(h_list)
        out = self.main(hidden_state[:, -1, :])
        return out #out.size(): (N, 1) 未经过softmax

    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



def train(config):
    global args
    setup_seed(101)
    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = classification_with_RNNs(args, config)

    # Handle multi-gpu if desired
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999),
                           weight_decay=config["lambda"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # create the dataloader
    dataset = datasets.data_Classification(args.h_list_path, args.labels_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    batch_size = int(config["batch_size"])
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    dev_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    for epoch in range(args.num_epochs):
        ###training loop
        scheduler.step()
        model.train()
        for i, (h_list, y) in enumerate(train_dataloader):
            h_list, y = h_list.to(device), y.to(device)
            if args.noise == True:
                noise = torch.randn_like(h_list) * 3
                noise = noise.to(device)
                h_list = h_list + noise
            model.zero_grad()
            entropy = nn.CrossEntropyLoss()
            out = model(h_list)
            loss = entropy(out, y.long())
            loss.backward()
            optimizer.step()

        # computing loss, accuracy and auc on training set
        entropy_loss = []
        y_true = []
        y_scores = []
        train_accuracy = 0
        model.eval()
        with torch.no_grad():
            for i, (h_list, y) in enumerate(train_dataloader):
                h_list, y = h_list.to(device), y.to(device)
                if args.noise == True:
                    noise = torch.randn_like(h_list) * 3
                    noise = noise.to(device)
                    h_list = h_list + noise
                entropy = nn.CrossEntropyLoss()
                out = model(h_list)
                loss = entropy(out, y.long())
                y_true += list(y.cpu().numpy())
                soft = F.softmax(out.detach(), dim=1)
                y_scores += list(soft[:, 1].cpu().numpy())
                predict_list = np.argmax(out.cpu().detach().numpy(), axis=1)
                for j, pre in enumerate(predict_list):
                    if pre == y[j].cpu().item():
                        train_accuracy += 1
                entropy_loss.append(loss.cpu().item())
            train_auc = roc_auc_score(np.array(y_true), np.array(y_scores))
            train_accuracy /= (config["batch_size"] * (i + 1))
            train_loss = np.array(entropy_loss).mean()

        ###computing loss, accuracy and auc on testing set
        entropy_loss = []
        y_true = []
        y_scores = []
        dev_accuracy = 0
        model.eval()
        with torch.no_grad():
            for i, (h_list, y) in enumerate(dev_dataloader):
                h_list, y = h_list.to(device), y.to(device)
                if args.noise == True:
                    noise = torch.rand_like(h_list) * 3
                    noise = noise.to(device)
                    h_list = h_list + noise
                entropy = nn.CrossEntropyLoss()
                out = model(h_list)
                loss = entropy(out, y.long())
                y_true += list(y.cpu().numpy())
                soft = F.softmax(out.detach(), dim=1)
                y_scores += list(soft[:, 1].cpu().numpy())
                predict_list = np.argmax(out.cpu().detach().numpy(), axis=1)
                for j, pre in enumerate(predict_list):
                    if pre == y[j].cpu().item():
                        dev_accuracy += 1
                entropy_loss.append(loss.cpu().item())
            dev_auc = roc_auc_score(np.array(y_true), np.array(y_scores))
            dev_accuracy /= (config["batch_size"] * (i + 1))

            dev_loss = np.array(entropy_loss).mean()
        tune.track.log(train_loss=train_loss, train_auc=train_auc, train_accuracy=train_accuracy,
                       dev_loss=dev_loss, dev_auc=dev_auc, dev_accuracy=dev_accuracy)

    del optimizer


if __name__ == "__main__":
    args = get_arguments()

    config = {"gamma": tune.choice([0.5, 0.7, 1]),
              "step_size": tune.choice([20, 30, 40]),
              "lr": tune.loguniform(0.0001, 0.000001),
              "beta1": 0.9,
              "lambda": tune.loguniform(0.0001, 0.1),
              "batch_size": tune.choice([64, 128]),
              "dropout": tune.uniform(0, 0.5),
              "hidden_size": 128}
   
    ray.shutdown()
    ray.init()
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="dev_accuracy",
        mode="max",
        max_t=500,
        grace_period=500
    )
    analysis = tune.run(
        train,
        config=config,
        scheduler=sched,
        num_samples=1,
        resources_per_trial={"gpu": 2},
        local_dir="./results",
        name="4000_10000",
        stop={"dev_accuracy": 0.99,
              "training_iteration": 5 if args.smoke_test else 1000}
    )


