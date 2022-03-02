#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertAdam

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from tqdm import tqdm


# In[2]:


data = pd.read_csv('CLEAN.csv',index_col=0)
# print(data['misarticulation_index'].values[:15])
data = data.loc[data['first_lang_english']==1]
data.drop(columns=['first_lang_english'],inplace=True)
data = data[['response_text','misarticulation_index']]
# 每三分之一一个档
data['misarticulation_index'] = (data['misarticulation_index']/0.33333).astype(int).astype(float)/18
# print(data['misarticulation_index'].values[:15])


# In[3]:


split_ = np.random.RandomState(seed=0).permutation(data.shape[0])
num_train = int(data.shape[0]*0.6)

data_train, data_test =data[:num_train], data[num_train:]


# In[4]:


class TMPDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.X = df['response_text'].values
        self.y = df['misarticulation_index'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 8
    
train_dataset = TMPDataset(data_train)
test_dataset = TMPDataset(data_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


# In[6]:


tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = ".") # bert-base-uncased
model = BertModel.from_pretrained(pretrained_model_name_or_path = ".") # bert-base-uncased
# tokenizer.save_pretrained('./')
# model.save_pretrained('./')


# In[6]:


regressor = nn.Sequential(nn.Linear(768,64),nn.ELU(),nn.Linear(64,1))


# In[7]:


optimizer = BertAdam(list(model.parameters())+list(regressor.parameters()), lr=5e-5,
                     weight_decay=1e-2, warmup=0.2, t_total=10*(num_train//batch_size))


# In[8]:


loss_fn = torch.nn.L1Loss()


# In[9]:


for i in range(10):    
    for X,y in tqdm(train_dataloader):
#         y = y.cuda()
        model.train()
        regressor.train()
        optimizer.zero_grad()
        
        encoded_input = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt",max_length=512)
#         encoded_input = {k:v.cuda() for k,v in encoded_input.items()}
        output = model(**encoded_input)
        # print(output.pooler_output)
        logits = regressor(output.pooler_output)
        loss = loss_fn(logits.flatten(), y)
        # print(loss)
        loss.backward()
        optimizer.step()

        # break
    print(f"---------------------epoch {i}----------------------")
    with torch.no_grad():
        model.eval()
        regressor.eval()
        y_pred_list = []
        y_true_list = []
        for X,y in test_dataloader:
            encoded_input = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt",max_length=512)
            output = model(**encoded_input)
            y_pred = regressor(output.pooler_output).flatten().cpu().numpy()
            y_pred_list.append(y_pred)
            y_true_list.append(y)
        y_pred = np.hstack(y_pred_list)
        y_true = np.hstack(y_true_list)
        ms = [mean_absolute_error, mean_squared_error, median_absolute_error]
        for m in ms:
            print(m, m(y_true*6, y_pred*6))

