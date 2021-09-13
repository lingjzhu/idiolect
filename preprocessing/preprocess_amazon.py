#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:11:59 2020

"""

import json
import os
import re
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.utils import shuffle


def clean(text):
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r'&nbsp;','',text)
    return text



path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/Amazon'

data = pd.read_csv(os.path.join(path,'5-core','5-samples'),sep='\t',header=None)
data = data.rename(columns={0:'user',1:'product',2:'text',3:'price',4:'date',5:'sentiment',
                    6:'length'})
print(data.head())

meta = pd.read_csv('./all_products',sep='\t\t',header=None)
mapping = {k:v for k,v in zip(meta[0],meta[1])}
domains = [mapping[i] for i in data['product']]
data['domain'] = domains

cleaned_texts = [clean(i) for i in tqdm(data['text'])]
data['text'] = cleaned_texts



users = []
with open(os.path.join(path,'5-core','users'),'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t\t')
        if int(line[1])>1:
            users.append(line[0])
print("users loaded!")      
data = data[data['user'].isin(users)]


print(data.head())
authors = data['user'].unique().tolist()
authors = shuffle(authors)
print(len(authors))
size = len(authors)
train_size = int(size*0.40)
de_size = int(size*0.1)
te_size = int(size*0.50)
authors_tr = authors[:train_size]
authors_de = authors[train_size:train_size+de_size]
authors_te = authors[train_size+de_size:]


data_tr = data[data['user'].isin(authors_tr)]
data_de = data[data['user'].isin(authors_de)]
data_te = data[data['user'].isin(authors_te)]
del data

data_tr.to_csv(os.path.join(path,'5-core/balanced_partition','tr-samples'),index=None,sep='\t')
data_de.to_csv(os.path.join(path,'5-core/balanced_partition','de-samples'),index=None,sep='\t')
data_te.to_csv(os.path.join(path,'5-core/balanced_partition','te-samples'),index=None,sep='\t')


with open(os.path.join(path,'5-core',"contrastive_tr"),'w') as out:
    
    data_tr = pd.read_csv(os.path.join(path,'5-core','tr-samples'),sep='\t')
    authors = data_tr['0'].unique().tolist()
    for author in authors:
        samples = data_tr[data_tr['0']==author]
        reviews = [row for row in samples.iterrows()]
        reviews = shuffle(list(combinations(reviews,2)))
        for i,j in reviews[:5]:
            out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(author+"--"+author,i[1][1]+"--"+j[1][1],clean(i[1][2]),clean(j[1][2]),1))
        negatives = data_tr[data_tr['0']!=author].sample(5)
        for i,j in zip(samples.iterrows(),negatives.iterrows()):
            out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(i[1][0]+"--"+j[1][0],i[1][1]+"--"+j[1][1],clean(i[1][2]),clean(j[1][2]),0))


with open(os.path.join(path,'5-core',"contrastive_te"),'w') as out:
    
    data_tr = pd.read_csv(os.path.join(path,'5-core','te-samples'),sep='\t')
    authors = data_tr['0'].unique().tolist()
    for author in authors:
        samples = data_tr[data_tr['0']==author]
        reviews = [row for row in samples.iterrows()]
        reviews = shuffle(list(combinations(reviews,2)))
        for i,j in reviews[:5]:
            out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(author+"--"+author,i[1][1]+"--"+j[1][1],clean(i[1][2]),clean(j[1][2]),1))
        negatives = data_tr[data_tr['0']!=author].sample(5)
        for i,j in zip(samples.iterrows(),negatives.iterrows()):
            out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(i[1][0]+"--"+j[1][0],i[1][1]+"--"+j[1][1],clean(i[1][2]),clean(j[1][2]),0))



with open(os.path.join(path,'5-core',"contrastive_de"),'w') as out:
    
    data_tr = pd.read_csv(os.path.join(path,'5-core','de-samples'),sep='\t')
    authors = data_tr['0'].unique().tolist()
    for author in authors:
        samples = data_tr[data_tr['0']==author]
        reviews = [row for row in samples.iterrows()]
        reviews = shuffle(list(combinations(reviews,2)))
        for i,j in reviews[:5]:
            out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(author+"--"+author,i[1][1]+"--"+j[1][1],clean(i[1][2]),clean(j[1][2]),1))
        negatives = data_tr[data_tr['0']!=author].sample(5)
        for i,j in zip(samples.iterrows(),negatives.iterrows()):
            out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(i[1][0]+"--"+j[1][0],i[1][1]+"--"+j[1][1],clean(i[1][2]),clean(j[1][2]),0))

