#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from tqdm import tqdm




samples = pd.DataFrame(columns=['user','id','text','length','domain'])


with open('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/reddit','r') as file:

    for line in tqdm(file.readlines()):
        line = line.strip()
#        line = re.sub(r'\\','',line).split('\t\t')
        line = line.split('\t\t')
        text = re.sub(r'\(http.*?\)','',line[4])
        sample = {'user':[line[0]],'length':[line[1]],'domain':[line[2]],'id':[line[3]],'text':[text]}
        sample = pd.DataFrame.from_dict(sample)
        samples = pd.concat([samples,sample])

samples.to_csv('/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/reddit_samples',sep='\t',index=None)



with open('Authorship/reddit','w') as out:
    with open('Authorship/reddit_samples','r') as file:
    
        for line in tqdm(file.readlines()):
            line = line.strip()
            split = re.sub(r'\\','',line).split('\t\t')
            if len(split) == 5:
                out.write('\n'+line)
            else:
                out.write(' <paragraph>'+line)


path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship'
files = ['reddit_train','reddit_test','reddit_dev']
for file in files:
    with open(os.path.join(path,file+"_contrastive"),'w') as out:
        
        data = pd.read_csv(os.path.join(path,file),sep='\t')
        authors = data['user'].unique().tolist()
        for author in tqdm(authors):
            samples = data[data['user']==author]
            instances = samples.sample(6)
            remain = samples.drop(instances.index)
            for i,instance in instances.iterrows():
                pair = remain.sample(1)
                out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(instance['user']+'--'+pair['user'].item(),instance['id']+'--'+pair['id'].item(),instance['text'],pair['text'].item(),1,instance['domain']+'--'+pair['domain'].item()))
            
             
            negatives = data[data['user']!=author]
            for k in range(6):
                instance = samples.sample(1)
                negative = negatives.sample(1)
                out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(instance['user'].item()+"--"+negative['user'].item(),instance['id'].item()+"--"+negative['id'].item(),instance['text'].item(),negative['text'].item(),0,instance['domain'].item()+"--"+negative['domain'].item()))
                
