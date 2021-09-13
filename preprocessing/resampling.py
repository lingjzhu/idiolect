#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
from tqdm import tqdm
import os


path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/resample'


files = ['amazon_moderate_conicity','amazon_random_conicity','amazon_upper_conicity','amazon_lower_conicity']
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
                out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(instance['user']+'--'+pair['user'].item(),instance['product']+'--'+pair['product'].item(),instance['text'],pair['text'].item(),1,instance['domain']+'--'+pair['domain'].item()))
            
             
            negatives = data[data['user']!=author]
            for k in range(6):
                instance = samples.sample(1)
                negative = negatives.sample(1)
                out.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n"%(instance['user'].item()+"--"+negative['user'].item(),instance['product'].item()+"--"+negative['product'].item(),instance['text'].item(),negative['text'].item(),0,instance['domain'].item()+"--"+negative['domain'].item()))