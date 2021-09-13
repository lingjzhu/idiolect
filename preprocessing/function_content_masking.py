#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import transformers
import spacy
from sklearn.utils import shuffle
from spacy.lang.en import English
from tqdm import tqdm

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

processor = spacy.load("en_core_web_sm",disable=["tagger", "parser", "lemmatizer",'ner'])


def masking(text,function=True):
    out = processor(text)
    if not function:
        masked = ' '.join([tokenizer.mask_token if w.is_stop else w.text for w in out])
    else:
        masked = ' '.join([tokenizer.mask_token if not w.is_stop and not w.is_punct else w.text for w in out])
    return masked
	
	
	
path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/'
#files = ['tr-samples_contrastive','de-samples_contrastive','te-samples_contrastive']
files = ['reddit_train_contrastive','reddit_dev_contrastive','reddit_test_contrastive']
for f in files:
    with open(os.path.join(path,f+'_function_bert'),'w') as out:
        with open(os.path.join(path,f),'r') as file:
            for line in tqdm(file.readlines()):
                line = line.strip().split('\t\t')
                a = masking(line[2],function=True)
                b = masking(line[3],function=True)
                line[2] = a
                line[3] = b
                out.write('\t\t'.join(line)+'\n')
				
				
path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/'
#files = ['tr-samples_contrastive','de-samples_contrastive','te-samples_contrastive']
files = ['reddit_train_contrastive','reddit_dev_contrastive','reddit_test_contrastive']
for f in files:
    with open(os.path.join(path,f+'_content_bert'),'w') as out:
        with open(os.path.join(path,f),'r') as file:
            for line in tqdm(file.readlines()):
                line = line.strip().split('\t\t')
                a = masking(line[2],function=False)
                b = masking(line[3],function=False)
                line[2] = a
                line[3] = b
                out.write('\t\t'.join(line)+'\n')