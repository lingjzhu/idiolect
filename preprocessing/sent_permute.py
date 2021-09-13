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

tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')

nlp = English()
nlp.add_pipe("sentencizer")

def permute_sent(sents):
    out = tokenizer.encode_plus(sents,add_special_tokens=True, max_length=102,truncation=True,return_tensors="np")['input_ids']
    segmented = tokenizer.decode(out[0],skip_special_tokens=True)
    doc = nlp(segmented)
    sents = shuffle([i.text for i in doc.sents])
    return ' '.join(sents)
    
    
path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/authorship/'
files = ['reddit_train_contrastive','reddit_dev_contrastive','reddit_test_contrastive']
#files = ['amazon_train_contrastive','amazon_dev_contrastive','amazon_test_contrastive']

for f in files:
    with open(os.path.join(path,f+'_spermute'),'w') as out:
        with open(os.path.join(path,f),'r') as file:
            for line in tqdm(file.readlines()):
                line = line.strip().split('\t\t')
                a = permute_sent(line[2])
                b = permute_sent(line[3])
                line[2] = a
                line[3] = b
                out.write('\t\t'.join(line)+'\n')