#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:42:26 2020

@author: lukeum
"""

import pandas as pd
import torch
import transformers
import numpy as np
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, AdamW,RobertaTokenizerFast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.autograd import grad

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from pingouin import mwu
import seaborn as sns
#data = pd.read_csv('Data/food',sep='\t\t',header=None)
#price_range = pd.read_csv('Data/price_range',sep='\t\t',header=None)


device = "cuda"
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

'''
products = price_range[0].unique().tolist()
data = data[data[1].isin(products)]
price_mapping = {k:int(v) for k,v in zip(price_range[0],price_range[2])}
data['y'] = [price_mapping[i] for i in data[1]]
data = data[~data['y'].isin([3,4,5,6])]
labels = [1 if i>=6 else 0 for i in data['y']]
data['y'] = labels

x_train, x_test, y_train, y_test = train_test_split(data[3].tolist(),data['y'].tolist())
'''


high = pd.read_csv('./Data/high_users',sep='\t')
low = pd.read_csv('./Data/low_users',sep='\t')

htexts = [' '.join(i.split(' ')[:100]) for i in high['text']]
ltexts = [' '.join(i.split(' ')[:100]) for i in low['text']]


'''
Test for readability
'''
lreadability = []
for i in tqdm(ltexts):
    lreadability.append(textstat.gunning_fog(i))
    
hreadability = []
for i in tqdm(htexts):
    hreadability.append(textstat.gunning_fog(i))

sns.distplot(lreadability)
sns.distplot(hreadability)



X = htexts + ltexts
y = [1 for i in range(len(high))] + [0 for i in range(len(low))]

X_train, X_test, y_train, y_test = train_test_split(X,y)


def tokenize(textA,seq_len=102):
    
    length = len(textA)
    input_ids_a = torch.ones(length,seq_len).long()
    attention_mask_a = torch.zeros(length,seq_len).long()
    
    for k, a in tqdm(zip(range(length),textA)):
        seqA = tokenizer.encode_plus(a,add_special_tokens=True, max_length=seq_len,truncation=True,return_tensors="pt")
        input_ids_a[k,:seqA['input_ids'].shape[1]] = seqA['input_ids']
        attention_mask_a[k,:seqA['attention_mask'].shape[1]] = seqA['attention_mask']
    return input_ids_a,attention_mask_a


X_train_feat, X_train_mask = tokenize(X_train)
X_test_feat, X_test_mask = tokenize(X_test)



def batchify(text_a,attn_a,labels,size=16,shuffling=False):
    length = len(text_a)
    
    if shuffling == True:
        text_a,text_b,attn_a,attn_b,labels = shuffle(text_a,attn_a,labels)
        
    for i in range(0,length,size):
        sampleA = text_a[i:min(i+size,length)]
        sample_attn_a = attn_a[i:min(i+size,length)]
        y = labels[i:min(i+size,length)]

        yield sampleA, sample_attn_a,torch.tensor(y).long()



def masking(text,mlm_prob=0.1,mask=50264):
    
    indices_replaced = torch.bernoulli(torch.full(text.shape, mlm_prob)).bool()
    text[indices_replaced] = mask
    return text
    



#model = RobertaModel.from_pretrained('roberta-base')
#model = torch.load('model_food_cosine')
#model.to(device)

loss_fn = nn.BCEWithLogitsLoss()



class SESclassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.linear = nn.Sequential(
                        nn.Linear(768,100),
                        nn.ReLU(),
                        nn.Linear(100,100),
                        nn.ReLU(),
                        nn.Linear(100,1))        
    def forward(self,texts, masks):
        hidden = self.roberta(texts,masks)
        hidden = hidden[0][:,0,:].squeeze()
        logits = self.linear(hidden).squeeze()
        return logits


class pretrained_SESclassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.roberta = torch.load('./model/model-cosine-0.2-0.8-mask-0.1/model-2')
        self.linear = nn.Sequential(
                        nn.Linear(768,100),
                        nn.ReLU(),
                        nn.Linear(100,100),
                        nn.ReLU(),
                        nn.Linear(100,1))
        
        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self,texts, masks):
        hidden = self.roberta(texts,masks)
        hidden = hidden[0][:,0,:].squeeze()
        logits = self.linear(hidden).squeeze()
        return logits





model = SESclassifier().to(device)
model = pretrained_SESclassifier().to(device)
optimizer = AdamW(model.parameters(),lr=1e-4)


epochs = 2
for e in range(epochs):
    for i, (texts,masks,y) in tqdm(enumerate(batchify(X_train_feat,X_train_mask,y_train,size=32))):
#        texts = masking(texts)
        logits = model(texts.to(device),masks.to(device))
        loss = loss_fn(logits,y.to(device).float())
        loss.backward()
        if i%8 == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("Loss: %s"%loss.item())





preds = []
targets = []
model.eval()
for i, (texts,masks,y) in tqdm(enumerate(batchify(X_test_feat, X_test_mask,y_test,size=64))):
    with torch.no_grad():
        logits = model(texts.to(device),masks.to(device))
        logits = torch.sigmoid(logits)
        preds.extend(list(logits.cpu().detach().numpy()))
        targets.extend(list(y.cpu().detach().numpy()))




scores = [1 if i>=0.5 else 0 for i in preds]
accuracy = (np.array(targets)==np.array(scores))
accuracy = sum(accuracy)/len(accuracy)
print(accuracy)        
print(f1_score(targets,scores))    



#torch.save(model,'ses_classifier')

'''
Ling feat LR
'''

from writeprints import get_writeprints_transformer, prepare_entry
import pickle


with open('./temp_data/pan20_large_computed/transformers.p', 'rb') as f:
    transformer, scaler = pickle.load(f)



features = transformer.get_feature_names()


X_train_feat = []
for i in tqdm(X_train):
    parsed = prepare_entry(i)
    out = scaler.transform(transformer.transform([parsed])).todense()
    X_train_feat.append(out)
    
X_test_feat = []
for i in tqdm(X_test):
    parsed = prepare_entry(i)
    out = scaler.transform(transformer.transform([parsed])).todense()
    X_test_feat.append(out)    
    
X_train_feat = np.concatenate(X_train_feat,axis=0)
X_test_feat = np.concatenate(X_test_feat,axis=0)
lr = LogisticRegression()
lr.fit(X_train_feat,y_train)
lr.score(X_test_feat,y_test)


preds = lr.predict(X_test_feat)
preds = [1 if i>=0.5 else 0 for i in preds]
f1_score(y_test,preds)




with open('ling_emb_cca.pkl','wb') as out:
    pickle.dump((X_train_feat,y_train,X_test_feat,y_test),out)




'''
Style embeddings
'''

model = torch.load('../model/model-cosine-0.2-0.8-mask-0.1/model-2')

embeddings = []
model.eval()
for i, (texts,masks,y) in tqdm(enumerate(batchify(X_train_feat, X_train_mask,y_train,size=64))):
    with torch.no_grad():
            hidden = model(texts.to(device),masks.to(device))
            hidden = hidden[0][:,0,:].squeeze()
            hidden = F.normalize(hidden,dim=-1)
            hidden = hidden.cpu().detach().numpy()
            embeddings.append(hidden)

embeddings = np.concatenate(embeddings,axis=0)


tembeddings = []
model.eval()
for i, (texts,masks,y) in tqdm(enumerate(batchify(X_test_feat, X_test_mask,y_test,size=64))):
    with torch.no_grad():
            hidden = model(texts.to(device),masks.to(device))
            hidden = hidden[0][:,0,:].squeeze()
            hidden = F.normalize(hidden,dim=-1)
            hidden = hidden.cpu().detach().numpy()
            tembeddings.append(hidden)

tembeddings = np.concatenate(tembeddings,axis=0)

with open('sty_emb_cca','wb') as out:
    pickle.dump((embeddings,tembeddings),out)

with open('partition','wb') as out:
    pickle.dump((X_train,X_test,y_train,y_test),out)

'''
CCA
'''
with open('ling_emb_cca.pkl','rb') as out:
    X_train_feat, _, X_test_feat, _ = pickle.load(out)
    
with open('sty_emb_cca','rb') as out:
    embeddings,tembeddings = pickle.load(out)
    
    
with open('partition','rb') as out:
    X_train,X_test,y_train,y_test = pickle.load(out)
    
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr

cca = CCA(n_components=2)
cca.fit(X_train_feat, embeddings)

X_c, Y_c = cca.transform(X_test_feat,tembeddings)

plt.scatter(X_c[:,0],Y_c[:,1])
spearmanr(X_c[:,0],Y_c[:,1])

'''
BOW LR
'''


stopwords_ = list(stopwords.words('english'))
#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)
y = list(y_train)

X_te = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=2000).fit(X,y)
lr.score(X_te,y_test)

coeff = lr.coef_
features = vectorizer.get_feature_names()

#features = columns
weights =  pd.concat([pd.DataFrame(features,columns=['words']),pd.DataFrame(np.transpose(coeff),columns=['coef'])], axis = 1)
weights = weights.sort_values('coef')


preds = lr.predict(X_te)
preds = [1 if i>=0.5 else 0 for i in preds]
f1_score(y_test,preds)









'''
Gradients
'''

def _get_samples_input(input_tensor, baseline, 
                       num_samples):

    input_dims = list(input_tensor.size())[1:]
    num_input_dims = len(input_dims)
    batch_size = input_tensor.size()[0]
    

    batch_size = input_tensor.size()[0]
    input_expand = input_tensor.unsqueeze(1)
    reps = np.ones(len(baseline.shape)).astype(int)
    reps[0] = batch_size
    reference_tensor = baseline.repeat(list(reps)).unsqueeze(1)
#             reference_tensor = torch.as_tensor(sampled_baseline).unsqueeze(1).to(baseline.device)
    scaled_inputs = [reference_tensor + (float(i)/num_samples)*(input_expand - reference_tensor) \
                     for i in range(0,num_samples+1)]
    samples_input = torch.cat(scaled_inputs,dim=1)
    
    samples_delta = _get_samples_delta(input_tensor, reference_tensor)
    samples_delta = samples_delta
    
    return samples_input, samples_delta

def _get_samples_delta(input_tensor, reference_tensor):
    input_expand_mult = input_tensor.unsqueeze(1)
    sd = input_expand_mult - reference_tensor
    return sd


def _get_grads(samples_input, create_graph=False):

    grad_tensor = torch.zeros(samples_input.shape).float()

        
    k_ = samples_input.shape[1]


    for i in range(k_):
        particular_slice = samples_input[:,i]

        hidden = model.roberta.encoder(particular_slice)
        hidden = hidden[0][:,0,:]
        logit = model.linear(hidden)

        model_grads = grad(
                outputs=logit,
                inputs=particular_slice,
                grad_outputs=torch.ones_like(logit),
                create_graph=create_graph)
        
        grad_tensor[:,i,:,:] = model_grads[0].cpu()
    return grad_tensor


#model = torch.load('ses_classifier')


for param in model.roberta.parameters():
    param.requires_grad = True

texts,masks,y = next(iter(batchify(X_test_feat, X_test_mask,y_test,size=64)))

n = 42
len_a = torch.sum(masks[n])
sample = texts[n][:len_a].unsqueeze(0)

emb = model.roberta.embeddings(input_ids=sample.to(device))
hidden = model.roberta.encoder(emb.to(device))
hidden = hidden[0][:,0,:]
logit = model.linear(hidden)

print("Probability: %s - Target: %s"%(torch.sigmoid(logit),y[n]))
baseline = model.roberta.embeddings(torch.tensor([[50264]]).to(device))



samples_input, samples_delta = _get_samples_input(emb, baseline, num_samples=100)
grad_tensor = _get_grads(samples_input)

ig = grad_tensor*samples_delta.cpu()
ig = torch.sum(ig,dim=-1)
ig = torch.mean(ig,dim=1)
ig = ig.detach().cpu().squeeze().numpy()

sentence = tokenizer.convert_ids_to_tokens(list(sample.squeeze().numpy()))
sentence = [i.replace('Ġ', ' ') for i in sentence]
text_plot(np.array(sentence)[1:-1],
          ig.squeeze()[1:-1],
          include_legend=True)



