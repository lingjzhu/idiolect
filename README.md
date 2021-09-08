### *Idiosyncratic but not Arbitrary: Learning Idiolects in Online Registers Reveals Distinctive yet Consistent Individual Styles*  
To appear in EMNLP 2021 main conference. [arXiv](https://arxiv.org/abs/2109.03158)  

#### Requirements
python >= 3.7  
transformers >= 4.4  
torch >= 1.7  

#### Data & Pre-trained weights
The data and pre-trained models used in the study can be found [here](https://drive.google.com/drive/folders/1yIK56tYtFSeJxPWb8RujdGmkJGq4b67_?usp=sharing).


#### Training a model
```python model_roberta_self_attention.py  --training_data ./authorship/Amazon_train_contrastive --develop_data ./authorship/Amazon_dev_contrastive --train --tau_low 0.4 --tau_high 0.6 --loss modified_anchor --mask_prob 0.1 --test_data ./authorship/Amazon_test_contrastive --test --save_test_data --distance cosine --epochs 6 --alpha 30 ```

#### Evaluating style embeddings
