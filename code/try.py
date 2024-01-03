#%%
from dataset import *
from config import *

from torch.utils.data import DataLoader
from pprint import pprint

import matplotlib.pyplot as plt

in_path = '/Users/filippomerlo/Desktop/Datasets/SOLA'
source = 'train'
in_base = bn_train
types = tyimgs
dic = dic_train_logical
vocab = all_vocabs

dt = MyDataset(in_path, source, in_base, types, dic, vocab)
data_loader = DataLoader(dt, batch_size=132, shuffle=True)

#%%
train_labels, train_features = next(iter(data_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
pprint(train_features)

#%% Class Methods
#dt.__len__()
a,b = dt.__getitem__(1240)
print(a)
#%% NEW Functions
# - With attributes pairing between positive and negative samples 

attr = 'color'
lesson = 'red'
names_sim, images_sim, names_dif, images_dif = dt.get_paired_batches(attr,lesson)

for i,n in enumerate(names_sim):
    print('**********',i,'**********')
    print(n,'\n',names_dif[i])
    
