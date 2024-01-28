#%%
import pickle
import torch 
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import clip 
from models import Decoder
from config import *
from dataset import MyDataset
from util import *
from pprint import pprint
import random 

random.seed(42)

print(torch.backends.mps.is_available()) #the MacOS is higher than 12.3+
print(torch.backends.mps.is_built()) #MPS is activated
device = torch.device('mps')

memory_path = '/Users/filippomerlo/Desktop/memories/best_mem_decoder_logic_small.pickle'

with open(memory_path, 'rb') as f:
        memory_base = pickle.load(f)

def get_key_from_value(dictionary, target_value):
    target = ''
    for key, value in dictionary.items():
        for v in value:
            if v == target_value:
                target = key
    return target 

in_path = '/Users/filippomerlo/Desktop/Datasets/SOLA'
source = 'train'
in_base = bn_train
types = ['rgba']
dic = dic_train_logical
vocab = vocabs

clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
clip_model.eval()

dt = MyDataset(in_path, source, in_base, types, dic, vocab, clip_preprocessor)
data_loader = DataLoader(dt, batch_size=100, shuffle=True)

train_labels, train_features = next(iter(data_loader))
_, idxs = train_labels.topk(3)
idxs, _ = torch.sort(idxs)

# some operations
with torch.no_grad():
    acc = dict()
    n_trials_per_attr = dict()
    n_trials = 100
    for trial in range(n_trials):
        # get samples for the trial 
        # get their one-hot encoded features
        train_labels, train_features = next(iter(data_loader))
        _, idxs = train_labels.topk(3)
        idxs, _ = torch.sort(idxs)
        # encode the images with clip
        ans = []
        for i,im in enumerate(train_features):
            ans.append(clip_model.encode_image(im.unsqueeze(0).to(device)).squeeze(0))
        ans = torch.stack(ans)

        # get the answers
        #for attr in types_learning:
        for lesson in memory_base.keys():
            if 'decoder' in memory_base[lesson].keys(): 
                attr = get_key_from_value(dic, lesson)
                if attr not in acc.keys():
                    acc[attr] = 0
                    n_trials_per_attr[attr] = 0
                n_trials_per_attr[attr] += 1
                answers = dict()
                #for lesson in dic[attr]:
                centroid = memory_base[lesson]['centroid'].to(device)
                dec = Decoder(latent_dim).to(device)
                dec.load_state_dict(memory_base[lesson]['decoder'])
                decoded_rep = dec(centroid)
                C = decoded_rep.repeat(ans.shape[0], 1)
                disi = ((ans - C)**2).mean(dim=1).detach().to('cpu')
                v, topk_idxs = disi.topk(1, largest=False)
                answers[lesson] = [idxs[i] for i in topk_idxs]

                for k in answers.keys():
                    for coded in answers[k]:
                        color = vocabs[coded[0]]
                        material = vocabs[coded[1]]
                        shape = vocabs[coded[2]]
                        if 'and' in k.split():
                            l1 = k.split()[0]
                            l2 = k.split()[2]
                            if l1 in [color, material, shape] and l2 in [color, material, shape]:
                                acc[attr] += 1
                        elif 'or' in k.split():
                            l1 = k.split()[0]
                            l2 = k.split()[2]
                            if l1 in [color, material, shape] or l2 in [color, material, shape]:
                                acc[attr] += 1
                        else:
                            if k in [color, material, shape]:
                                acc[attr] += 1

# print the results
for k in acc.keys():
    print(f'{k}: ',acc[k]/n_trials_per_attr[k])
#%%
color_acc = acc['color']/(len(colors)*n_trials)
material_acc = acc['material']/(len(materials)*n_trials)
shape_acc = acc['shape']/(len(shapes)*n_trials)
print('Color accuracy: {}'.format(color_acc))
print('Material accuracy: {}'.format(material_acc))
print('Shape accuracy: {}'.format(shape_acc))
tot_acc = (color_acc+material_acc+shape_acc)/3
print('Total accuracy: {}'.format(tot_acc))

import matplotlib.pyplot as plt

# Accuracy values
categories = ['Color', 'Material', 'Shape', 'Total']
accuracies = [color_acc, material_acc, shape_acc, tot_acc]

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(categories, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1)  # Setting y-axis limits to represent accuracy values between 0 and 1
plt.title('Accuracy Metrics')
plt.xlabel('Categories')
plt.ylabel('Accuracy')
plt.show()

