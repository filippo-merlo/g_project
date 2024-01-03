#%%
import pickle
import os
import torch
import clip
import time
import pickle
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from torch.utils.data import DataLoader

from config import *
from dataset import *
from models import *

device = "cuda" if torch.cuda.is_available() else "cpu"

path = 'best_mem.pickle'

with open(path, 'rb') as f:
    memory = pickle.load(f)
#%%
in_path = '/Users/filippomerlo/Desktop/Datasets/SOLA'
source = 'train'
memory = memory
in_base = bn_train
types = tyimgs
dic = dic_train_logical
vocab = all_vocabs
# %%	
def my_clip_evaluation(in_path, source, memory, in_base, types, dic, vocab):

	with torch.no_grad():
		# get vocab dictionary
		if source == 'train':
			dic = dic_test
		else:
			dic = dic_train

		# get dataset
		clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
		dt = MyDataset(in_path, source, in_base, types, dic, vocab,
					clip_preprocessor=clip_preprocess)
		data_loader = DataLoader(dt, batch_size=132, shuffle=True)

		top3 = 0
		top3_color = 0
		top3_material = 0
		top3_shape = 0
		tot_num = 0
		i = 0
		#for base_is, images in data_loader: # labels (one hot), images (clip embs)
		base_is, images = next(iter(data_loader))
		# Prepare the inputs
		images = images.to(device)
		ans = []
		rel = []
		batch_size_i = len(base_is)

		# go through memory
		for label in vocab: # select a label es 'red'
			if label not in memory.keys():
				ans.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
				continue

			# load model
			model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
			model.load_state_dict(memory[label]['model']) # load weights corresponding to red
			model.to(device)
			model.eval() # freeze

			# load centroid
			centroid_i = memory[label]['centroid'].to(device)
			centroid_i = centroid_i.repeat(batch_size_i, 1)

			# compute stats
			z = model(clip_model, images).squeeze(0)
			disi = ((z - centroid_i)**2).mean(dim=1)
			ans.append(disi.detach().to('cpu'))

		# get top3 indices
		ans = torch.stack(ans, dim=1)
		values, indices = ans.topk(3, largest=False)
		_, indices_lb = base_is.topk(3)
		# base_is = [00001000000010000001]
		# indices_lb [5,12,19]
		indices_lb, _ = torch.sort(indices_lb)

		# calculate stats
		tot_num += len(indices)
		for bi in range(len(indices)):
			ci = 0
			mi = 0
			si = 0
			print('***',indices[bi],'***')
			if indices_lb[bi][0] in indices[bi]:
				print(indices_lb[bi][0])
				ci = 1
			if indices_lb[bi][1] in indices[bi]:
				print(indices_lb[bi][1])
				mi = 1
			if indices_lb[bi][2] in indices[bi]:
				print(indices_lb[bi][2])
				si = 1

			top3_color += ci
			top3_material += mi
			top3_shape += si
			if (ci == 1) and (mi == 1) and (si == 1):
				top3 += 1

		print(tot_num, top3_color/tot_num, top3_material/tot_num,
				top3_shape/tot_num, top3/tot_num)
	return top3/tot_num

n = my_clip_evaluation(in_path, source, memory, in_base, types, dic, vocab)
print(n)