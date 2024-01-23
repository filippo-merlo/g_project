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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from config import *
from dataset import *
from models import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def my_train_clip_encoder(dt, memory, attr, lesson):
	# get model
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
	if lesson in memory.keys():
		print("______________ loading_____________________")
		model.load_state_dict(memory[lesson]['model'])
	optimizer = optim.Adam(model.parameters(), lr=lr)
	model.train().to(device)

	loss_sim = None
	loss_dif = None
	loss = 10
	ct = 0
	centroid_sim = torch.rand(1, latent_dim).to(device)
	while loss > 0.008:
		ct += 1
		if ct > 5:
			break
		for i in range(200):
			# Get Inputs: sim_batch, (sim_batch, 4, 132, 132)
			names_sim, images_sim, names_dif, images_dif = dt.get_paired_batches(attr,lesson)
			images_sim = images_sim.to(device)

			# run similar model
			z_sim = model(clip_model, images_sim)
			centroid_sim = centroid_sim.detach()
			centroid_sim, loss_sim = get_sim_loss(torch.vstack((z_sim, centroid_sim)))

			# Run Difference
			images_dif = images_dif.to(device)

			# run difference model
			z_dif = model(clip_model, images_dif)
			loss_dif = get_sim_not_loss(centroid_sim, z_dif)

			# compute loss
			loss = (loss_sim)**2 + (loss_dif-1)**2
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print('[', ct, ']', loss.detach().item(), loss_sim.detach().item(),
				loss_dif.detach().item())

	############ save model #########
	with torch.no_grad():
		memory[lesson] = {'model': model.to('cpu').state_dict(),
						'arch': ['Filter', ['para_block1']],
						'centroid': centroid_sim.to('cpu')
						}
	return memory

def my_clip_train(in_path, out_path, n_split, model_name, source, in_base,
				types, dic, vocab, pre_trained_model=None):
	# get data
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	dt = MyDataset(in_path, source, in_base, types, dic, vocab,
					clip_preprocessor=clip_preprocess)

	# load encoder models from memory
	memory = {}
	if pre_trained_model is not None:
		print(">>>>> loading memory >>>>>")
		in_memory = os.path.join(out_path, pre_trained_model)
		infile = open(in_memory, 'rb')
		memory = pickle.load(infile)
		infile.close()

	best_nt = 0
	t_tot = 0
	if n_split == 0:
		learning_list = types_logical_with_learning
	elif n_split == 1:
		learning_list = types_logical_with_learning_1
	elif n_split == 2:	
		learning_list = types_logical_with_learning_2
	elif n_split == 3:
		learning_list = types_logical_with_learning_3

	for i in range(epochs):
		for tl in learning_list:  # attr
			random.shuffle(dic[tl])
			for vi in dic[tl]:  # lesson
				print("#################### Learning: " + str(i) + " ----- " + str(vi))
				t_start = time.time()
				memory = my_train_clip_encoder(dt, memory, tl, vi)
				t_end = time.time()
				t_dur = t_end - t_start
				t_tot += t_dur

				print("Time: ", t_dur, t_tot)
				with open(os.path.join(out_path, model_name+'_'+str(n_split)), 'wb') as handle:
					pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--in_path', '-i',
				help='Data input path', required=True)
	argparser.add_argument('--out_path', '-o',
				help='Model memory output path', required=True)
	argparser.add_argument('--n_split', '-s', default=0,
				help='Split number', required=None)
	argparser.add_argument('--model_name', '-n', default=f'my_best_mem.pickle',
				help='Best model memory to be saved file name', required=False)
	argparser.add_argument('--pre_train', '-p', default=None,
				help='Pretrained model import name (saved in outpath)', required=False)
	
	args = argparser.parse_args()

	my_clip_train(args.in_path, args.out_path, args.model_name,
				'train/', bn_train, ['rgba'], dic_train_logical, all_vocabs, args.pre_train)
