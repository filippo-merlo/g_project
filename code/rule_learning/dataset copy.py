import os
import torch
import random 
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as TT
from torch.utils.data.dataset import Dataset

from config import *
from util import *


class MyDataset():
	def __init__(self, in_path, source, in_base, types,
					dic, vocab, clip_preprocessor=None):
		self.dic = dic
		self.dic_without_logical = {k:v for k,v in self.dic.items() if ' ' not in k}
		self.source = source
		self.types = types
		self.in_path = in_path
		self.totensor = TT.ToTensor()
		self.resize = TT.Resize((resize, resize))
		self.clip_preprocessor = clip_preprocessor
		
		# convert vocab list to dic
		self.vocab = vocab
		self.vocab_nums = {xi: idx for idx, xi in enumerate(self.vocab)}

		# Get list of test images
		self.names_list = []
		with open(os.path.join(self.in_path, 'names', in_base)) as f:
			lines = f.readlines()
			for line in lines:
				self.names_list.append(line[:-1])

		self.name_set = set(self.names_list)

		# Add a filter for names not allowed in the train set:
		# define a set of objects to remove
		
		self.name_set_filtered = self.name_set.copy()
		for name in self.name_set_filtered:
			split = name.split('_')
			c = split[0]
			m = split[1]
			s = split[2]
			if (c,m,s) in obj_to_remove:
				self.name_set_filtered.remove(name)

	def __len__(self):
		return len(self.names_list)

	# only for CLIP emb
	def __getitem__(self, idx):
		base_name = self.names_list[idx]
		image = self.img_emb(base_name)

		# get label indicies
		nm = pareFileNames(base_name)
		num_labels = [self.vocab_nums[li] for li in [nm['color'],
						nm['material'], nm['shape']]]

		#  turn num_labels into one-hot
		labels = torch.zeros(len(self.vocab))
		for xi in num_labels:
			labels[xi] = 1

		return labels, image

	def img_emb(self, base_name):
		# get names
		names = []
		for tp in self.types:
			names.append(os.path.join(self.in_path, self.source,
							base_name + '_' + tp + '.png'))

		# if clip preprocess
		if self.clip_preprocessor is not None:
			images = self.clip_preprocessor(Image.open(names[0]))
			return images

		# preprocess images
		images = []
		for ni in range(len(names)):
			input_image = Image.open(names[ni]).convert('RGB')
			input_image = self.totensor(input_image)

			if names[ni][-16:] == "segmentation.png":
				input_image = input_image.sum(dim=0)
				vals_seg = torch.unique(input_image)
				seg_map = []

				# generate one hot segmentation mask
				for i in range(len(vals_seg)):
					mask = input_image.eq(vals_seg[i])
					# hack: only keep the non-background segmentation masks
					if mask[0][0] is True:
						continue
					seg_mapi = torch.zeros([input_image.shape[0],
						input_image.shape[1]]).masked_fill_(mask, 1)
					seg_map.append(seg_mapi)

				seg_map = torch.cat(seg_map).unsqueeze(0)
				images.append(seg_map)
			else:
				images.append(input_image)

			images[ni] = self.resize(images[ni])

		# (d, resize, resize), d = 3 + #objs (+ other img types *3)
		images = torch.cat(images)
		return images
	
	# Define batch for rules discovering learning 		
	def get_batches_for_rules(self, rule, batch_size = 132, force_rule = False):
		sim_batch = batch_size
		base_names_sim = []
		base_names_dif = []
		images_sim = []
		images_dif = []

		fact_1_class = rule[0][0]
		fact_1_attr = rule[0][1]
		fact_2_class = rule[1][0]
		fact_2_attr = rule[1][1]

		def get_random_attribute(attribute_list, exclude=None):
			attr = random.choice(attribute_list)
			while attr == exclude:
				attr = random.choice(attribute_list)
			return attr

		def create_base_name(names_dic):
			return f'{names_dic["color"]}_{names_dic["material"]}_{names_dic["shape"]}_shade_{names_dic["shade"]}_stretch_{names_dic["stretch"]}_scale_{names_dic["scale"]}_brightness_{names_dic["brightness"]}_view_{names_dic["view"]}'

		while len(base_names_sim) < sim_batch: #133
			names_dic_sim = {}
			names_dic_dif = {}
			skip = False
			for k, v in self.dic_without_logical.items(): # iterate on 'attribute_type':[list of attributes]
				if skip and k == fact_2_class:
					continue
				names_dic_sim[k] = get_random_attribute(v) 
				names_dic_dif[k] = get_random_attribute(v) 
				# If force rule make all the positives examples of the rule
				if force_rule and k == fact_1_class:
					names_dic_sim[fact_1_class] = fact_1_attr
				
				if k == fact_1_class and names_dic_sim[k] == fact_1_attr:
					skip = True
					names_dic_sim[fact_2_class] = random.sample(fact_2_attr,1)[0]
					names_dic_dif[fact_2_class] = get_random_attribute(v) 
			base_name_sim = create_base_name(names_dic_sim) # we create the name of the image from the dict
			base_name_dif = create_base_name(names_dic_dif) # we create the name of the image from the dict		
			
			print(base_name_dif)

			if base_name_sim in self.name_set and base_name_dif in self.name_set:
					base_names_sim.append(base_name_sim)
					image = self.img_emb(base_name_sim)
					images_sim.append(image)

					base_names_dif.append(base_name_dif)
					image = self.img_emb(base_name_dif)
					images_dif.append(image)

		images_sim = torch.stack(images_sim) 
		images_dif = torch.stack(images_dif)
		return base_names_sim, images_sim, base_names_dif, images_dif


