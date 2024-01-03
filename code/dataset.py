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
	
	def get_paired_batches(self, attribute, lesson):
		base_names_sim = []
		base_names_dif = []
		images_sim = []
		images_dif = []
		if 'and' in attribute.split() or 'or' in attribute.split():
			lesson = (lesson.split()[0], lesson.split()[2])
		elif 'not' in attribute.split():
			lesson = lesson.split()[1]

		if ' ' not in attribute: # if the attribute is not logical
			while len(base_names_sim) < sim_batch: #133
				names_dic_sim = {}
				names_dic_dif = {}

				for k, v in self.dic.items(): # iterate on 'attribute_type':[list of attributes]
					if ' ' not in k: # if the attribute is not logical
						if k == attribute: # if the attribute is the one we want to teach e.g. color
							names_dic_sim[k] = lesson # we take the lesson e.g. red
							tp = random.choice(v)
							while (tp == lesson):
								tp = random.choice(v)
							names_dic_dif[k] = tp
						else:
							tpo = random.choice(v) # we take a random attribute from the list of attributes e.g. blue
							names_dic_sim[k] = tpo 
							names_dic_dif[k] = tpo 
				base_name_sim = f'{names_dic_sim["color"]}_{names_dic_sim["material"]}_{names_dic_sim["shape"]}_shade_{names_dic_sim["shade"]}_stretch_{names_dic_sim["stretch"]}_scale_{names_dic_sim["scale"]}_brightness_{names_dic_sim["brightness"]}_view_{names_dic_sim["view"]}' # we create the name of the image from the dict
				base_name_dif = f'{names_dic_dif["color"]}_{names_dic_dif["material"]}_{names_dic_dif["shape"]}_shade_{names_dic_dif["shade"]}_stretch_{names_dic_dif["stretch"]}_scale_{names_dic_dif["scale"]}_brightness_{names_dic_dif["brightness"]}_view_{names_dic_dif["view"]}' # we create the name of the image from the dict		
				
				if base_name_sim in self.name_set and base_name_dif in self.name_set:
					base_names_sim.append(base_name_sim)
					image = self.img_emb(base_name_sim)
					images_sim.append(image)

					base_names_dif.append(base_name_dif)
					image = self.img_emb(base_name_dif)
					images_dif.append(image)

		else: # if the attribute is logical
			while len(base_names_sim) < sim_batch: #133
				names_dic_sim = {}
				names_dic_dif = {}
				if 'and' in attribute.split():
					attribute1 = attribute.split()[0]
					attribute2 = attribute.split()[2]
					for negative_case in range(3): # 0,1,2 [negatives]
						for k, v in self.dic.items(): # iterate on 'attribute_type':[list of attributes]
							if ' ' not in k: # if the attribute is not logical
								if k == attribute1:
									names_dic_sim[k] = lesson[0]
									if negative_case == 0:
										names_dic_dif[k] = lesson[0]
									else:
										tp = random.choice(v)
										while (tp == lesson[0]):
											tp = random.choice(v)
										names_dic_dif[k] = tp
								elif k==attribute2:
									names_dic_sim[k] = lesson[1]
									if negative_case == 1:
										names_dic_dif[k] = lesson[1]
									else:
										tp = random.choice(v)
										while (tp == lesson[1]):
											tp = random.choice(v)
										names_dic_dif[k] = tp
								else:
									tpo = random.choice(v) # we take a random attribute from the list of attributes e.g. blue
									names_dic_sim[k] = tpo 
									names_dic_dif[k] = tpo 
						base_name_sim = f'{names_dic_sim["color"]}_{names_dic_sim["material"]}_{names_dic_sim["shape"]}_shade_{names_dic_sim["shade"]}_stretch_{names_dic_sim["stretch"]}_scale_{names_dic_sim["scale"]}_brightness_{names_dic_sim["brightness"]}_view_{names_dic_sim["view"]}' # we create the name of the image from the dict
						base_name_dif = f'{names_dic_dif["color"]}_{names_dic_dif["material"]}_{names_dic_dif["shape"]}_shade_{names_dic_dif["shade"]}_stretch_{names_dic_dif["stretch"]}_scale_{names_dic_dif["scale"]}_brightness_{names_dic_dif["brightness"]}_view_{names_dic_dif["view"]}' # we create the name of the image from the dict
						
						if base_name_sim in self.name_set and base_name_dif in self.name_set:
							base_names_sim.append(base_name_sim)
							image = self.img_emb(base_name_sim)
							images_sim.append(image)
				
							base_names_dif.append(base_name_dif)
							image = self.img_emb(base_name_dif)
							images_dif.append(image)

				elif 'or' in attribute.split():
					attribute1 = attribute.split()[0]
					attribute2 = attribute.split()[2]
					if attribute1 == attribute2:
						for negative_case in range(2): # 0,1 [negatives]
							for k, v in self.dic.items(): # iterate on 'attribute_type':[list of attributes]
								
								if ' ' not in k: # if the attribute is not logical
									if k == attribute1:
										tp = random.choice(v)
										while (tp == lesson[0] or tp == lesson[1]):
											tp = random.choice(v)
										names_dic_dif[k] = tp

										if negative_case == 0:
											names_dic_sim[k] = lesson[0]
										if negative_case == 1:
											names_dic_sim[k] = lesson[1]
									else:
										tpo = random.choice(v) # we take a random attribute from the list of attributes e.g. blue
										names_dic_sim[k] = tpo 
										names_dic_dif[k] = tpo 
							base_name_sim = f'{names_dic_sim["color"]}_{names_dic_sim["material"]}_{names_dic_sim["shape"]}_shade_{names_dic_sim["shade"]}_stretch_{names_dic_sim["stretch"]}_scale_{names_dic_sim["scale"]}_brightness_{names_dic_sim["brightness"]}_view_{names_dic_sim["view"]}' # we create the name of the image from the dict
							base_name_dif = f'{names_dic_dif["color"]}_{names_dic_dif["material"]}_{names_dic_dif["shape"]}_shade_{names_dic_dif["shade"]}_stretch_{names_dic_dif["stretch"]}_scale_{names_dic_dif["scale"]}_brightness_{names_dic_dif["brightness"]}_view_{names_dic_dif["view"]}' # we create the name of the image from the dict
							
							if base_name_sim in self.name_set and base_name_dif in self.name_set:
								base_names_sim.append(base_name_sim)
								image = self.img_emb(base_name_sim)
								images_sim.append(image)
					
								base_names_dif.append(base_name_dif)
								image = self.img_emb(base_name_dif)
								images_dif.append(image)
					else:
						for negative_case in range(3): # 0,1,2 [negatives]
							for k, v in self.dic.items(): # iterate on 'attribute_type':[list of attributes]
			
								if ' ' not in k: # if the attribute is not logical
									if k == attribute1:
										tp = random.choice(v)
										while (tp == lesson[0]):
											tp = random.choice(v)
										names_dic_dif[k] = tp

										if negative_case == 0 or negative_case == 1:
											names_dic_sim[k] = lesson[0]
										else:
											names_dic_sim[k] = names_dic_dif[k]

									elif k==attribute2:
										tp = random.choice(v)
										while (tp == lesson[1]):
											tp = random.choice(v)
										names_dic_dif[k] = tp

										if negative_case == 0 or negative_case == 2:
											names_dic_sim[k] = lesson[1]
										else:
											names_dic_sim[k] = names_dic_dif[k]
									else:
										tpo = random.choice(v) # we take a random attribute from the list of attributes e.g. blue
										names_dic_sim[k] = tpo 
										names_dic_dif[k] = tpo 
							base_name_sim = f'{names_dic_sim["color"]}_{names_dic_sim["material"]}_{names_dic_sim["shape"]}_shade_{names_dic_sim["shade"]}_stretch_{names_dic_sim["stretch"]}_scale_{names_dic_sim["scale"]}_brightness_{names_dic_sim["brightness"]}_view_{names_dic_sim["view"]}' # we create the name of the image from the dict
							base_name_dif = f'{names_dic_dif["color"]}_{names_dic_dif["material"]}_{names_dic_dif["shape"]}_shade_{names_dic_dif["shade"]}_stretch_{names_dic_dif["stretch"]}_scale_{names_dic_dif["scale"]}_brightness_{names_dic_dif["brightness"]}_view_{names_dic_dif["view"]}' # we create the name of the image from the dict
							
							if base_name_sim in self.name_set and base_name_dif in self.name_set:
								base_names_sim.append(base_name_sim)
								image = self.img_emb(base_name_sim)
								images_sim.append(image)
					
								base_names_dif.append(base_name_dif)
								image = self.img_emb(base_name_dif)
								images_dif.append(image)

				elif 'not' in attribute.split():
					attribute1 = attribute.split()[1]
					for k, v in self.dic.items(): # iterate on 'attribute_type':[list of attributes]
						if ' ' not in k: # if the attribute is not logical
							if k == attribute1: # if the attribute is the one we want to teach e.g. color
								names_dic_dif[k] = lesson # we take the lesson e.g. red
								tp = random.choice(v)
								while (tp == lesson):
									tp = random.choice(v)
								names_dic_sim[k] = tp
							else:
								tpo = random.choice(v) # we take a random attribute from the list of attributes e.g. plastic
								names_dic_sim[k] = tpo 
								names_dic_dif[k] = tpo 
					base_name_sim = f'{names_dic_sim["color"]}_{names_dic_sim["material"]}_{names_dic_sim["shape"]}_shade_{names_dic_sim["shade"]}_stretch_{names_dic_sim["stretch"]}_scale_{names_dic_sim["scale"]}_brightness_{names_dic_sim["brightness"]}_view_{names_dic_sim["view"]}' # we create the name of the image from the dict
					base_name_dif = f'{names_dic_dif["color"]}_{names_dic_dif["material"]}_{names_dic_dif["shape"]}_shade_{names_dic_dif["shade"]}_stretch_{names_dic_dif["stretch"]}_scale_{names_dic_dif["scale"]}_brightness_{names_dic_dif["brightness"]}_view_{names_dic_dif["view"]}' # we create the name of the image from the dict		
					
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
