#%%
'''
Learning Attributes:
	- Color (6)
	- Material (3)
	- Shape (8)
Additional Attributes:
	- Color (2)
	- Material (1)
	- Shape (3)
Flexibility:
	- Camera angle (6)
	- Lighting (3)
Variability (Only in testing):
	- Size (2) 		[Default: large]
	- Stretch (3) 	[Default: normal]
	- Color shade (2) [Default: base]

Naming convension:
[color]_[material]_[shape]_shade_[]_stretch_[]_scale_[]_brightness_view_[]_[tyimg].png
e.g.
aqua_glass_cone_shade_base_stretch_normal_scale_large_brightness_bright_view_0_-2_3_rgba.png
'''
# Learning attributes:
colors = ['brown', "green", "blue", "aqua", "purple", "red", "yellow", 'white']
materials = ['rubber', 'metal', 'plastic', 'glass']
shapes = ["cube", "cylinder", "sphere", "cone", "torus", "gear",
			"torus_knot", "sponge", "spot", "teapot", "suzanne"]
vocabs = colors+materials+shapes


# Flexibility:
views = ['0_3_2', '-2_-2_2', '-2_2_2',  '1.5_-1.5_3', '1.5_1.5_3', '0_-2_3']
brightness = ['dim', 'normal', 'bright']

# Variability
scale_train = ['large']
stretch_train = ['normal']
shade_train = ['base']

scale_test = ['small', 'medium', 'large']
stretch_test = ['normal', 'x', 'y', 'z']
shade_test = ['base', 'light', 'dark']

others = views + brightness + scale_test + stretch_test + shade_test

# Types of images
tyimgs = ['rgba', 'depth', 'normal', 'object_coordinates', 'segmentation']


dic_train = {"color": colors,
			"material": materials,
			"shape": shapes,
			"view": views,
			'brightness': brightness,
			"scale": scale_train,
			'stretch': stretch_train,
			'shade': shade_train
			}
dic_test = {"color": colors,
			"material": materials,
			"shape": shapes,
			"view": views,
			'brightness': brightness,
			"scale": scale_test,
			'stretch': stretch_test,
			'shade': shade_test
			}

types_learning = ['color', 'material', 'shape']
types_flebility = ['color', 'material', 'shape', 'brightness', 'view']
types_variability = ['scale', 'stretch', 'shade']
types_all = ['color', 'material', 'shape', 'brightness',
				'view', 'shade', 'stretch', 'scale']

# make dicts for logical traing and testing
relations = ['and', 'or', 'not'] # <--- new 
types_logical = [] 
for i in types_learning:
	for j in relations:
		if j == 'not':
			types_logical.append(j+' '+i)
		else:
			for h in types_learning:
				if h+' '+j+' '+i not in types_logical:
					if j == 'and' and i == h:
						pass
					else:
						types_logical.append(i+' '+j+' '+h)
types_logical_with_learning =  types_logical + types_learning 

from itertools import product
from pprint import pprint
dic_train_logical = dic_train.copy()
for rel in types_logical:
	if rel.split(' ')[0] == 'not':
		attr = rel.split(' ')[1]
		dic_train_logical[rel] = [f'not {x}' for x in dic_train[attr]]
	else:
		attr1 = rel.split(' ')[0]
		r = rel.split(' ')[1]
		attr2 = rel.split(' ')[2]
		dic_train_logical[rel] = [f'{x} {r} {y}' for x, y in product(dic_train[attr1], dic_train[attr2]) if x != y]

all_vocabs = []
for v in dic_train_logical.values():
	for n in v:
		if n not in others:
			all_vocabs.append(n)

# to make training shorter
import random as rn 
rn.seed(42)

short_types_logical_with_learning = [
 'color and material',
 'color and shape',
 'color or material',
 'color or shape',
 'not color',
 'material and shape',
 'material or shape',
 'not material',
 'not shape']

short_dic_train_logical = dic_train_logical.copy()
for k in short_types_logical_with_learning:
    if len(dic_train_logical[k]) > 5:
        short_dic_train_logical[k] = rn.sample(dic_train_logical[k], 5)

short_types_logical_with_learning += types_learning

#print(all_vocabs)
#pprint(dic_train_logical)
#print(types_logical_with_learning)

# paths and filenames
bn_n_train = "bn_n_train.txt"
bsn_novel_train_1 = "bsn_novel_train_1.txt"
bsn_novel_train_2 = "bsn_novel_train_2.txt"
bsn_novel_train_2_nw = "bsn_novel_train_2_nw.txt"
bsn_novel_train_2_old = "bsn_novel_train_2_old.txt"

bn_n_test = "bn_n_test.txt"
bsn_novel_test_1 = "bsn_novel_test_1.txt"
bsn_novel_test_2_nw = "bsn_novel_test_2_nw.txt"
bsn_novel_test_2_old = "bsn_novel_test_2_old.txt"

bn_train = "bn_train.txt"
bn_test = "bn_test.txt"
bsn_test_1 = "bsn_test_1.txt"
bsn_test_2_nw = "bsn_test_2_nw.txt"
bsn_test_2_old = "bsn_test_2_old.txt"

# train parameters
resize = 224
lr = 1e-3
epochs = 5

sim_batch = 132
gen_batch = 132
batch_size = 33

# model architecture
hidden_dim_clip = 128
latent_dim = 16
