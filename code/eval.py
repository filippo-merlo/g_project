#%%
import torch
import clip

from torch.utils.data import DataLoader

from config import *
from dataset import *
from models import *

import pickle
import argparse

#print(torch.backends.mps.is_available())  # the MacOS is higher than 12.3+
#print(torch.backends.mps.is_built())  # MPS is activated
#device = torch.device('mps')
device = "cuda" if torch.cuda.is_available() else "cpu"

def my_clip_evaluation(in_path, source, memory, in_base, types, dic, vocab):
    with torch.no_grad():
        # get vocab dictionary
        if source == 'train':
            dic = dic_test
        else:
            dic = dic_test

        # get dataset
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        dt = MyDataset(in_path, source, in_base, types, dic, vocab, clip_preprocessor=clip_preprocess)
        data_loader = DataLoader(dt, batch_size=10, shuffle=True)

        top3 = 0
        top3_color = 0
        top3_material = 0
        top3_shape = 0

        top3_and = 0
        top3_color_and_material = 0
        top3_color_and_shape = 0
        top3_material_and_shape = 0
        tot_num_and = 0

        top3_or = 0
        top3_color_or_material = 0
        top3_color_or_shape = 0
        top3_material_or_shape = 0
        top3_color_or_color = 0
        top3_material_or_material = 0
        top3_shape_or_shape = 0
        tot_num_or = 0

        top3_not = 0
        top3_not_color = 0
        top3_not_material = 0
        top3_not_shape = 0
        tot_num_not = 0

        tot_num = 0

        for base_is, images in data_loader:
            # Prepare the inputs
            images = images.to(device)
            ans = []
            batch_size_i = len(base_is)

            # go through memory
            for label in vocabs:
                if label not in memory.keys():
                    ans.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
                    continue

                # load model
                model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
                model.load_state_dict(memory[label]['model'])
                model.to(device)
                model.eval()

                # load centroid
                centroid_i = memory[label]['centroid'].to(device)
                centroid_i = centroid_i.repeat(batch_size_i, 1)

                # compute stats
                z = model(clip_model, images).squeeze(0)
                disi = ((z - centroid_i) ** 2).mean(dim=1)
                ans.append(disi.detach().to('cpu'))

            # get top3 incicies
            ans = torch.stack(ans, dim=1)
            values, indices = ans.topk(3, largest=False)
            _, indices_lb = base_is.topk(3)
            indices_lb, _ = torch.sort(indices_lb)

            # calculate stats
            tot_num += len(indices)
            for bi in range(len(indices)):
                ci = 0
                mi = 0
                si = 0
                if indices_lb[bi][0] in indices[bi]:
                    ci = 1
                if indices_lb[bi][1] in indices[bi]:
                    mi = 1
                if indices_lb[bi][2] in indices[bi]:
                    si = 1

                top3_color += ci
                top3_material += mi
                top3_shape += si

                if (ci == 1) and (mi == 1) and (si == 1):
                    top3 += 1

            print(tot_num, top3_color / tot_num, top3_material / tot_num, top3_shape / tot_num, top3 / tot_num)

            rel_list = []
            ans_logical = []
            for label in all_vocabs:
                if ' ' in label:
                    if label not in memory.keys():
                        ans_logical.append(torch.full((batch_size_i, 1), 1000.0).squeeze(1))
                        continue
                    s = label.split(' ')
                    if 'not' in s:
                        rel = s[0]
                        attr1 = s[1]
                        attr2 = None
                        rel_list.append([rel, int(vocabs.index(attr1))])
                        print(vocabs.index(attr1))
                    else:
                        rel = s[1]
                        attr1 = s[0]
                        attr2 = s[2]
                        rel_list.append([rel, int(vocabs.index(attr1)), int(vocabs.index(attr2))])
                        print(vocabs.index(attr1), vocabs.index(attr2))
                    # load model
                    model = CLIP_AE_Encode(hidden_dim_clip, latent_dim, isAE=False)
                    model.load_state_dict(memory[label]['model'])
                    model.to(device)
                    model.eval()

                    # load centroid
                    centroid_i = memory[label]['centroid'].to(device)
                    centroid_i = centroid_i.repeat(batch_size_i, 1)

                    # compute stats
                    z = model(clip_model, images).squeeze(0)
                    disi = ((z - centroid_i) ** 2).mean(dim=1)
                    ans_logical.append(disi.detach().to('cpu'))
            # get top3 incicies
            ans_logical = torch.stack(ans_logical, dim=1)
            values, indices = ans_logical.topk(10, largest=False)
            print(indices)

            _, indices_lb = base_is.topk(3)
            indices_lb, _ = torch.sort(indices_lb)
            print(indices_lb)

            # calculate stats
            tot_num += len(indices)
            for bi in range(len(indices)):
                rel = rel_list[bi][0]

                if rel == 'not':
                    attr = rel[1]
                    if attr not in indices[bi]:
                        top3_not += 1
                        tot_num_not += 1
                elif rel == 'and':
                    attr1 = rel[1]
                    attr2 = rel[2]
                    if attr1 in indices[bi] and attr2 in indices[bi]:
                        top3_and += 1
                        tot_num_and += 1
                elif rel == 'or':
                    attr1 = rel[1]
                    attr2 = rel[2]
                    if attr1 in indices[bi] or attr2 in indices[bi]:
                        top3_or += 1
                        tot_num_or += 1

            tot_logical = tot_num_not + tot_num_and + tot_num_or
            print('Logical, tot, not, and , or')
            print(tot_logical / tot_num, top3_not / tot_num_not, top3_and / tot_num_and, top3_or / tot_num_or)

    return top3 / tot_num


#TESTING


source = 'novel_test/'
in_base = bsn_novel_test_1
types = ['rgba']
dic = dic_test_logical
vocab = all_vocabs

#in_path = '/Users/filippomerlo/Desktop/Datasets/SOLA'
#memory_path = '/Users/filippomerlo/Desktop/memories/my_best_mem_1.pickle'

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in_path', type=str, required=True)
    argparser.add_argument('--memory_path', type=str, required=True)
    args = argparser.parse_args()

    with open(args.memory_path, 'rb') as f:
            memory_complete = pickle.load(f)
    for i in range(2, 7):
        pieces = args.memory_path.split('my_best_mem_')
        new_path = pieces[0] + f'my_best_mem_{i}.pickle'
        with open(new_path, 'rb') as f:
            memory = pickle.load(f)
        for k in memory.keys():
            memory_complete[k] = memory[k]

    t = my_clip_evaluation(args.in_path, source, memory_complete, in_base, types, dic, vocab)

