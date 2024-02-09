#%%
import os
import json
import clip 
from config import *
from dataset import MyDataset
from util import *
import argparse

# Function to load a list from a file using json
def load_list(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        # If the file does not exist yet, return an empty list
        return []

# Function to save a list to a file using json
def save_list(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # If the target_value is not found in the dictionary, you can return a default value or raise an exception.
    # In this example, None is returned if the value is not found.
    return None

# Build the dataset object
def get_datasets(in_path,out_path):
    parameters_list = [
        ['train', bn_train, ['rgba']],
        ['test', bn_test, ['rgba']],
    ]
    dic = dic_train_logical
    vocab = all_vocabs
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    for parameters in parameters_list:
        source, in_base, types = parameters
        out_path = os.path.join(out_path, parameters[0]+'_dataset.json')
        save_list(out_path, []) ## After doing this one time, comment this line
        dt = MyDataset(in_path, source, in_base, types, dic, vocab,
                            clip_preprocessor=clip_preprocessor)
        
        for lesson in vocab:
            attribute = get_key_from_value(dic, lesson)

            for i in range(500):
            print('Batches completed:',i/500,'%')
            if source == 'train':
                train = True
            else:
                train = False
            base_names_sim, base_names_dif = dt.get_paired_batches_names(attribute, lesson, 132, train)
            all_lessons = load_list(out_path)
            all_lessons.append(
                {
                'attribute' : attribute,
                'lesson' : lesson,
                'base_names_sim' : base_names_sim,
                'base_names_sim' : base_names_dif
                }
            )
            save_list(out_path, all_lessons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get datasets')
    parser.add_argument('--in_path', type=str, help='Path to the dataset')
    parser.add_argument('--out_path', type=str, help='Path to the output')
    args = parser.parse_args()
    
    get_datasets(args.in_path,args.out_path)




