#%%
import os
import json
import clip 
from config import *
from dataset import MyDataset
from util import *
import argparse
from tqdm import tqdm

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
        ['train_novel_obj', bn_train, ['rgba'], dic_train_logical],
        ['test_novel_obj', bn_test, ['rgba'], dic_train_logical],
        ['train_var', bn_train, ['rgba'], dic_train_logical],
        ['test_var', bn_test, ['rgba'], dic_test_logical],
    ]
    vocab = vocabs

    for parameters in parameters_list:
        source, in_base, types, dic = parameters

        if source == 'train_novel_obj':
            train = True
        else:
            train = False

        out_path = os.path.join(out_path, parameters[0]+'_dataset.json')
        save_list(out_path, []) ## After doing this one time, comment this line
        dt = MyDataset(in_path, source, in_base, types, dic, vocab)
        pprint(dic_train_logical)
        for lesson in tqdm(all_vocabs, desc="Lessons", unit="lesson"):
            attribute = get_key_from_value(dic_train_logical, lesson)

            for i in tqdm(range(500), desc="Batches", unit="batch"):               
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




