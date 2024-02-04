#%%
import pickle
import clip 
from config import *
from dataset import MyDataset
from util import *
import argparse

# Function to load a list from a file using pickle
def load_list(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        # If the file does not exist yet, return an empty list
        return []

# Function to save a list to a file using pickle
def save_list(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Build the dataset object
def get_datasets(in_path,out_path):
    parameters_list = [
        ['train', bn_train, ['rgba']],
        ['test', bn_test, tyimgs],
    ]
    dic = dic_train_logical
    vocab = all_vocabs
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    for parameters in parameters_list:
        source, in_base, types = parameters
        out_path = os.path.join(out_path, parameters[0]+'_dataset.pickle')
        dt = MyDataset(in_path, source, in_base, types, dic, vocab,
                            clip_preprocessor=clip_preprocessor)
        
        for attribute, lessons in dic_train_logical.items():
            l = list(dic_train_logical.keys())
            print('Attributes completed:',l.index(attribute)/len(dic_train_logical.items()),'%')

            for lesson in lessons:
                 
                 for i in range(500):
                    print('Batches completed:',i/500,'%')
                    base_names_sim, images_sim, base_names_dif, images_dif = dt.get_paired_batches(attribute, lesson, batch_size = 132)
                    all_lessons = load_list(out_path)
                    all_lessons.append(
                        [
                        attribute,lesson,
                        images_sim,images_dif,
                        base_names_sim,base_names_dif
                        ]
                    )
                    save_list(out_path, all_lessons)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--in_path', '-i',
                help='Data input path', required=True)
    argparser.add_argument('--out_path', '-o',
                help='Dataset folder output path', required=True)

    args = argparser.parse_args()

    get_datasets(args.in_path,args.out_path)
    
