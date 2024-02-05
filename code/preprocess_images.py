#%%
import os
import pickle
import clip 
from config import *
from dataset import MyDataset
from util import *
import argparse
from PIL import Image
import re


# Build the dataset object
def get_preprocessed_images(in_path,out_path):
    images_sets = [
        'train',
        'test',
        'novel_train',
        'novel_test'
    ]
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    for images_set in images_sets:
        print(images_set)
        folder_path = os.path.join(in_path, images_set)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if re.search(re.escape(r'\.png$'),file_path):
                image = clip_preprocessor(Image.open(file_path))
                out_file_path = os.path.join(out_path, images_set, filename)
                out_file_path = re.sub(r'\.png$', '', out_file_path, flags=re.IGNORECASE)
                with open(out_file_path+'.pickle', 'wb') as file:
                    pickle.dump(image, file)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--in_path', '-i',
                help='Data input path', required=True)
    argparser.add_argument('--out_path', '-o',
                help='Dataset folder output path', required=True)

    args = argparser.parse_args()

    get_preprocessed_images(args.in_path,args.out_path)
    
