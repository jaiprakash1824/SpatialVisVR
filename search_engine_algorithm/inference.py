#### Libraries

import json
from os import makedirs
from os.path import exists, join
from argparse import ArgumentParser
from os import makedirs, path
import torch
from Utils._aux import create_dir, save_transformation, load_transformation
import tifffile as tif
import cv2 as cv
import numpy as np
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import random
from datetime import datetime
import csv
from PIL import Image
import pandas as pd
import scanpy as sc
import anndata as ad
import re
import heapq
from utils import get_sorted_df, get_votes, get_majority_counts,HE_to_codex,get_subdictionary,list_to_csv,preprocessing_data
import requests

import matplotlib.pyplot as plt

from Model.model import CustomVAE
#from data_module import HE_DataModule
#from Utils._aux import create_dir

def main_func(args, file_path=None):

    dict_args = vars(args)

    #print(dict_args )

    data_args=get_subdictionary(dict_args,0,5)
    model_args=get_subdictionary(dict_args,5,14)
    # TODO change the path
    ch_address='./epoch=54-step=605_colon.ckpt'
    #filename='/home/data/nolan_lab/Tonsil_HandE/bestFocus/reg001_X02_Y05_Z08.tif'
   
    channels_database=[(0,'Hoechst'),(22,'CD4'),(9,'CD8'),(17,'HLADR'),(19,'Ki67'),(21,'CD45RA'),(23,'CD21')]
    #channels_database=[0,22,9,17,19,21,23]
    
    

    specific_slide_addr = ''
    if file_path is not None:
        specific_slide_addr = file_path
    else:  
        # TODO change the path
        specific_slide_addr = './cropped/cropped_8.png'

    data_obj= preprocessing_data(**data_args)
    
    model = CustomVAE(**model_args)
    
    input_data,result = data_obj._patching(specific_slide_addr)


    for i in range(len(result)):
        a=result[i].reshape(64,64)
        a=a.numpy()

    
        image = (a * 255).astype(np.uint8)
        image1 = Image.fromarray(image, 'L')
        image1.save('img_phone/'+str(i)+'.png')

    print('have fun')
    
    model = model.load_from_checkpoint(ch_address,  map_location=torch.device("cpu"))

    print('have fun1')
    
    for i in range(len(input_data)):
        output=model(input_data[i][3])
        del input_data[i][3]
        input_data[i].append(output)

    print('have fun2')

    input_data_df= pd.DataFrame(input_data)
    input_df_addr='./input_data_df.csv'
    input_data_df.to_csv(input_df_addr, index=False)

    print('have fun3')
    
    similarity_patches=HE_to_codex(channels_database,input_df_addr,specific_slide_addr)

    print('have fun4')


    sorted_similarity_patches = get_sorted_df(similarity_patches, save=False)
    patch_slide_channel_sim_dict = get_votes(sorted_similarity_patches , threshold=0.2)
    majority_vote_dict = get_majority_counts(patch_slide_channel_sim_dict)

    print(majority_vote_dict)

    # requests.request('GET', "http://172.20.10.7:5002/setsearchresults/", headers={'Content-Type': 'application/json'}, json=list(majority_vote_dict.keys())[:5])
    
    data_list = list(majority_vote_dict.keys())[:5]

    file_path = "results.txt"

    # Write to file
    with open(file_path, 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)
    
    
    print('"""""""""""""""""""""""""""""""""""""""""')
    
def preprocess(file_path): 
    parser = ArgumentParser()
    
    parser=preprocessing_data.specific_args(parser)
    parser = CustomVAE.add_model_specific_args(parser)
   
    args = parser.parse_args()
    
    main_func(args, file_path)
    
#def sendToVR(majority_vote_dict):
    # Play Here only