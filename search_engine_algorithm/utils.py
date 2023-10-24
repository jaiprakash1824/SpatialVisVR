#### Libraries

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
import scanpy as sc
from PIL import Image
import pandas as pd
import heapq
#from utils import get_sorted_df, get_votes, get_majority_counts,HE_to_codex


import matplotlib.pyplot as plt

from Model.model import CustomVAE
#from data_module import HE_DataModule
from Utils._aux import create_dir

#### Functions and Classes
import pandas as pd

def update_codex_channel(row):
    # Condition to update the value
    if 'HOECHST' in row['Codex channel']:
        return 'HOECHST'
    return row['Codex channel'].split(' ')[0]


def get_column_mapping():
    column_mapping = {
        'name of the slide': 'slide_name',
        'patch coordination': 'codex_patch_coord',
        'input patch coordination': 'he_patch_coord',
    }
    return column_mapping


def get_sorted_df(df, save=True, file_to_save='temp'):
    #Standardize Channel Names
    
    df['channel_name'] = df.apply(update_codex_channel, axis = 1)
    

    #Standardize Column Names
    sorted_df = df.groupby(['input patch coordination']).apply(lambda x: x.sort_values('similarity', ascending=False)).reset_index(drop=True)


    # Update the column names based on the dictionary mapping
    sorted_df = sorted_df.rename(columns=get_column_mapping())

    #Drop some column that are not necessary now
    sorted_df.drop(['input slide', 'Codex channel', 'H&E'], axis = 1, inplace=True)
    
    if save:
        sorted_df.to_csv(file_to_save, index=False)
    return sorted_df

def get_votes(df, threshold):
    # Group by he_patch_coord and slide_name, calculate the sum of similarity values
    
    grouped_data = df.groupby(['he_patch_coord'])[['slide_name', 'channel_name', 'similarity']]

    #print(2)

    patch_slide_channel_sim_dict = {}
    
    for name, group in grouped_data:
    #     print(name, group)
        he_patch = name
        patch_slide_channel_sim_dict[he_patch] = {}
        
        this_patch_slide_channel_sim = {}
        
        for index, row in group.iterrows():
            slide_name = row['slide_name']
            channel_name = row['channel_name']
            similarity = row['similarity']
            
            if similarity > threshold:
                
                if slide_name not in this_patch_slide_channel_sim.keys():
                    this_patch_slide_channel_sim[slide_name] = []
                
                this_patch_slide_channel_sim[slide_name].append({channel_name: similarity})
        if len(this_patch_slide_channel_sim) == 0:
            patch_slide_channel_sim_dict[he_patch] = {
                f'Below threshold {threshold}': 'Below threshold'
                
            }
        else:
            patch_slide_channel_sim_dict[he_patch] = this_patch_slide_channel_sim
            
    return patch_slide_channel_sim_dict 


def get_majority_counts(patch_slide_channel_sim_dict, verbose=False):
    patch_vote = {}
    for key in patch_slide_channel_sim_dict:
        max_voted_key_for_this_patch = max(patch_slide_channel_sim_dict[key], key=lambda k: len(patch_slide_channel_sim_dict[key][k]))
    #     print(max_voted_key_for_this_patch)
        
        patch_vote[max_voted_key_for_this_patch] = patch_vote.get(max_voted_key_for_this_patch, 0) + 1 
        

    sorted_dict = {k: v for k, v in sorted(patch_vote.items(), key=lambda item: item[1], reverse=True)}

    # Print the slide votes
    if verbose:
        for slide_name, votes in sorted_dict.items():
            
            print(f"Slide {slide_name} received {votes} vote(s)")
        
    return sorted_dict




def remove_last_item(lst):
    if len(lst) > 0:
        lst.pop()
    return lst




def remove_substring_from_list(input_list):
    return [item.replace("]]", "") for item in input_list]




def HE_to_codex(channels_database_addr,input_df_addr,specific_slide_addr):

    L_df=[]

    HE_colon=pd.read_csv(input_df_addr)
    HE_colon["3"] =  HE_colon.apply(lambda row: row["3"].replace("\n", ""), axis=1)
    HE_colon["3"] =  HE_colon.apply(lambda row: row["3"].replace(" ", ""), axis=1)
    HE_colon["3"] =  HE_colon.apply(lambda row: row["3"].replace("tensor", ""), axis=1)
    HE_colon["3"] =  HE_colon.apply(lambda row: row["3"].strip("()"), axis=1)
    HE_colon["3"] =  HE_colon.apply(lambda row: row["3"].strip("[]"), axis=1)
    HE_colon["3"] =  HE_colon.apply(lambda row: row["3"].split(","), axis=1)
       
    HE_colon['3'] = HE_colon['3'].apply(remove_last_item)
    HE_colon['3'] = HE_colon['3'].apply(remove_substring_from_list)
    HE_colon["3"] =  HE_colon.apply(lambda row: [float(num) for num in row["3"]], axis=1)
    HE_colon["3"] =  HE_colon.apply(lambda row: np.array(row["3"]), axis=1)

    


    HE_colon['sampled_coords'] = HE_colon.apply(lambda row: f"({row['0']}, {row['1']})", axis=1)
    HE_colon.rename(columns={'2': 'filename'}, inplace=True)
    string_to_add = 'H&E_colon'
    HE_colon['primary_site'] = string_to_add
    HE_colon = HE_colon.drop(columns=['0','1'])

    expanded_HE_colon= pd.DataFrame(HE_colon['3'].tolist())
    HE_colon= pd.concat([HE_colon,expanded_HE_colon], axis=1)
    HE_colon= HE_colon.drop('3', axis=1)


    for ch in channels_database_addr:
    
            add=f"./logs/codex_logs/latent_spaces_{ch[0]}_{ch[1]}.csv"
            
            codex_colon= pd.read_csv(add)
        
            codex_colon['primary_site']= codex_colon['primary_site'].replace('Colon', ch[1])  
        

            codex_colon["latent_value"] = codex_colon.apply(lambda row: row["latent_value"].replace("\n", ""), axis=1)
            codex_colon["latent_value"] = codex_colon.apply(lambda row: row["latent_value"].replace(" ", ""), axis=1)
            codex_colon["latent_value"] = codex_colon.apply(lambda row: row["latent_value"].replace("tensor", ""), axis=1)
            codex_colon["latent_value"] = codex_colon.apply(lambda row: row["latent_value"].strip("()"), axis=1)
            codex_colon["latent_value"] = codex_colon.apply(lambda row: row["latent_value"].strip("[]"), axis=1)
            codex_colon["latent_value"] = codex_colon.apply(lambda row: row["latent_value"].split(","), axis=1)
            codex_colon["latent_value"] = codex_colon.apply(lambda row: [float(num) for num in row["latent_value"]], axis=1)
            codex_colon["latent_value"] = codex_colon.apply(lambda row: np.array(row["latent_value"]), axis=1)

            expanded_codex_colon= pd.DataFrame(codex_colon['latent_value'].tolist())
            codex_colon= pd.concat([codex_colon,expanded_codex_colon], axis=1)
            codex_colon= codex_colon.drop('latent_value', axis=1)
            df = pd.concat([codex_colon,HE_colon], axis=0)
            df = df.reset_index(drop=True)
       


        

            
       
        

        

        


            adata = sc.AnnData(
            X=df.iloc[:, 3:]
            )
            a= pd.concat([adata.obs, df['primary_site']], axis=1)
            a=a.dropna(subset=['primary_site'])
            adata.obs=a


            corrected_features=sc.pp.combat(adata, key='primary_site',inplace=False)

        
            adata_corrected = adata.copy()
            adata_corrected.X=corrected_features

            corrected_df= pd.DataFrame(corrected_features)
            filename_corrected_df = pd.DataFrame(df['filename'])
            sampled__corrected_df= pd.DataFrame(df['sampled_coords'])
            primary_corrected_df = pd.DataFrame(df['primary_site'])


            corrected_df= pd.concat([primary_corrected_df, corrected_df], axis=1)
            corrected_df= pd.concat([sampled__corrected_df, corrected_df], axis=1)
            corrected_df= pd.concat([filename_corrected_df, corrected_df], axis=1)
            corrected_df

            HandE_sample_slide= corrected_df[(corrected_df['primary_site'] == 'H&E_colon') & (corrected_df['filename'] == specific_slide_addr)]
            database= corrected_df[~((corrected_df['primary_site'] == 'H&E_colon') & (corrected_df['filename'] == specific_slide_addr))]
            HandE_sample_slide_array=np.array(HandE_sample_slide)
            database_array=np.array(database)

            HandE_codex_similarity=[]

            for i in range(len(HandE_sample_slide_array)):
     
                L_sub=[]
                for j in range(len(database_array)):

                    dot_product = np.dot(HandE_sample_slide_array[i][3:], database_array[j][3:])


                    magnitude_a = np.linalg.norm(HandE_sample_slide_array[i][3:])
                    magnitude_b = np.linalg.norm(database_array[j][3:])
                    cosine_similarity = dot_product / (magnitude_a * magnitude_b)

                    L_sub.append((cosine_similarity,database_array[j][0],database_array[j][1],database_array[j][2],HandE_sample_slide_array[i][0],HandE_sample_slide_array[i][1],HandE_sample_slide_array[i][2]))

    
                HandE_codex_similarity.append(L_sub)

            L_sim=[]
            for i in range(len(HandE_codex_similarity)):
                largest_values = heapq.nlargest(5, HandE_codex_similarity[i])
                dataf= pd.DataFrame(largest_values, columns=['similarity','name of the slide','patch coordination','Codex channel',
                                            'input slide','input patch coordination','H&E'])
                L_sim.append(dataf)

            
            concatenated_df = pd.concat(L_sim, ignore_index=True)

        

            L_df.append(concatenated_df)

    whole_df= pd.concat(L_df, ignore_index=True)

    return  whole_df



def get_subdictionary(dictionary, start, end):
    keys = list(dictionary.keys())[start:end]
    sub_dict = {key: dictionary[key] for key in keys}
    return sub_dict


def list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


class preprocessing_data():
      
      @staticmethod
      def specific_args(parent_parser):
           parser = ArgumentParser(parents=[parent_parser], add_help=False)

           parser.add_argument(
            "--patch_size",
            type = int,
            default = 64,
            help = "Height of the images. [default: 64]"
           )
           

           parser.add_argument(
            "--patching_seed",
            type = int,
            default = 64,
            help = "Height of the images. [default: 64]"
           )

           parser.add_argument(
            "--num_patches_per_image",
            type = int,
            default = 64,
            help = "Height of the images. [default: 64]"
           )

           parser.add_argument(
            "--transformations_read_dir",
            type = str,
            help = "Height of the images. [default: 64]"
           )

           parser.add_argument(
            "--whitespace_threshold",
            type = float,
            help = "Height of the images. [default: 64]"
           )


           return parser

            

           




      
      def __init__(self,patch_size,patching_seed,num_patches_per_image,transformations_read_dir,whitespace_threshold):
          
        #   self.patch_size=patch_size
        #   self.patching_seed=patching_seed
        #   self.num_patches_per_image=num_patches_per_image
        #   self.transformations_read_dir=transformations_read_dir
        #   self.whitespace_threshold=whitespace_threshold

          self.patch_size=64
          self.patching_seed=2
          self.num_patches_per_image=95
          self.transformations_read_dir='./logs/HandE_gray_colon_128_256/transformation'
          self.whitespace_threshold=0.82


      def _load_file(self, file):



        image = Image.open(file)


        image_array = np.array(image)

        return image_array
      
      """
      def _load_file(self, file):

        
           image = tif.imread(file)
           gr,r,g,b = cv.split(image[1,])[0]
           new_image = cv.merge((r,g,b))

           return new_image
      """

      def cropping(self,img,i,j):

        cropped_img= img[i: i+self.patch_size, j:j+self.patch_size]    
        cropped_img=cropped_img.astype(np.float32)
            
        Max=np.max(cropped_img)

        
        cropped_img=cropped_img/Max


        return cropped_img
      

      def _img_to_tensor(self, img):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        output= trans(img)


        gray_tf=transforms.Grayscale(num_output_channels=1)
        output=gray_tf(output)

    
        return output
      

      def overlap(self,i,j,coords):
    
        if len(coords) == 0:
            return True
        else:
            ml = set(map(lambda b: self.overlap_sample(b[0], b[1], i, j), coords))
            if False in ml:
                return False
            else: 
                return True
            
      def overlap_sample(self,a,b,i,j):
        
            if abs(i-a)>self.patch_size or abs(j-b)>self.patch_size :
                return True
        
            else:
                return False
        

      def _filter_whitespace(self, tensor_3d, threshold):
            
            r= np.mean(np.array(tensor_3d[0]))
            #g = np.mean(np.array(tensor_3d[1]))
            #b = np.mean(np.array(tensor_3d[2]))
            #channel_avg = np.mean(np.array([r, g, b]))
            if 0.6<r <0.8:
                return True
            else:
                return False
            
        

    


      def _patching(self,fname):
        
           transformations= load_transformation(path.join(self.transformations_read_dir, "trans.obj"))
           random.seed(self.patching_seed)
           
            
           img=self._load_file(fname)

           

           
           
           result=[]
           coords=[]
           count = 0
           start_time = datetime.now()
           spent_time = datetime.now() - start_time
        
           while count < self.num_patches_per_image and spent_time.total_seconds() < 10:
               
               rand_i = random.randint(0, img.shape[0] - self.patch_size)
               rand_j = random.randint(0, img.shape[1]- self.patch_size)
            
               cropped_img=self.cropping(img,rand_i,rand_j)

               
            
        
               
               output=self._img_to_tensor(cropped_img)

               #print(output)


            
               if self._filter_whitespace(output, threshold=self.whitespace_threshold):
                  if self.overlap(rand_i, rand_j, coords):

                    result.append(output)
                    output= transformations(output)
                
                    output=output.reshape(1,1,64,64)
                    coords.append([rand_i, rand_j,fname,output])
                    count += 1
                    print(count)
               spent_time = datetime.now() - start_time

               
           #print('""""""""""""""""""""""final""""""""""""' + str(count)) 
           return coords, result
      


