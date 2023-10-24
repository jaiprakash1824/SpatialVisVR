import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from modelclass import SegmentationModel
import os
import numpy as np
from inference import infer


folder_path = './results/cropped'
i = 0
# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename.endswith('.png'):
        # Load and preprocess the input image (replace 'input_image.jpg' with the actual path)
        input_image_path = folder_path+'/'+filename
        input_image = Image.open(input_image_path)

        infer(input_image, i)
        i+=1


"""
        #mask  =np.ma.masked_where((0 < output[0][0]) & (output[0][0] < 50), output[0][0])
        #print(mask[422])
        plt.imshow(input_image, cmap='gray',interpolation='none')
        plt.imshow(output[0][0], cmap='jet', alpha=0.5,interpolation='none')  
        plt.colorbar()  
        plt.show()

        



        threshold_value = 60  # Adjust this value as needed
        heatmap_mask = (np.array(output[0][0]) < threshold_value).astype(np.uint8) * 255
        heatmap_mask = Image.fromarray(heatmap_mask)

        extracted_image = Image.new('RGB', input_image.size)
        extracted_image.paste(input_image, mask=heatmap_mask)



        #mask  =np.ma.masked_where((0 < output[0][0]) & (output[0][0] < 50), output[0][0])
        #print(mask[422])
        #extracted_image.show()
        plt.imshow(extracted_image)  
        plt.show()

"""
