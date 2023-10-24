from PIL import Image

def tif_to_png(input_file, output_file):
    with Image.open(input_file) as img:
        # Ensure the image is loaded
        img.load()
        # Get the first frame/page from the TIF
        img.seek(0)

        if img.mode == "I;16B":
            img = img.convert("L")

        img.save(output_file, 'PNG')

input_path = "/Volumes/Jai_Lab_T7/Data/Codex/CRC_TMA_A_hyperstacks_bestFocus/bestFocus/reg002_X01_Y01_Z10.tif"
output_path = "./out.png"
tif_to_png(input_path, output_path)

import tifffile as tf
from PIL import Image
import numpy as np

def tif_to_png(input_file, output_file):
    with tf.imread(input_file) as tif_data:
        # tifffile returns numpy arrays. Convert the first page to a PIL Image
        img = Image.fromarray(tif_data[0])
        
        # If the image is a 16-bit grayscale, convert it to 8-bit
        if img.mode == "I;16B":
            img = img.point(lambda x: x * (1./256.)).convert("L")
        
        # Save as PNG
        img.save(output_file, 'PNG')

input_path = "/Volumes/Jai_Lab_T7/Data/Codex/CRC_TMA_A_hyperstacks_bestFocus/bestFocus/reg002_X01_Y01_Z10.tif"
output_path = "./out.png"
tif_to_png(input_path, output_path)

