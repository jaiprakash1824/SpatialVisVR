import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches, text, patheffects
from modelclass import SegmentationModel
import numpy as np
from skimage import measure
import cv2


def find_dense_section_bounding_boxes(heatmap, threshold):
    # Apply thresholding to identify high-certainty pixels
    high_certainty_pixels = (heatmap >= threshold)

    # Use connected component analysis to find clusters of high-certainty pixels
    labels, num_features = measure.label(
        high_certainty_pixels, connectivity=2, return_num=True)

    # Calculate bounding boxes for each dense section
    bounding_boxes = []
    for i in range(1, num_features + 1):
        dense_section_indices = np.argwhere(labels == i)
        min_row, min_col = np.min(dense_section_indices, axis=0)
        max_row, max_col = np.max(dense_section_indices, axis=0)
        if (max_row - min_row) * (max_col - min_col) >= 2500:
            bounding_box = (min_row, min_col, max_row, max_col)
            bounding_boxes.append(bounding_box)

    return bounding_boxes


def is_pinkish(section):
    # Ensure the data type is np.uint8 and the values are in the expected range [0, 255]
    section = section.astype(np.uint8)

    # Convert the RGB values to LAB color space
    lab_pixels = cv2.cvtColor(section, cv2.COLOR_RGB2LAB)

    # Extract the 'a' channel (color) from the LAB color space
    l_channel = lab_pixels[:, :, 0]
    a_channel = lab_pixels[:, :, 1]
    b_channel = lab_pixels[:, :, 2]

    print("Min-Max L Channel:", np.min(l_channel), np.max(l_channel))
    print("Min-Max A Channel:", np.min(a_channel), np.max(a_channel))
    print("Min-Max B Channel:", np.min(b_channel), np.max(b_channel))

    #lower_l = 0
    #upper_l = 230

    lower_a = 120
    upper_a = 150

    lower_b = 130
    upper_b = 160

    # Create a mask to check if 'a' values fall within the pinkish range
    #l_mask = np.logical_and(l_channel >= lower_l, l_channel <= upper_l)
    a_mask = np.logical_and(a_channel >= lower_a, a_channel <= upper_a)
    b_mask = np.logical_and(b_channel >= lower_b, b_channel <= upper_b)

    lab_mask = np.logical_and(a_mask, b_mask)

    #lab_mask = np.logical_and(np.logical_and(l_mask, a_mask), b_mask)

    return lab_mask


def infer(frame, count=0):
    model = SegmentationModel(2)


    # TODO added here
    
    model.load_state_dict(torch.load('best.pth',map_location=torch.device("cpu")))
    #model = torch.load('backbone.pth')
    
    model.eval()
    
    # print(input_image)

    # Define preprocessing transformations
    preprocess = transforms.Compose([
    # Resize to match the model's input size
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.Pad(30),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),           # Convert to tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

    ])
    postprocess = transforms.Compose([
            # Resize to match the model's input size
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.Pad(30),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),           # Convert to tensor
    ])
        
        # Apply preprocessing to the input image
    input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension


    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    # print(output)

    p = 32

    input_tensor = transforms.functional.crop(input_tensor,p, p, 640-p, 640-p)

    #input_tensor = postprocess(input_tensor)
    print("hello")
    cropped_output = output[0][0][p:-p, p:-p]

    threshold = 230

    boxes = find_dense_section_bounding_boxes(
        cropped_output, threshold=threshold)

    image_array = np.array(frame)
    print("hello2")
    # If the image is RGBA (contains an alpha channel), you can convert it to RGB
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    pinkish_bounding_boxes = []  # Create a list to store pinkish bounding boxes

    # # Iterate through bounding boxes and check for pinkish pixels
    # for box in boxes:
    #     min_row, min_col, max_row, max_col = box
    #     # Extract the section within the bounding box
    #     section = image_array[min_row:max_row, min_col:max_col]
    #     is_pinkish_mask = is_pinkish(section)
    #     pinkish_pixel_count = np.sum(is_pinkish_mask)
    #     total_pixels = is_pinkish_mask.size
    #     pinkish_pixel_ratio = pinkish_pixel_count / total_pixels

    #     # Check if the pinkish pixel ratio is above a threshold (e.g., 0.5 for 50%)
    #     print(pinkish_pixel_ratio)
    #     if pinkish_pixel_ratio > 0.2:
    #         pinkish_bounding_boxes.append(box)



    print("hello3", frame.shape)
    output_image = Image.new("RGB", (frame.shape[0], frame.shape[1]))

    image_array = transforms.functional.crop(postprocess(frame).unsqueeze(0),p, p, 640-p, 640-p)[0].permute(1, 2, 0).cpu().numpy()  # Channels last

    # Convert the NumPy array to a PIL image
    #cropped_pil = Image.fromarray((image_array * 255).astype('uint8'))
    print("hello4")
    resolution_ratio_x = (640/576)*(int(frame.shape[0])/640)
    resolution_ratio_y = (640/576)*(int(frame.shape[1])/640)

    print("hello5")

    for box in boxes:
        print(type(frame))
        # section_image=frame[int(box[0]*resolution_ratio_y):int(box[3]*resolution_ratio_x), int(box[0]*resolution_ratio_y):int(box[2]*resolution_ratio_y)]
        # print(section_image.shape)
        section_image = Image.fromarray(frame).crop((int(box[1]*resolution_ratio_x), int(box[0]*resolution_ratio_y), int(box[3]*resolution_ratio_x), int(box[2]*resolution_ratio_y)))
        output_image.paste(section_image, (int(box[1]*resolution_ratio_x), int(box[0]*resolution_ratio_y), int(box[3]*resolution_ratio_x), int(box[2]*resolution_ratio_y)))
        
    print("output image")
    output_image.show()
    output_image.save('./cropped_1/cropped_'+str(count)+".png")


    # plt.imshow(input_tensor[0].permute(1, 2, 0),cmap='gray', interpolation='none')
    # plt.imshow(cropped_output, cmap='jet', alpha=0.5, interpolation='none')
    # for box in boxes:
    #     plt.gca().add_patch(patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], fill=False, edgecolor='red', lw=2))
    # for box in pinkish_bounding_boxes:
    #     plt.gca().add_patch(patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], fill=False, edgecolor='blue', lw=2))
    # plt.colorbar()
    # #plt.savefig('./results/heatmap/heatmap_'+str(count)+'.png')
    # plt.show()
    # plt.close()


"""
    output_image = Image.new("RGB", (cropped_output.shape[0], cropped_output.shape[1]))

    image_array = input_tensor[0].permute(1, 2, 0).cpu().numpy()  # Channels last

    # Convert the NumPy array to a PIL image
    cropped_pil = Image.fromarray((image_array * 255).astype('uint8'))

    for rect in pinkish_bounding_boxes:
        section_image = cropped_pil.crop(rect)
        output_image.paste(section_image, rect[:2])
    
    output_image.show()
"""


#input_image = Image.open('IMG_5889.jpg')
# infer(input_image)
