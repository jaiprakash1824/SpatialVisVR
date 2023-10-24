import cv2
from inference_obj import infer
import keyboard
from inference import preprocess
import torchvision.transforms as transforms 

frametransform = transforms.ToPILImage()

try:
    # Initialize video capture
    Capture = cv2.VideoCapture('http://10.182.49.100:4747/video.mov')
    if not Capture.isOpened():
        print("Error: Could not open video stream.")
        exit(1)

    while Capture.isOpened():
        # Capture frame-by-frame

        ret, frame = Capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        if(keyboard.is_pressed('esc')):

            infer(frametransform(frame), 0)
            break

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    image_file_path = './cropped_1/cropped_0.png'
    print('begin')
    preprocess(image_file_path)


     
except Exception as e:
    print(f"An exception occurred: {e}")

