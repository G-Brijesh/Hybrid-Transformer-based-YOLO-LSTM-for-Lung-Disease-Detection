import cv2
import os
import numpy as np


dataset_path = r"C:\Users\brijesh\OneDrive\Desktop\LUNG PROJECT\filtered"
output_path = r"C:\Users\brijesh\OneDrive\Desktop\LUNG PROJECT\dataset"


os.makedirs(output_path, exist_ok=True)

for img_name in os.listdir(dataset_path):
    if img_name.endswith(('.png', '.jpg', '.jpeg')): 
        img_path = os.path.join(dataset_path, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  

        if image is None:
            print(f"Skipping {img_name} (Error loading image)")
            continue

        
        median_filtered = cv2.medianBlur(image, 5)

     
        lab = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

       
        resized_image = cv2.resize(contrast_enhanced, (640, 640))

       
        if len(resized_image.shape) == 2:  
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

        
        save_path = os.path.join(output_path, img_name)
        cv2.imwrite(save_path, resized_image)  
        print(f"Processed & Saved: {save_path}")

print("\nPreprocessing Completed! All images saved in:", output_path)
