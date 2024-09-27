import os
import cv2
import numpy as np
import json
from tqdm import tqdm

# Replace these paths with your actual paths
input_dir = "C:\\Users\\LENOVO\\Videos\\data\\sam_data\\datasetlane3.v1i.coco-segmentation\\train"  # Replace with the path to your dataset
output_dir = "C:\\Users\\LENOVO\\Videos\\data\\sam_data\\new_data"  # Replace with the path to save the instance maps
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
with open(os.path.join(input_dir, "C:\\Users\\LENOVO\\Videos\\data\\sam_data\\datasetlane3.v1i.coco-segmentation\\train\\_annotations.coco.json")) as f:  # Replace 'annotations.json' if your annotation file has a different name
    coco_data = json.load(f)

# Create a dictionary to map image ID to file name
image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

# Category mapping for your specific classes
category_map = {
    1: "line_1",
    2: "line_2",
    3: "line_3"
}

# Create empty maps for each image
for img_id, img_file in tqdm(image_id_to_filename.items()):
    # Load the corresponding image to get dimensions
    image_path = os.path.join(input_dir, "C:\\Users\\LENOVO\\Videos\\data\\sam_data\\datasetlane3.v1i.coco-segmentation\\train", img_file)  # Replace 'images' with the folder containing your images
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot open image {image_path}")
        continue

    height, width, _ = image.shape

    # Initialize an empty instance map with 3 channels
    instance_map = np.zeros((height, width, 3), dtype=np.uint8)

    # Process annotations for the current image
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            segmentation = ann['segmentation']
            category_id = ann['category_id']
            mask = np.zeros((height, width), dtype=np.uint8)

            # Create a binary mask from segmentation points
            for seg in segmentation:
                points = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)

            # Assign mask to the correct channel based on the category_id
            if category_id == 1:  # ung_thu_hac_to
                instance_map[:, :, 0][mask > 0] = ann['id']
            elif category_id == 2:  # ung_thu_te_bao_day
                instance_map[:, :, 1][mask > 0] = ann['id']
            elif category_id == 3:  # ung_thu_te_bao_vay
                instance_map[:, :, 2][mask > 0] = ann['id']

    # Save the instance map
    output_file = os.path.join(output_dir, img_file.replace('.jpg', '.png'))  # Replace '.jpg' and '.png' with your image formats if different
    cv2.imwrite(output_file, instance_map)

print("Conversion complete!")
