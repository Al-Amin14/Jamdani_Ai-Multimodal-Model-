import os
import cv2
import pandas as pd
import albumentations as A

# 1. Define paths (Update these to match your local directories)
input_dir = "images"
output_dir = "augmented_images"
os.makedirs(output_dir, exist_ok=True)

# 2. Load the original dataset
df = pd.read_excel("Jamdani_Samples.xlsx")

# 3. Define the 4 specific Albumentations pipelines
augmentations = {
    "fog": A.RandomFog(fog_coef_lower=0.6, fog_coef_upper=0.9, alpha_coef=0.065, always_apply=True),
    "flip": A.HorizontalFlip(always_apply=True),
    "bright": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.0, always_apply=True),
    "rot": A.Rotate(limit=30, always_apply=True)
}

# 4. Process and save physical images
for index, row in df.iterrows():
    original_img_name = row['image_name']
    name_part, ext_part = original_img_name.split('.')
    img_path = os.path.join(input_dir, original_img_name)
    
    # Read image via OpenCV
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: {original_img_name} not found. Skipping...")
        continue
        
    # Convert BGR to RGB for Albumentations compatibility
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Copy original to output directory (optional)
    cv2.imwrite(os.path.join(output_dir, original_img_name), image)
    
    # Apply and save each augmentation
    for aug_name, transform in augmentations.items():
        # Apply transformation
        augmented = transform(image=image_rgb)['image']
        
        # Convert back to BGR for OpenCV saving
        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        
        # Save with the exact suffix mapping generated in the new Excel file
        new_filename = f"{name_part}_{aug_name}.{ext_part}"
        save_path = os.path.join(output_dir, new_filename)
        cv2.imwrite(save_path, augmented_bgr)

print(f"Offline augmentation complete. Images saved to: {output_dir}")