import os 
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from typing import List, Tuple, Union
from scipy import ndimage
from scipy.ndimage import binary_dilation

data_path = "/Users/aaryaamoharir/Desktop/Summer 2025 /Research /corruptionML/CIFAR-10-C"
output_dir = "/Users/aaryaamoharir/Desktop/Summer 2025 /Research /corruptionML/CIFAR-10-C/transformed"
store_dir = "/Users/aaryaamoharir/Desktop/Summer 2025 /Research /corruptionML/CIFAR-10-C/store"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(store_dir, exist_ok=True)

def load_data_npy(data_path):
    print("hi")
    npy_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    for i, file in enumerate(npy_files):
        print(f"{i+1}: {file}")
    
    severity_indices = [0,1001,2002,3003,4004, 10000, 10001, 12002, 13003, 14004, 15005, 20000,22002, 23003, 24004, 25005, 30000, 40000] 
    severity_labels = [1, 2, 3, 4, 5] # For naming

    extracted_count = 0
    for file_path in npy_files:
        try:
            corruption_data = np.load(file_path) # Expected shape (50000, 32, 32, 3)
            corruption_name = os.path.splitext(os.path.basename(file_path))[0]
            
            print(f"\n--- Processing corruption type: '{corruption_name}' ---")
            
            # Verify the expected shape
            if corruption_data.shape != (50000, 32, 32, 3):
                print(f"  Warning: Unexpected shape for {corruption_name}.npy: {corruption_data.shape}. Skipping.")
                continue

            for s_idx, severity_level in zip(severity_indices, severity_labels):
                img_array = corruption_data[s_idx] # Get the image at the specific severity index

                # Standard CIFAR-10-C images are already (H, W, C) and uint8
                # So, no complex transposing or dtype conversion should be needed for standard data
                # Adding a small sanity check just in case, but generally won't trigger for CIFAR-10-C
                if img_array.ndim == 2: # grayscale
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[0] in [1, 3] and img_array.ndim == 3 and img_array.shape[1] == 32: # potential C,H,W
                    img_array = np.transpose(img_array, (1, 2, 0)) # Transpose to H,W,C

                if img_array.dtype != np.uint8:
                     img_array = img_array.astype(np.uint8)

                pil_img = Image.fromarray(img_array)
# Construct filename: corruption_type_severityX_indexY_labelZ.png
                image_filename = os.path.join(store_dir, 
                                              f"{corruption_name}_severity{severity_level}_idx{s_idx}.png")
                pil_img.save(image_filename)
                print(f"  Saved: {os.path.basename(image_filename)}")
                extracted_count += 1

        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")

    print(f"\nSuccessfully extracted {extracted_count} images into '{output_dir}'.")
    return extracted_count

def load_data(data_path):
    image_paths = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.jpeg'):
                image_paths.append(os.path.join(root, file))
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")  # Convert to RGB in case some are grayscale or CMYK
            images.append((img, path)) 
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
    print(f"Loaded {len(images)} images.")
    
    return images


def apply_all_transformations(images):
    
    # 2d transformations with their bounds 
    transformations_2d = {
        'scale': {'min': 0.9, 'max': 1.4, 'step': 0.1},
        'rotation': {'min': -22.5, 'max': 22.5, 'step': 2.5},
        'lighten_darken': {'min': -0.05, 'max': 0.05, 'step': 0.01},
        'gaussian_noise': {'min': 0.0, 'max': 0.1, 'step': 0.01},
        'translation': {'min': -50, 'max': 50, 'step': 5},  # pixels
        'contrast': {'min': 0, 'max': 1, 'step': 0.1}, 
        'blur': {'min': 0, 'max': 5, 'step': 0.5},  
        'shear': {'min': 0, 'max': 1, 'step': 0.1}
      
    }
    
    # combine them all 
    transformations = {**transformations_2d}
    
    transformed_images = []
    total_transforms = 0
    
    for i, (img, path) in enumerate(images):
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        ext = '.jpg'
        
        # apply each transformation to each image (amount of transformation is random)
        for transform_type in transformations.keys():
            params = transformations[transform_type]
            
            # generate random value within the range 
            if transform_type == 'translation' or transform_type == 'xy_translation_3d':
                # For translation transforms, we need x and y values
                num_steps = int((params['max'] - params['min']) / params['step']) + 1
                possible_values = [params['min'] + j * params['step'] for j in range(num_steps)]
                tx = random.choice(possible_values)
                ty = random.choice(possible_values)
                new_filename = f"{name}_{transform_type}_{tx}_{ty}_corrupted{ext}"
            elif transform_type == 'background':
                # For background, choose from predefined RGB values
                bg_color = random.choice(params['values'])
                new_filename = f"{name}_{transform_type}_{bg_color[0]}_{bg_color[1]}_{bg_color[2]}_corrupted{ext}"
            else:
                # create discrete values 
                num_steps = int((params['max'] - params['min']) / params['step']) + 1
                possible_values = [params['min'] + j * params['step'] for j in range(num_steps)]
                transform_value = random.choice(possible_values)
                new_filename = f"{name}_{transform_type}_{transform_value}_corrupted{ext}"
            
            # apply transformation 
            if transform_type == 'scale':
                transformed_img = apply_scale(img, transform_value)
            elif transform_type == 'rotation':
                transformed_img = apply_rotation(img, transform_value)
            elif transform_type == 'lighten_darken':
                transformed_img = apply_brightness(img, transform_value)
            elif transform_type == 'gaussian_noise':
                transformed_img = apply_gaussian_noise(img, transform_value)
            elif transform_type == 'translation':
                transformed_img = apply_translation(img, tx, ty)
            elif transform_type == 'contrast':
                transformed_img = apply_contrast(img, transform_value)
            elif transform_type == 'shear':
                transformed_img = apply_shear(img, transform_value)
            elif transform_type == 'blur':
                transformed_img = apply_blur(img, transform_value)
            
            
            # Save the transformed image
            save_path = os.path.join(output_dir, new_filename)
            transformed_img.save(save_path)
            transformed_images.append(transformed_img)
            total_transforms += 1
        
        # progress each 1000 images to debug 
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(images)} original images, created {total_transforms} transformed images")
    
    return transformed_images

#apply scaling to the images (zoom in and zoom out)
def apply_scale(img: Image.Image, scale_factor: float) -> Image.Image:
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # resize the image 
    scaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # depending on if the image is scaled up or down, crop the images 
    if scale_factor > 1.0:
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height
        scaled = scaled.crop((left, top, right, bottom))
    elif scale_factor < 1.0:
        # if the image is scaled down, we need to pad it to the original size
        result = Image.new('RGB', (width, height), (0, 0, 0))
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        result.paste(scaled, (paste_x, paste_y))
        scaled = result
    
    return scaled

def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    #using the ImageCV library to do rotations 
    rotated = img.rotate(-angle, fillcolor=(0, 0, 0), expand=False)
    return rotated

def apply_contrast(img: Image.Image, contrast_amount: float) -> Image.Image:
    img_np = np.array(img)
    if img_np.shape[2] == 4: # Check if it's RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    adjusted_image = cv2.convertScaleAbs(img_np, alpha=contrast_amount, beta=0)
    #convert back to PIL Image
    adjusted_image_pil = Image.fromarray(adjusted_image)
    return adjusted_image_pil

def apply_shear(img: Image.Image, shear_factor: float) -> Image.Image:
    width, height = img.size
    #new width and height after shear
    shift_in_pixels = int(math.ceil(shear_factor * height))
    new_width = width + shift_in_pixels
    shear_image = img.transform(
        (new_width, height),  #new image size 
        Image.AFFINE,         # affine transformation since shear is a linear transformation
        (1, shear_factor, -shift_in_pixels if shear_factor > 0 else 0,  #this is the transformation matrix
         0, 1, 0),
        resample=Image.BICUBIC,
        fillcolor=(255, 255, 255)  # or any background color you prefer
    )

    return shear_image

def apply_blur(img: Image.Image, blur_radius: float) -> Image.Image:
    img_np = np.array(img)

    #convert to bgr array if needed (apparently most PIL images are RGB)
    if img_np.ndim == 3 and img_np.shape[2] == 3: # RGB
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif img_np.ndim == 3 and img_np.shape[2] == 4: # RGBA 
        img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else: 
        img_np_bgr = img_np.copy()

    ksize = int(blur_radius * 6)
    if ksize % 2 == 0:
        ksize += 1
    # minimum kernal size 
    if ksize < 3 and blur_radius > 0:
        ksize = 3
    elif blur_radius == 0: #no blur 
        return img 
    
    # apply gaussian blur
    blurred_image_np_bgr = cv2.GaussianBlur(img_np_bgr, (ksize, ksize), blur_radius)

    if img_np.ndim == 3 and (img_np.shape[2] == 3 or img_np.shape[2] == 4):
        blurred_image_np_rgb = cv2.cvtColor(blurred_image_np_bgr, cv2.COLOR_BGR2RGB)
    else: 
        blurred_image_np_rgb = blurred_image_np_bgr
    blurred_image_pil = Image.fromarray(blurred_image_np_rgb)

    return blurred_image_pil
    

#brighten or darken the image 
def apply_brightness(img: Image.Image, brightness_factor: float) -> Image.Image:
    # brightness_factor ranges from -0.05 to 0.05
    # brightness enhancer: factor < 1.0 darkens, factor > 1.0 lightens
    enhancement_factor = 1.0 + brightness_factor
    
    enhancer = ImageEnhance.Brightness(img)
    adjusted = enhancer.enhance(enhancement_factor)
    
    return adjusted

#apply gaussian noise to the image
def apply_gaussian_noise(img: Image.Image, noise_std: float) -> Image.Image:
    img_array = np.array(img)
    noise = np.random.normal(0, noise_std * 255, img_array.shape).astype(np.float32)
    noisy = img_array.astype(np.float32) + noise
    
    # convert the image back to uint8 and clip values to valid range
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # convert back to PIL Image
    return Image.fromarray(noisy)

#apply translation to the image by tx and ty pixels
def apply_translation(img: Image.Image, tx: float, ty: float) -> Image.Image:
    width, height = img.size
    
    result = Image.new('RGB', (width, height), (0, 0, 0))
    paste_x = int(tx)
    paste_y = int(ty)
    
    # handle negative translations 
    crop_left = max(0, -paste_x)
    crop_top = max(0, -paste_y)
    crop_right = min(width, width - paste_x)
    crop_bottom = min(height, height - paste_y)
    
    # crop the image if the translation would result in an empty area
    if crop_left < crop_right and crop_top < crop_bottom:
        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # calculate where to paste the cropped image in the result
        result_paste_x = max(0, paste_x)
        result_paste_y = max(0, paste_y)
        
        result.paste(cropped, (result_paste_x, result_paste_y))
    
    return result
    
def apply_camera_distance(img: Image.Image, distance_factor: float) -> Image.Image:
    #3D transformation (not needed)
    neutral_distance = 2.75
    scale_factor = neutral_distance / distance_factor
    
    return apply_scale(img, scale_factor)

def apply_xy_translation_3d(img: Image.Image, tx: float, ty: float) -> Image.Image:
    width, height = img.size
    pixel_tx = int(tx * width)
    pixel_ty = int(ty * height)
    
    return apply_translation(img, pixel_tx, pixel_ty)

#3d rotation (not needed)
def apply_rotation_3d(img: Image.Image, angle: float) -> Image.Image:
    return apply_rotation(img, angle)

# Apply background change to the image (not needed)
def apply_background_change(img: Image.Image, bg_color: Tuple[float, float, float]) -> Image.Image:
    #converts image for transparency 
    img_rgba = img.convert('RGBA')
    
    # create a background image with the color provided 
    bg_rgb = tuple(int(c * 255) for c in bg_color)
    background = Image.new('RGB', img.size, bg_rgb)
    img_rgb = img.convert('RGB')
    gray = img_rgb.convert('L')
    img_array = np.array(gray)
    
    edges = ndimage.sobel(img_array)
    edge_mask = edges > np.percentile(edges, 70) 
    foreground_mask = binary_dilation(edge_mask, iterations=3)
    mask = Image.fromarray((foreground_mask * 255).astype(np.uint8))
    
    result = Image.composite(img_rgb, background, mask)
    return result

# 3d transformation that we're not using 
def apply_background_change_simple(img: Image.Image, bg_color: Tuple[float, float, float]) -> Image.Image:
    # Convert color from 0-1 range to 0-255 range
    bg_rgb = tuple(int(c * 255) for c in bg_color)
    background = Image.new('RGB', img.size, bg_rgb)
    
    # Simple blend - 70% original image, 30% background
    return Image.blend(img.convert('RGB'), background, 0.3)

if __name__ == "__main__":
    images = load_data_npy(data_path)
    #images = images[:4]  # limit to first 4 images to test
    #print(f"Loaded {len(images)} images.")
    #print(f"Will create {len(images) * 5} transformed images (5 transformations per original image)")
    #transformed_images = apply_all_transformations(images)
    #print(f"Successfully transformed and saved {len(transformed_images)} images.")