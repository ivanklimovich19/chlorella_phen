#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


# In[ ]:


def setup_folders(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    hist_dir = os.path.join(output_folder, "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    return hist_dir  


# In[ ]:


def get_image_files(folder_path):
    return [f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]


# In[ ]:


def sample_pixels(img_array, step=5):
    return img_array[::step, ::step]


# In[ ]:


def analyze_colors(image_rgb, step=5):
    sampled_img = sample_pixels(image_rgb, step)
    r, g, b = cv2.split(sampled_img)
    return r, g, b


# In[ ]:


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast_enhanced, (5,5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# In[ ]:


def segment_cells(image, binary_image):
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), markers)
    return markers


# In[ ]:


def count_and_filter(image, markers, min_size=30, max_size=1000):
    counts = 0
    result_image = image.copy()
    areas = []
    
    for label in np.unique(markers):
        if label in [0, -1]: 
            continue
            
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == label] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
            if min_size <= area <= max_size:
                counts += 1
                cv2.drawContours(result_image, [cnt], -1, (0,255,0), 2)
    
    return counts, areas


# In[ ]:


def detect_background(image, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15,15), 0)
    _, bg_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    return bg_mask


# In[ ]:


def process_single_image(image_path, step=5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить: {image_path}")
        return None, None, None, None, None
    
    filename = os.path.basename(image_path)
    
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = analyze_colors(img_rgb, step)
        
        binary = preprocess_image(img)
        markers = segment_cells(img, binary)
        cell_count, areas = count_and_filter(img, markers)  # Получаем только два значения
        
        color_stats = {
            'filename': filename,
            'Red': np.mean(r),
            'Green': np.mean(g),
            'Blue': np.mean(b),
            'R/G': np.mean(r) / (np.mean(g) + 0.001),
            'R/B': np.mean(r) / (np.mean(b) + 0.001),
            'G/B': np.mean(g) / (np.mean(b) + 0.001),
            'pixels_sampled': r.size
        }
        
        cell_stats = {
            'Cell Count': cell_count,
            'Mean Area': np.mean(areas) if areas else 0,
            'Median Area': np.median(areas) if areas else 0,
            'Min Area': min(areas) if areas else 0,
            'Max Area': max(areas) if areas else 0,
            'Areas': areas
        }
        
        return color_stats, cell_stats, r, g, b
        
    except Exception as e:
        print(f"Ошибка при обработке {filename}: {str(e)}")
        return None, None, None, None, None


# In[ ]:


def save_histograms(all_r, all_g, all_b, hist_dir):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.hist(all_r, bins=256, color='red', alpha=0.7, range=(0, 256))
    plt.title('R-channel')
    plt.xlabel('Intensity')
    plt.ylabel('Pixels number')
    
    plt.subplot(132)
    plt.hist(all_g, bins=256, color='green', alpha=0.7, range=(0, 256))
    plt.title('G-channel')
    plt.xlabel('Intensity')
    plt.ylabel('Pixels number')
    
    plt.subplot(133)
    plt.hist(all_b, bins=256, color='blue', alpha=0.7, range=(0, 256))
    plt.title('B-сhannel')
    plt.xlabel('Intensity')
    plt.ylabel('Pixels number')
    
    plt.subplots_adjust(wspace=0.5)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = os.path.join(hist_dir, f'intensity_distribution_{timestamp}.png')
    plt.savefig(hist_path)
    plt.close()
    return hist_path


# In[ ]:


def main():
    input_folder = input("Enter the name of the input data folder: ")
    output_folder = input("Enter a folder name to save the results: ")
    pixel_sample_step = 5
    hist_dir = setup_folders(output_folder)

    image_files = get_image_files(input_folder)
    print(f"Files found in folder: {len(image_files)}")
    
    all_color_stats = []
    all_cell_stats = []
    all_r, all_g, all_b = [], [], []
    
    for filename in tqdm(image_files, desc="Image processing"):
        image_path = os.path.join(input_folder, filename)
        color_stats, cell_stats, r, g, b = process_single_image(image_path, pixel_sample_step) 
        
        if color_stats and cell_stats:
            all_color_stats.append(color_stats)
            all_cell_stats.append({
                'Filename': color_stats['filename'],
                **{k: v for k, v in cell_stats.items() if k != 'Areas'}
            })
            
            if r is not None and g is not None and b is not None:
                all_r.extend(r.ravel())
                all_g.extend(g.ravel())
                all_b.extend(b.ravel())

            print(f"Processed: {filename}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
    if all_color_stats:
        color_df = pd.DataFrame(all_color_stats)
        color_csv_path = os.path.join(output_folder, f'color_stats_{timestamp}.csv')
        color_df.to_csv(color_csv_path, index=False)
        print(f"\nRGB statistics are saved in: {color_csv_path}")
    
    if all_cell_stats:
        cell_df = pd.DataFrame(all_cell_stats)
        cell_csv_path = os.path.join(output_folder, f'cell_stats_{timestamp}.csv')
        cell_df.to_csv(cell_csv_path, index=False)
        print(f"Cell statistics are saved in:\n- {cell_csv_path}")
    
    if all_r and all_g and all_b:
        hist_path = save_histograms(all_r, all_g, all_b, hist_dir)
        print(f"\nHistograms are saved in: {hist_path}")

if __name__ == "__main__":
    main()


# In[ ]:


get_ipython().system('jupyter nbconvert --to python имя_вашего_файла.ipynb')

