import os
import gdown
import zipfile
import cv2
import numpy as np
import pandas as pd
import re
from glob import glob

# --- Colab Specific Setup (Run these only once in a Colab notebook cell) ---
# !pip install gdown

# --- Configuration for Images ---
GDRIVE_FILE_ID_MAIN_DATASET = "1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2" # Original Total-Text dataset ZIP
DOWNLOAD_PATH_MAIN_DATASET_ZIP = "/content/total_text_full_dataset.zip"
IMAGES_BASE_EXTRACT_DIR = "/content/TotalText_Images_Extracted"
TOTAL_TEXT_IMAGE_ROOT = "" # Will be set after extraction

print("--- Starting Image Dataset Download and Extraction ---")

# 1. Download the main dataset ZIP
print(f"Downloading main Total-Text Dataset (ID: {GDRIVE_FILE_ID_MAIN_DATASET}) to {DOWNLOAD_PATH_MAIN_DATASET_ZIP}...")
if not os.path.exists(DOWNLOAD_PATH_MAIN_DATASET_ZIP):
    gdown.download(id=GDRIVE_FILE_ID_MAIN_DATASET, output=DOWNLOAD_PATH_MAIN_DATASET_ZIP, quiet=False)
    print("Main dataset ZIP download complete!")
else:
    print("Main dataset ZIP already exists, skipping download.")

# 2. Extract the main dataset ZIP
print(f"Extracting {DOWNLOAD_PATH_MAIN_DATASET_ZIP} to {IMAGES_BASE_EXTRACT_DIR}...")
if not os.path.exists(IMAGES_BASE_EXTRACT_DIR):
    os.makedirs(IMAGES_BASE_EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(DOWNLOAD_PATH_MAIN_DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(IMAGES_BASE_EXTRACT_DIR)
    print("Main dataset extraction complete!")

    # Determine the actual image root directory within the extracted content
    # The original zip extracts to a folder named 'Total-Text-Dataset-master' inside IMAGES_BASE_EXTRACT_DIR
    # We need to find the directory that contains 'Dataset'
    potential_roots = [os.path.join(IMAGES_BASE_EXTRACT_DIR, d) for d in os.listdir(IMAGES_BASE_EXTRACT_DIR) if os.path.isdir(os.path.join(IMAGES_BASE_EXTRACT_DIR, d))]
    
    found_root = False
    for p_root in potential_roots:
        if os.path.exists(os.path.join(p_root, 'Dataset')):
            TOTAL_TEXT_IMAGE_ROOT = p_root
            found_root = True
            break
    
    if not found_root:
        # Fallback if the expected structure isn't found immediately
        print(f"Warning: Expected a 'Dataset' folder within a top-level folder inside {IMAGES_BASE_EXTRACT_DIR}. Falling back to base extract dir.")
        TOTAL_TEXT_IMAGE_ROOT = IMAGES_BASE_EXTRACT_DIR # Fallback, might need manual adjustment
else:
    print("Image dataset already extracted, skipping extraction.")
    # If already extracted, re-determine the structure
    potential_roots = [os.path.join(IMAGES_BASE_EXTRACT_DIR, d) for d in os.listdir(IMAGES_BASE_EXTRACT_DIR) if os.path.isdir(os.path.join(IMAGES_BASE_EXTRACT_DIR, d))]
    found_root = False
    for p_root in potential_roots:
        if os.path.exists(os.path.join(p_root, 'Dataset')):
            TOTAL_TEXT_IMAGE_ROOT = p_root
            found_root = True
            break
    if not found_root:
        TOTAL_TEXT_IMAGE_ROOT = IMAGES_BASE_EXTRACT_DIR # Fallback

# Ensure TOTAL_TEXT_IMAGE_ROOT ends up pointing to the 'Dataset' directory
# Assuming Total-Text-Dataset-master/Dataset/Train and Test
if os.path.exists(os.path.join(TOTAL_TEXT_IMAGE_ROOT, 'Dataset')):
    TOTAL_TEXT_IMAGE_ROOT = os.path.join(TOTAL_TEXT_IMAGE_ROOT, 'Dataset')
elif os.path.exists(os.path.join(IMAGES_BASE_EXTRACT_DIR, 'Dataset')):
     TOTAL_TEXT_IMAGE_ROOT = os.path.join(IMAGES_BASE_EXTRACT_DIR, 'Dataset')
else:
    print("Error: Could not find 'Dataset' directory within the extracted image files.")
    # You might want to exit or raise an error here if the path is critical.

print(f"Image dataset root set to: {TOTAL_TEXT_IMAGE_ROOT}")
print(f"Contents of image root ({TOTAL_TEXT_IMAGE_ROOT}): {os.listdir(TOTAL_TEXT_IMAGE_ROOT) if os.path.exists(TOTAL_TEXT_IMAGE_ROOT) else 'Path not found'}")
# Example: Check for 'Train' and 'Test' folders
if os.path.exists(os.path.join(TOTAL_TEXT_IMAGE_ROOT, 'Train')):
    print(f"Contents of {os.path.join(TOTAL_TEXT_IMAGE_ROOT, 'Train')}: {os.listdir(os.path.join(TOTAL_TEXT_IMAGE_ROOT, 'Train'))[:5]}...")


# --- Configuration for Ground Truth ---
GDRIVE_FILE_ID_ADDITIONAL_JSON = "1v-pd-74EkZ3dWe6k0qppRtetjdPQ3ms1"  # New Google Drive ID
DOWNLOAD_PATH_ADDITIONAL_JSON_ZIP = "/content/totaltext_groundtruth_new.zip"
GROUNDTRUTH_EXTRACT_DIR = "/content/TotalText_Groundtruth_New_Extracted"
TOTAL_TEXT_GT_ROOT = GROUNDTRUTH_EXTRACT_DIR # Will be updated after extraction

print("\n--- Starting Ground Truth Download and Extraction ---")

# 1. Download the additional JSONs ZIP
print(
    f"Downloading new JSON ground truth (ID: {GDRIVE_FILE_ID_ADDITIONAL_JSON}) to {DOWNLOAD_PATH_ADDITIONAL_JSON_ZIP}..."
)
if not os.path.exists(DOWNLOAD_PATH_ADDITIONAL_JSON_ZIP):
    gdown.download(
        id=GDRIVE_FILE_ID_ADDITIONAL_JSON,
        output=DOWNLOAD_PATH_ADDITIONAL_JSON_ZIP,
        quiet=False,
    )
    print("New JSON ground truth ZIP download complete!")
else:
    print("New JSON ground truth ZIP already exists, skipping download.")

# 2. Extract the additional JSONs ZIP
print(
    f"Extracting {DOWNLOAD_PATH_ADDITIONAL_JSON_ZIP} to {GROUNDTRUTH_EXTRACT_DIR}..."
)
if not os.path.exists(GROUNDTRUTH_EXTRACT_DIR):
    os.makedirs(GROUNDTRUTH_EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(DOWNLOAD_PATH_ADDITIONAL_JSON_ZIP, 'r') as zip_ref:
        zip_ref.extractall(GROUNDTRUTH_EXTRACT_DIR)
    print("New JSON ground truth extraction complete!")

    # Determine the actual GT root directory
    # The new JSON zip likely contains 'Groundtruth' directly or files within.
    # We need to find the directory that contains 'Train' and 'Test' ground truth files.
    potential_gt_roots = [os.path.join(GROUNDTRUTH_EXTRACT_DIR, d) for d in os.listdir(GROUNDTRUTH_EXTRACT_DIR) if os.path.isdir(os.path.join(GROUNDTRUTH_EXTRACT_DIR, d))]
    
    found_gt_root = False
    for p_gt_root in potential_gt_roots:
        if os.path.exists(os.path.join(p_gt_root, 'Train')) or os.path.exists(os.path.join(p_gt_root, 'Test')):
            TOTAL_TEXT_GT_ROOT = p_gt_root
            found_gt_root = True
            break
    
    if not found_gt_root:
        # Fallback if 'Groundtruth' folder not found as expected
        print(f"Warning: Expected 'Train'/'Test' folders for GT within a top-level folder inside {GROUNDTRUTH_EXTRACT_DIR}. Falling back to base extract dir.")
        TOTAL_TEXT_GT_ROOT = GROUNDTRUTH_EXTRACT_DIR # Fallback
else:
    print("Ground truth already extracted, skipping extraction.")
    # If already extracted, re-determine the structure
    potential_gt_roots = [os.path.join(GROUNDTRUTH_EXTRACT_DIR, d) for d in os.listdir(GROUNDTRUTH_EXTRACT_DIR) if os.path.isdir(os.path.join(GROUNDTRUTH_EXTRACT_DIR, d))]
    found_gt_root = False
    for p_gt_root in potential_gt_roots:
        if os.path.exists(os.path.join(p_gt_root, 'Train')) or os.path.exists(os.path.join(p_gt_root, 'Test')):
            TOTAL_TEXT_GT_ROOT = p_gt_root
            found_gt_root = True
            break
    if not found_gt_root:
        TOTAL_TEXT_GT_ROOT = GROUNDTRUTH_EXTRACT_DIR # Fallback


# Ensure TOTAL_TEXT_GT_ROOT ends up pointing to the directory containing 'Train' and 'Test' GT files
if os.path.exists(os.path.join(TOTAL_TEXT_GT_ROOT, 'Groundtruth')):
    TOTAL_TEXT_GT_ROOT = os.path.join(TOTAL_TEXT_GT_ROOT, 'Groundtruth')
elif os.path.exists(os.path.join(GROUNDTRUTH_EXTRACT_DIR, 'Groundtruth')):
    TOTAL_TEXT_GT_ROOT = os.path.join(GROUNDTRUTH_EXTRACT_DIR, 'Groundtruth')
elif not (os.path.exists(os.path.join(TOTAL_TEXT_GT_ROOT, 'Train')) or os.path.exists(os.path.join(TOTAL_TEXT_GT_ROOT, 'Test'))):
    print("Error: Could not find 'Groundtruth' or directly 'Train'/'Test' directories within the extracted ground truth files.")
    # You might want to exit or raise an error here if the path is critical.

print(f"Ground truth dataset root set to: {TOTAL_TEXT_GT_ROOT}")
print(f"Contents of ground truth root ({TOTAL_TEXT_GT_ROOT}): {os.listdir(TOTAL_TEXT_GT_ROOT) if os.path.exists(TOTAL_TEXT_GT_ROOT) else 'Path not found'}")
if os.path.exists(os.path.join(TOTAL_TEXT_GT_ROOT, "Train")):
    print(
        f"Contents of {os.path.join(TOTAL_TEXT_GT_ROOT, 'Train')}: {os.listdir(os.path.join(TOTAL_TEXT_GT_ROOT, 'Train'))[:5]}..."
    )


# === Directory Paths ===
# Use the dynamically determined roots
img_base_dir = TOTAL_TEXT_IMAGE_ROOT
gt_base_dir = TOTAL_TEXT_GT_ROOT
output_base_dir = "/content/cropped"

os.makedirs(output_base_dir + "/Train", exist_ok=True)
os.makedirs(output_base_dir + "/Test", exist_ok=True)

# === Crop Utility ===
def crop_polygon(img, x_coords, y_coords):
    # Ensure points are integers
    pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
    
    # Handle cases where points might be out of image bounds or invalid
    if pts.size == 0:
        return None # Or a blank image, depending on desired behavior

    # Get bounding rectangle
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect

    # Ensure crop dimensions are positive
    if w <= 0 or h <= 0:
        return None

    # Crop the rectangular region
    cropped_rect = img[y:y+h, x:x+w].copy()

    # Shift polygon points to be relative to the cropped rectangle's origin
    pts_shifted = pts - np.array([x, y])

    # Create a mask for the polygon within the cropped rectangle
    mask = np.zeros(cropped_rect.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts_shifted], 255)

    # Apply the mask
    cropped_polygon = cv2.bitwise_and(cropped_rect, cropped_rect, mask=mask)
    return cropped_polygon

# === Main Processing Function ===
def process_set(split):
    image_dir = os.path.join(img_base_dir, split)
    gt_dir = os.path.join(gt_base_dir, split)
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg"))) # Use os.path.join for robust pathing
    all_entries = []

    if not os.path.exists(image_dir):
        print(f"Image directory for {split} not found: {image_dir}")
        return
    if not os.path.exists(gt_dir):
        print(f"Ground truth directory for {split} not found: {gt_dir}")
        return

    print(f"\nProcessing {split} set from images: {image_dir} and GT: {gt_dir}")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        # Ground truth file name pattern: poly_gt_{image_name_without_ext}.txt
        gt_filename = f"poly_gt_{img_name.replace('.jpg', '.txt')}"
        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            # Attempt a common alternative if original doesn't work (e.g., if case differs)
            # This specific dataset typically uses 'poly_gt_<image_name>.txt'
            # Check for lowercased image name in GT file (less common but good to check)
            gt_filename_lower = f"poly_gt_{img_name.lower().replace('.jpg', '.txt')}"
            gt_path_lower = os.path.join(gt_dir, gt_filename_lower)
            if os.path.exists(gt_path_lower):
                gt_path = gt_path_lower
            else:
                print(f"Ground truth not found for {img_name} at {gt_path} or {gt_path_lower}, skipping...")
                continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Can't read image {img_path}. Skipping.")
            continue

        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            try:
                line = line.strip()
                # Regex to extract x, y coordinates and transcription text
                # It handles variations like 'u'' or double/single quotes around text.
                # It also handles potential spaces inside the coordinate lists (e.g., '10 20 30')
                match = re.search(r'x:\s*\[\[\s*([\d\s,]+?)\s*\]\]\s*,\s*y:\s*\[\[\s*([\d\s,]+?)\s*\]\](?:.*?)transcriptions:\s*\[\s*(?:u\'|\'|\")([^\]]*?)(?:u\'|\'|\")\s*\]', line)
                
                if not match:
                    # Try another regex if the first one fails, more general
                    match = re.search(r'x:\s*\[\[(.*?)\]\],\s*y:\s*\[\[(.*?)\]\].*transcriptions:\s*\[\s*["\'](.*?)["\']\s*\]', line)
                    if not match:
                        print(f"Warning: No valid data found in line {idx+1} of {gt_path}. Skipping.")
                        continue

                x_str, y_str, text = match.groups()
                
                # Convert coordinate strings to lists of integers, handling spaces and commas
                x = list(map(int, re.findall(r'\d+', x_str)))
                y = list(map(int, re.findall(r'\d+', y_str)))

                text = text.strip() # Clean up transcription text

                if text in ['#', '###']:
                    continue

                cropped_img = crop_polygon(img, x, y)
                
                if cropped_img is None:
                    print(f"Warning: Could not crop polygon for {img_name}, line {idx+1}. Skipping.")
                    continue

                out_name = f"{img_name[:-4]}_{idx}.png"
                out_path = os.path.join(output_base_dir, split, out_name)
                
                # Check if the cropped image is empty (e.g., all black or very small)
                # This can happen if the polygon is degenerate or outside image bounds entirely
                if cropped_img.size == 0 or np.all(cropped_img == 0):
                    print(f"Warning: Cropped image for {img_name}, line {idx+1} is empty/all black. Skipping save.")
                    continue

                cv2.imwrite(out_path, cropped_img)
                all_entries.append({"filename": out_path, "text": text})
            except Exception as e:
                print(f"Error processing {img_name}, line {idx+1}: {e}. Line content: '{line}'")
                continue

    # Save CSV
    df = pd.DataFrame(all_entries)
    csv_path = os.path.join(output_base_dir, f"{split.lower()}_labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Processed {len(all_entries)} entries for {split}. CSV saved to {csv_path}")

# === Run for Train and Test ===
process_set("Train")
process_set("Test")

# === List sample results ===
print("\nSample cropped images and CSV entries (first 5 of each split if available):")

# List a few cropped images
train_cropped_dir = os.path.join(output_base_dir, "Train")
test_cropped_dir = os.path.join(output_base_dir, "Test")

train_samples = os.listdir(train_cropped_dir)
test_samples = os.listdir(test_cropped_dir)

print(f"\nTrain Cropped Samples ({len(train_samples)} total):")
for i, fname in enumerate(train_samples[:5]):
    print(os.path.join(train_cropped_dir, fname))

print(f"\nTest Cropped Samples ({len(test_samples)} total):")
for i, fname in enumerate(test_samples[:5]):
    print(os.path.join(test_cropped_dir, fname))

# Read and print sample from CSVs
train_csv_path = os.path.join(output_base_dir, "train_labels.csv")
test_csv_path = os.path.join(output_base_dir, "test_labels.csv")

if os.path.exists(train_csv_path):
    print(f"\nSample from {train_csv_path}:")
    print(pd.read_csv(train_csv_path).head())

if os.path.exists(test_csv_path):
    print(f"\nSample from {test_csv_path}:")
    print(pd.read_csv(test_csv_path).head())