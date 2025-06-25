import os
import pandas as pd
import shutil

# -------- Config --------
CONFIGS = {
    "train": {
        "csv": "cropped/train_labels.csv",
        "image_dir": "cropped/Train",
        "max_samples": 1000
    },
    "test": {
        "csv": "cropped/test_labels.csv",
        "image_dir": "cropped/Test",
        "max_samples": 200
    }
}
DEST_ROOT = "real_ocr_data"

# -------- Setup Base Path --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Process Each Dataset (Train/Test) Separately --------
for split, config in CONFIGS.items():
    csv_path = os.path.join(BASE_DIR, config["csv"])
    image_src_dir = os.path.join(BASE_DIR, config["image_dir"])
    dest_image_dir = os.path.join(BASE_DIR, DEST_ROOT, split, "images")
    dest_label_txt = os.path.join(BASE_DIR, DEST_ROOT, split, "labels.txt")

    # -------- Create output folders if needed --------
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dest_label_txt), exist_ok=True)

    # -------- Load and limit CSV --------
    if not os.path.exists(csv_path):
        print(f"❌ Missing CSV: {csv_path}")
        continue

    df = pd.read_csv(csv_path, header=None, names=["path", "label"])
    df = df.head(config["max_samples"])

    # -------- Copy images and write labels --------
    with open(dest_label_txt, 'w') as label_file:
        for _, row in df.iterrows():
            filename = os.path.basename(row["path"].strip())
            label = row["label"].strip()

            src_image_path = os.path.join(image_src_dir, filename)
            dest_image_path = os.path.join(dest_image_dir, filename)

            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dest_image_path)
                label_file.write(f"{filename} {label}\n")
            else:
                print(f"⚠️ Missing image: {src_image_path}")

    print(f"\n✅ {split.upper()} set processed: {len(df)} images")
    print(f"   → Labels: {dest_label_txt}")
    print(f"   → Images: {dest_image_dir}")