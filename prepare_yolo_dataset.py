#!/usr/bin/env python3
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# === CONFIG ===
image_dir = "/Users/indrajitkar/Downloads/Abdominal/train"  # folder with images
csv_path = os.path.join(image_dir, "_annotations.csv")      # annotation file
output_dir = "yolo_dataset"                                # where YOLO dataset will be saved
train_ratio = 0.8

# === CREATE FOLDERS ===
os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)

# === LOAD ANNOTATIONS ===
df = pd.read_csv(csv_path)

required_cols = ["filename", "xmin", "ymin", "xmax", "ymax", "class"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV missing required column: {col}")

# Get unique classes & assign IDs
classes = sorted(df["class"].unique())
class_to_id = {c: i for i, c in enumerate(classes)}

print("Classes mapping:", class_to_id)

# === SPLIT TRAIN/VAL ===
unique_files = df["filename"].unique()
train_files, val_files = train_test_split(unique_files, train_size=train_ratio, random_state=42)

def convert_and_save(image_set, subset_name):
    for filename in image_set:
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {filename}, image not found.")
            continue

        # Load image to get width & height
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not load {filename}.")
            continue
        h, w, _ = img.shape

        # Copy image
        out_img_path = os.path.join(output_dir, "images", subset_name, filename)
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        cv2.imwrite(out_img_path, img)

        # Write label file
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(output_dir, "labels", subset_name, label_filename)

        with open(label_path, "w") as f:
            for _, row in df[df["filename"] == filename].iterrows():
                xmin, ymin, xmax, ymax, label = row[["xmin", "ymin", "xmax", "ymax", "class"]]
                xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])
                class_id = class_to_id[label]

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

# Convert train and val
convert_and_save(train_files, "train")
convert_and_save(val_files, "val")

# === CREATE dataset.yaml ===
yaml_path = os.path.join(output_dir, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {os.path.abspath(output_dir)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"nc: {len(classes)}\n")
    f.write("names: " + str(classes) + "\n")

print("\n✅ Conversion complete! YOLOv8 dataset saved in:", output_dir)
print("📄 dataset.yaml created for training")
