#!/usr/bin/env python3
import torch
from ultralytics import YOLO

def main():
    # ✅ Check MPS availability (Apple Silicon GPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # ✅ Load a YOLOv8 Nano model (good for M2, fast and light)
    model = YOLO("yolov8n.pt")  # you can try yolov8s.pt for slightly better accuracy

    # ✅ Train
    model.train(
        data="yolo_dataset/dataset.yaml",  # path to dataset.yaml
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        augment=True
    )

    print("\n✅ Training complete!")
    print("Check results in: runs/detect/train/")
    print("Best model weights: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()

