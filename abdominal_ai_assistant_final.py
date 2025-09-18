#!/usr/bin/env python3
import os
import cv2
import json
import re
import traceback
import torch
import pandas as pd
from ultralytics import YOLO
from langchain_community.llms import Ollama
from datetime import datetime

# === CONFIG ===
image_dir = "/Users/indrajitkar/Downloads/Abdominal/test"
weights_path = "/Users/indrajitkar/Downloads/runs/detect/train/weights/best.pt"
csv_path = os.path.join(image_dir, "_annotations.csv")

# YOLO model
device = "mps" if torch.backends.mps.is_available() else "cpu"
yolo_model = YOLO(weights_path)

# MedGemma model
llm = Ollama(model="amsaravi/medgemma-4b-it:q6")

def safe_parse_json_from_text(text):
    """Try to parse JSON out of model text output, tolerant to extra text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        candidate = m.group(1)
        for trim in range(0, 50):
            try:
                return json.loads(candidate[:-trim] if trim else candidate)
            except Exception:
                continue
    return None

def append_to_conversation(base_name, section, content):
    """Append a new section with timestamp to the conversation log file."""
    conv_path = os.path.join(image_dir, f"{base_name}_conversation.txt")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(conv_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{section}] {ts}\n")
        if isinstance(content, dict):
            f.write(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
        else:
            f.write(str(content).strip() + "\n")
        f.write("---\n")
    return conv_path

def analyze_with_medgemma(image_path, original_filename):
    """Send overlay image to MedGemma and save structured analysis."""
    if not os.path.exists(image_path):
        print("❌ No overlay image found")
        return

    instruction_text = (
        "You are a medical imaging assistant. "
        "Do provide internal chain-of-thought or private reasoning. "
        "Provide a concise, professional analysis of the annotated abdominal scan. "
        "Return ONLY a JSON object (no extra prose) with these fields:\n"
        "  - summary\n"
        "  - findings\n"
        "  - confidence\n"
        "  - recommended_actions\n"
        "  - brief_rationale\n"
        "If you cannot provide confident clinical interpretation, set findings to an empty list and "
        "explain limitations in brief_rationale. KEEP the brief_rationale short (<=3 sentences)."
    )

    messages = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction_text},
             {"type": "image_url", "image_url": f"file://{os.path.abspath(image_path)}"}
         ]}
    ]

    print("\n⚡ Sending image to MedGemma for structured analysis...")
    try:
        response = llm.invoke(messages)
    except Exception:
        traceback.print_exc()
        return

    resp_text = response if isinstance(response, str) else str(response)
    base_name = os.path.splitext(original_filename)[0]

    # Save raw response in conversation log
    conv_path = append_to_conversation(base_name, "ANALYSIS RAW RESPONSE", resp_text)

    # Try parse JSON
    parsed = safe_parse_json_from_text(resp_text)
    if parsed:
        # Save structured JSON separately
        json_path = os.path.join(image_dir, f"{base_name}_analysis.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2, ensure_ascii=False)

        # Save parsed JSON into conversation log
        append_to_conversation(base_name, "ANALYSIS PARSED JSON", parsed)

        print(f"💾 Parsed JSON saved to: {json_path}")
        print(f"💾 Conversation updated: {conv_path}")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print("⚠️ Could not parse JSON. See conversation log.")

def followup_with_medgemma(image_path, question, original_filename):
    """Free-form Q&A follow-up with MedGemma (text only)."""
    context = (
        f"You are continuing Q&A about the previously analyzed abdominal scan "
        f"(image file: {os.path.basename(image_path)}). "
    )
    messages = [{"role": "user", "content": context + "\n\nQuestion: " + question}]
    
    try:
        print("\n⚡ Sending follow-up question to MedGemma...")
        resp = llm.invoke(messages)
        resp_text = resp if isinstance(resp, str) else str(resp)

        print("\n===== Follow-up Response =====")
        print(resp_text)
        print("=============================\n")

        base_name = os.path.splitext(original_filename)[0]
        conv_path = append_to_conversation(base_name, f"FOLLOW-UP Q: {question}", f"A: {resp_text.strip()}")

        print(f"💾 Conversation updated: {conv_path}")

    except Exception:
        traceback.print_exc()

def draw_boxes_from_csv(img, filename):
    """Draw bounding boxes for a given image filename using _annotations.csv."""
    if not os.path.exists(csv_path):
        print("⚠️ No annotation CSV found.")
        return img, False

    df = pd.read_csv(csv_path)
    rows = df[df["filename"] == filename]

    if rows.empty:
        print("⚠️ No boxes found in CSV for", filename)
        return img, False

    for _, row in rows.iterrows():
        try:
            xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            label = str(row.get("class", "CSV_box"))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(img, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as e:
            print("⚠️ Error drawing CSV box:", e)
    return img, True

def main():
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    total = len(image_files)
    for idx, filename in enumerate(image_files, start=1):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not load {filename}.")
            continue

        print(f"\n👀 Processing {filename} ({idx}/{total})")

        # Build unique overlay path for this image
        base_name = os.path.splitext(filename)[0]
        overlay_path = os.path.join(image_dir, f"{base_name}_overlay.jpg")

        # Run YOLO detection
        results = yolo_model(img_path, device=device, conf=0.25, verbose=False)

        if results[0].boxes and len(results[0].boxes) > 0:
            annotated = results[0].plot()
            cv2.putText(annotated, filename, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(overlay_path, annotated)
            print(f"✅ YOLO boxes drawn. Saved overlay: {overlay_path}")
        else:
            print("⚠️ YOLO found no boxes. Using CSV annotations instead...")
            annotated, ok = draw_boxes_from_csv(img.copy(), filename)
            cv2.putText(annotated, filename, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(overlay_path, annotated)
            if ok:
                print(f"✅ CSV boxes drawn. Saved overlay: {overlay_path}")
            else:
                print("⚠️ No bounding boxes available (YOLO or CSV).")

        # --- Interactive loop ---
        while True:
            cv2.imshow("YOLOv8 Viewer (Keys: 'a'=analyze | 'f'=follow-up Q | 'n'=next | 'q'=quit)", annotated)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('a'):
                analyze_with_medgemma(overlay_path, filename)

            elif key == ord('f'):
                custom_q = input("\n❓ Enter your follow-up question for MedGemma: ").strip()
                if custom_q:
                    followup_with_medgemma(overlay_path, custom_q, filename)

            elif key == ord('n'):
                break  # move to next image, overlay file is kept

            elif key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
