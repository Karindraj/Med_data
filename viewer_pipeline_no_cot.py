#!/usr/bin/env python3
import os
import cv2
import pandas as pd
import json
import re
import traceback
from langchain_community.llms import Ollama

# === CONFIG ===
image_dir = "/Users/indrajitkar/Downloads/Abdominal/test"
csv_path = os.path.join(image_dir, "_annotations.csv")
overlay_path = "current_overlay.jpg"
model_name = "amsaravi/medgemma-4b-it:q6"

# Initialize Ollama MedGemma model
llm = Ollama(model=model_name)

def safe_parse_json_from_text(text):
    """Try to extract JSON from a text blob and parse it.
    Returns parsed dict or None."""
    # First, try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find the first JSON object in the text
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        candidate = m.group(1)
        # Try balancing braces if truncated (basic attempt)
        # Attempt parse; if fails, progressively remove trailing characters until valid
        for trim in range(0, 50):
            try:
                return json.loads(candidate[:-trim] if trim else candidate)
            except Exception:
                continue

    return None

def analyze_with_medgemma(image_path, original_filename):
    """Send overlay image to MedGemma and save structured analysis.
       NOTE: This function explicitly requests NO chain-of-thought, only a structured JSON + short rationale.
    """
    if not os.path.exists(image_path):
        print("❌ No overlay image found")
        return

    # Compose the safe, structured instruction. We explicitly forbid chain-of-thought.
    instruction_text = (
        "You are a medical imaging assistant. "
        "Do NOT provide internal chain-of-thought or step-by-step private deliberation. "
        "Provide a concise, professional analysis of the annotated abdominal scan. "
        "Return ONLY a JSON object (no extra prose) with these fields:\n"
        "  - summary: one-sentence high-level summary\n"
        "  - findings: a list of brief finding strings (what is visible / abnormal)\n"
        "  - confidence: integer 0-100 indicating overall confidence\n"
        "  - recommended_actions: a list of suggested next steps (diagnostic/follow-up)\n"
        "  - brief_rationale: 1-3 short sentences explaining the key reason for the findings\n"
        "If you cannot provide confident clinical interpretation, set findings to an empty list and "
        "explain limitations in brief_rationale. KEEP the brief_rationale short (<=3 sentences)."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text},
                {"type": "image_url", "image_url": f"file://{os.path.abspath(image_path)}"}
            ],
        }
    ]

    print("\n⚡ Sending image to MedGemma for structured analysis (no chain-of-thought)...")
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print("❌ Error invoking MedGemma/Ollama:")
        traceback.print_exc()
        return

    # Convert response to string for saving/parsing
    if isinstance(response, str):
        resp_text = response
    else:
        # Some wrappers return objects; convert to string
        resp_text = str(response)

    # Save raw response to .txt
    base_name = os.path.splitext(original_filename)[0]
    txt_path = os.path.join(image_dir, f"{base_name}_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(resp_text)
    print(f"💾 Raw analysis saved to: {txt_path}")

    # Try to parse JSON out of the response
    parsed = safe_parse_json_from_text(resp_text)
    if parsed is not None:
        json_path = os.path.join(image_dir, f"{base_name}_analysis.json")
        try:
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(parsed, jf, indent=2, ensure_ascii=False)
            print(f"💾 Parsed JSON saved to: {json_path}")
            # Pretty print summary to terminal
            print("\n===== Parsed Structured Analysis =====")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
            print("======================================\n")
        except Exception as e:
            print("⚠️ Failed to save parsed JSON:", e)
    else:
        print("⚠️ Could not parse JSON from model response. See the raw analysis .txt file for details.")
        print("\n===== Raw Model Output Preview =====")
        print(resp_text[:2000])  # print a preview
        print("====================================\n")

def main():
    # Load annotations
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["filename", "xmin", "ymin", "xmax", "ymax", "class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    grouped = df.groupby("filename")
    group_names = list(grouped.groups.keys())
    total = len(group_names)

    # Iterate through images
    for idx, filename in enumerate(group_names, start=1):
        group = grouped.get_group(filename)
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {filename} (not found).")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not load {filename}.")
            continue

        print(f"\n👀 Showing {filename} ({idx}/{total})")

        # Draw bounding boxes
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax, label = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
                row["class"],
            )
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, str(label), (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay filename and index
        overlay_text = f"{filename}  ({idx}/{total})"
        cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Save overlay for analysis
        cv2.imwrite(overlay_path, img)

        # Show image
        cv2.imshow("Image Viewer (Press any key for next | 'a' = analyze | 'q' = quit)", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            analyze_with_medgemma(overlay_path, filename)
        # Any other key → just next image

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
