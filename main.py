# main.py
import argparse
import os
import csv
from master_agent import build_master_agent
from agents.patient_context_agent import PatientContextAgent

try:
    from google.colab import files
    COLAB = True
except ImportError:
    COLAB = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}

def list_images(folder):
    names = []
    for fn in sorted(os.listdir(folder)):
        _, ext = os.path.splitext(fn.lower())
        if ext in IMAGE_EXTS:
            names.append(fn)
    return names

def extract_row(image_name, artifacts):
    analysis = artifacts.get("analysis_findings") or {}
    coding = artifacts.get("coding_result") or {}
    return {
        "image": image_name,
        "analysis_pathology": analysis.get("pathology", ""),
        "analysis_confidence": analysis.get("confidence", ""),
        "final_report": artifacts.get("final_report", ""),
        "ICD_10": coding.get("ICD_10", "") or coding.get("ICD_10_Code", ""),
        "CPT": coding.get("CPT", "") or coding.get("CPT_Code", ""),
        "memory_status": artifacts.get("memory_status", ""),
    }

def process_images(input_folder, output_csv, auto_download=True):
    imgs = list_images(input_folder)
    print("Found", len(imgs), "images")

    master = build_master_agent()
    pat_agent = PatientContextAgent()

    rows = []
    for im in imgs:
        path = os.path.join(input_folder, im)
        print("Processing:", path)
        initial = {"user_request": path, "patient_data": pat_agent.run(path)}
        artifacts = master.run(initial)
        print("-> artifacts returned:", artifacts)
        rows.append(extract_row(im, artifacts))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_csv}")

    if COLAB and auto_download:
        try:
            files.download(output_csv)
        except Exception as e:
            print("Download failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder with images")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    args = parser.parse_args()

    process_images(args.input, args.output)
