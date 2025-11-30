# main.py
import argparse
import os
import csv
from master_agent import build_master_agent
from agents.patient_context_agent import PatientContextAgent


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




def main(input_folder, output_csv):
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
rows.append(extract_ro
