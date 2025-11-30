# agents/report_generation_agent.py
from datetime import datetime


class ReportGenerationAgent:
def __init__(self):
pass


def run(self, input_data):
# input_data: dict with patient_data and analysis_findings
patient = input_data.get("patient_data") if isinstance(input_data, dict) else None
findings = input_data.get("analysis_findings") if isinstance(input_data, dict) else None


patient_name = "Unknown"
if isinstance(patient, dict):
patient_name = patient.get("name") or patient.get("patient_id") or "Unknown"


if not isinstance(findings, dict):
return "Final Report: No findings."


pathology = findings.get("pathology", "unspecified")
confidence = findings.get("confidence", "0")


report = (
f"Final Report: {pathology} for patient {patient_name}."
f" Confidence: {confidence}."
f" Generated: {datetime.utcnow().isoformat()}Z"
)
return report
