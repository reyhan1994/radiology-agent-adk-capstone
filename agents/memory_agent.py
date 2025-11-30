# agents/memory_agent.py
import os
import json
from datetime import datetime


DB_PATH = os.path.join("memory", "patient_db.json")


class MemoryAgent:
def __init__(self, db_path=DB_PATH):
self.db_path = db_path
os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
if not os.path.exists(self.db_path):
with open(self.db_path, "w", encoding="utf-8") as f:
json.dump({}, f)


def run(self, input_data):
# input_data expected: dict with patient_id, final_report, analysis_findings, coding_result
if not isinstance(input_data, dict):
return "Invalid input for memory agent"


patient_id = "unknown"
if isinstance(input_data.get("patient_data"), dict):
patient_id = input_data["patient_data"].get("patient_id") or input_data["patient_data"].get("name")
elif input_data.get("user_request"):
patient_id = input_data.get("user_request")


entry = {
"timestamp": datetime.utcnow().isoformat() + "Z",
"analysis_findings": input_data.get("analysis_findings"),
"final_report": input_data.get("final_report"),
"coding_result": input_data.get("coding_result"),
}


# atomic append/update
with open(self.db_path, "r+", encoding="utf-8") as f:
try:
data = json.load(f)
except Exception:
data = {}
if patient_id not in data:
data[patient_id] = []
data[patient_id].append(entry)
f.seek(0)
json.dump(data, f, indent=2)
f.truncate()


return "Consolidation Successful"
