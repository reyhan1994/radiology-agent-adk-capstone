# agents/coding_agent.py


CODE_LOOKUP = {
"Pneumothorax": {"ICD_10": "J93.9", "CPT": "71045"},
"Atelectasis": {"ICD_10": "J98.11", "CPT": "71046"},
"Normal": {"ICD_10": "Z00.00", "CPT": "71047"},
}


class CodingAgent:
def __init__(self):
pass


def run(self, input_data):
# input_data: could be final_report string or analysis_findings dict
if isinstance(input_data, dict):
pathology = input_data.get("pathology", "")
else:
# try crude string parse
s = str(input_data)
pathology = "Pneumothorax" if "Pneumothorax" in s else ("Atelectasis" if "Atelectasis" in s else "Normal")


# match available prefixes
for key in CODE_LOOKUP:
if key.lower() in pathology.lower():
return CODE_LOOKUP[key]


return {"ICD_10": "", "CPT": ""}
