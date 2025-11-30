# agents/coding_agent.py

CODE_LOOKUP = {
    "Pneumothorax": {"ICD_10": "J93.9", "CPT": "71045"},
    "Atelectasis": {"ICD_10": "J98.11", "CPT": "71046"},
    "Normal": {"ICD_10": "Z00.00", "CPT": "71047"},
    "Pneumonia": {"ICD_10": "J18.9", "CPT": "71046"},
    "Other": {"ICD_10": "R69", "CPT": "71049"},
}

class CodingAgent:
    def __init__(self):
        pass

    def run(self, input_data):
        # input_data: could be final_report string or analysis_findings dict
        if isinstance(input_data, dict):
            pathology = input_data.get("pathology", "")
        else:
            s = str(input_data)
            if "Pneumothorax" in s:
                pathology = "Pneumothorax"
            elif "Atelectasis" in s:
                pathology = "Atelectasis"
            elif "Pneumonia" in s:
                pathology = "Pneumonia"
            elif "Normal" in s:
                pathology = "Normal"
            else:
                pathology = "Other"

        # match to CODE_LOOKUP
        for key in CODE_LOOKUP:
            if key.lower() == pathology.lower():
                return CODE_LOOKUP[key]

        return {"ICD_10": "", "CPT": ""}
