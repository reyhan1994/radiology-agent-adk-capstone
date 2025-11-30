# master_agent.py
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.coding_agent import CodingAgent
from agents.patient_context_agent import PatientContextAgent

class MasterAgent:
    """
    MasterAgent combines:
    - ImageAnalysisAgent: for CXR image classification
    - CodingAgent: for ICD-10 / CPT codes
    - PatientContextAgent: for optional patient info enrichment
    """
    def __init__(self, device=None):
        self.device = device
        self.image_agent = ImageAnalysisAgent(
            model_path="/content/drive/MyDrive/radiology_agent/models/chest_classifier.pt",
            device=self.device,
            threshold=0.45,   # confidence threshold         
        )
        self.coding_agent = CodingAgent()
        self.patient_agent = PatientContextAgent()

    def run(self, input_data):
        """
        input_data: dict with 'user_request' (image path) and optional patient_data
        """
        # Patient context (optional)
        patient_info = self.patient_agent.run(input_data.get("user_request"))

        # Image analysis
        img_result = self.image_agent.run(input_data)

        # ICD/CPT coding
        coding_result = self.coding_agent.run(img_result.get("pathology"))

        # Final report string
        confidence = img_result.get("confidence", "0%")
        final_report = f"Final Report: {img_result.get('pathology')} for patient {patient_info.get('name','Unknown')}. Confidence: {confidence}."

        return {
            "user_request": input_data.get("user_request"),
            "patient_data": patient_info,
            "analysis_findings": img_result,
            "coding_result": coding_result,
            "final_report": final_report,
            "memory_status": "Consolidation Successful"
        }


# Helper function to match previous interface
def build_master_agent(device=None):
    return MasterAgent(device=device)
