# master_agent.py
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.coding_agent import CodingAgent
from agents.patient_context_agent import PatientContextAgent

class MasterAgent:
    """
    MasterAgent combines:
    - ImageAnalysisAgent: for CXR classification
    - CodingAgent: ICD-10 / CPT codes
    - PatientContextAgent: optional demographic enrichment
    """
    def __init__(self, device=None):
        self.device = device

       
        self.image_agent = ImageAnalysisAgent(
            model_path="models/chest_classifier.pt",
            device=self.device,
            threshold=0.45,
            tta=True
        )

        self.coding_agent = CodingAgent()
        self.patient_agent = PatientContextAgent()

    def run(self, input_data):
        """
        input_data: {"user_request": <image_path>}
        """
        img_path = input_data.get("user_request")

        # Patient context
        patient_info = self.patient_agent.run(img_path)

        # CXR image classification
        img_result = self.image_agent.run(input_data)

        # ICD/CPT code generation
        coding_result = self.coding_agent.run(img_result.get("pathology"))

        # New: confidence is float (not string)
        conf = img_result.get("confidence", 0.0)

        final_report = (
            f"Final Report: {img_result.get('pathology')} "
            f"for patient {patient_info.get('name','Unknown')}. "
            f"Confidence: {conf:.2f}."
        )

        return {
            "user_request": img_path,
            "patient_data": patient_info,
            "analysis_findings": img_result,
            "coding_result": coding_result,
            "final_report": final_report,
            "memory_status": "Consolidation Successful"
        }


def build_master_agent(device=None):
    return MasterAgent(device=device)
