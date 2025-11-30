# master_agent.py
import os
import inspect
import asyncio
from adk.agent import SequentialAgent, StepResult
from adk.step import Step


from agents.image_analysis_agent import ImageAnalysisAgent
from agents.report_generation_agent import ReportGenerationAgent
from agents.coding_agent import CodingAgent
from agents.memory_agent import MemoryAgent


# build a simple SequentialAgent that maps steps to agent.run outputs


def build_master_agent():
img_agent = ImageAnalysisAgent()
report_agent = ReportGenerationAgent()
coding_agent = CodingAgent()
memory_agent = MemoryAgent()


steps = [
Step("run_image_analysis", img_agent, "user_request", "analysis_findings"),
Step("generate_final_report", report_agent, ["patient_data", "analysis_findings"], "final_report"),
Step("run_pathology_coding", coding_agent, "analysis_findings", "coding_result"),
Step("store_long_term_memory", memory_agent, ["patient_data", "analysis_findings", "final_report", "coding_result"], "memory_status"),
]


return SequentialAgent(steps)
